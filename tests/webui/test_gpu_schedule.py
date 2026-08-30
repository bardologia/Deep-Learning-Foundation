"""Tests for the GPU schedule that resizes live job pools by time of day.

Covers weekday/night/weekend window arithmetic, phase transitions that grow and
shrink a running job's GPU pool, withholding of devices held by other users, the
greedy reclaim sweep, charity donations after a guard bounce, and settings
persistence and validation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from gpu_schedule import GpuSchedule, NightWindow, WeekWindow
from web_logger   import WebLogger

from tests.webui.conftest import StubPaths

FRIDAY_MORNING   = datetime(2026, 7, 17, 9, 0)
FRIDAY_EVENING   = datetime(2026, 7, 17, 18, 30)
SATURDAY         = datetime(2026, 7, 18, 12, 0)
SUNDAY_NIGHT     = datetime(2026, 7, 19, 23, 30)
MONDAY_EARLY     = datetime(2026, 7, 20, 7, 30)
MONDAY_WORKDAY   = datetime(2026, 7, 20, 9, 0)
WEDNESDAY        = datetime(2026, 7, 22, 15, 0)
WEDNESDAY_NIGHT  = datetime(2026, 7, 22, 21, 0)
THURSDAY_EARLY   = datetime(2026, 7, 23, 2, 0)
THURSDAY_WORKDAY = datetime(2026, 7, 23, 9, 0)


class StubProcesses:
    """Stand-in process manager recording every GPU pool write.

    Attributes:
        jobs   : Job descriptors returned by list_jobs.
        pools  : Per-job GPU pool state keyed by job id.
        writes : Recorded (job_id, gpus) pairs applied through set_gpus.
        refuse : When true, every set_gpus call is rejected.
    """
    def __init__(self) -> None:
        """Starts with no jobs, no pools and an accepting set_gpus."""
        self.jobs   = []
        self.pools  = {}
        self.writes = []
        self.refuse = False

    def list_jobs(self) -> list[dict]:
        """Returns a copy of the registered job descriptors."""
        return list(self.jobs)

    def gpu_pool(self, job_id: str) -> dict:
        """Returns the pool state for a job, or an unsupported placeholder when unknown."""
        return self.pools.get(job_id, {"ok": True, "supported": False, "live": False})

    def set_gpus(self, job_id: str, gpus) -> dict:
        """Records a pool write and applies it unless the stub is set to refuse."""
        if self.refuse:
            return {"ok": False, "error": "job is not running"}

        self.writes.append((job_id, list(gpus)))
        self.pools[job_id]["gpus"] = list(gpus)
        return {"ok": True, "gpus": list(gpus), "parked": False}

    def add_running_fan_out(self, job_id: str) -> None:
        """Registers a running fan-out job holding a live pool of GPU 0."""
        self.jobs.append({"job_id": job_id, "status": "running", "script": "train_backbone"})
        self.pools[job_id] = {"ok": True, "supported": True, "live": True, "gpus": [0]}


class StubGuard:
    """Stand-in GPU guard holding bounced launch attempts.

    Attributes:
        bounced : Attempt records not yet drained by take_bounced.
    """
    def __init__(self) -> None:
        """Starts with an empty bounce queue."""
        self.bounced = []

    def bounce(self, gpu: int, user: str, at: datetime) -> None:
        """Appends a survived bounce record for a GPU, user and timestamp."""
        self.bounced.append({"gpu_index": gpu, "user": user, "status": "survived", "bounced_at": at.isoformat(timespec="seconds")})

    def take_bounced(self) -> list[dict]:
        """Returns the queued bounce records and empties the queue."""
        bounced      = list(self.bounced)
        self.bounced = []
        return bounced


class StubSystem:
    """Stand-in system monitor reporting per-GPU process owners.

    Attributes:
        user    : Name of the local user whose processes never block a pool.
        holders : Owner names occupying each GPU index.
        blind   : When true, occupancy is reported as unknown.
    """
    def __init__(self) -> None:
        """Starts with four unoccupied GPUs owned by nobody."""
        self.user    = "bard"
        self.holders = {}
        self.blind   = False

    def occupy(self, index: int, user: str) -> None:
        """Adds a process owner to the given GPU index."""
        self.holders.setdefault(index, []).append(user)

    def release(self, index: int) -> None:
        """Removes every process owner from the given GPU index."""
        self.holders.pop(index, None)

    def gpu_occupancy(self) -> list[dict] | None:
        """Returns per-GPU process listings for four devices, or None when blind."""
        if self.blind:
            return None

        return [
            {"index": index, "procs": [{"pid": 100 + index, "owner": owner} for owner in self.holders.get(index, [])]}
            for index in range(4)
        ]


@pytest.fixture
def schedule(tmp_path):
    """Returns an enabled GpuSchedule alongside its stub process manager and system monitor."""
    processes = StubProcesses()
    system    = StubSystem()
    schedule  = GpuSchedule(StubPaths(tmp_path), WebLogger(), processes, system, StubGuard())
    schedule.settings["enabled"] = True
    return schedule, processes, system


def test_default_window_spans_friday_evening_to_monday_morning():
    """The default week window covers Friday evening through Monday morning and excludes ordinary weekdays."""
    window = WeekWindow(**{key: GpuSchedule.DEFAULTS[key] for key in ("start_day", "start_hour", "end_day", "end_hour")})

    assert window.contains(FRIDAY_EVENING) is True
    assert window.contains(SATURDAY)       is True
    assert window.contains(SUNDAY_NIGHT)   is True
    assert window.contains(MONDAY_EARLY)   is True

    assert window.contains(FRIDAY_MORNING) is False
    assert window.contains(MONDAY_WORKDAY) is False
    assert window.contains(WEDNESDAY)      is False


def test_window_that_does_not_wrap_the_week():
    """A window whose start day precedes its end day contains only the days between them."""
    window = WeekWindow(start_day=1, start_hour=8, end_day=3, end_hour=8)

    assert window.contains(datetime(2026, 7, 21, 9, 0))  is True
    assert window.contains(datetime(2026, 7, 22, 23, 0)) is True
    assert window.contains(datetime(2026, 7, 23, 9, 0))  is False
    assert window.contains(datetime(2026, 7, 20, 9, 0))  is False


def test_window_boundaries_are_inclusive_at_the_start_and_exclusive_at_the_end():
    """The window includes its start hour and excludes its end hour."""
    window = WeekWindow(**{key: GpuSchedule.DEFAULTS[key] for key in ("start_day", "start_hour", "end_day", "end_hour")})

    assert window.contains(datetime(2026, 7, 17, 18, 0))  is True
    assert window.contains(datetime(2026, 7, 17, 17, 59)) is False
    assert window.contains(datetime(2026, 7, 20, 7, 59))  is True
    assert window.contains(datetime(2026, 7, 20, 8, 0))   is False


def test_default_night_window_spans_the_evening_to_the_next_morning():
    """The default night window wraps midnight, covering the evening start hour through the morning end hour exclusive."""
    window = NightWindow(**{key: GpuSchedule.DEFAULTS[key] for key in ("night_start_hour", "night_end_hour")})

    assert window.contains(WEDNESDAY_NIGHT)   is True
    assert window.contains(THURSDAY_EARLY)    is True
    assert window.contains(datetime(2026, 7, 22, 20, 0)) is True

    assert window.contains(WEDNESDAY)         is False
    assert window.contains(THURSDAY_WORKDAY)  is False
    assert window.contains(datetime(2026, 7, 22, 19, 59)) is False
    assert window.contains(datetime(2026, 7, 23, 8, 0))   is False


def test_phase_names_follow_the_window(schedule):
    """Phase resolution names weekend, weekday and night according to the configured windows."""
    scheduler, _processes, _system = schedule

    assert scheduler.phase(SATURDAY)         == "weekend"
    assert scheduler.phase(MONDAY_WORKDAY)   == "weekday"
    assert scheduler.phase(WEDNESDAY_NIGHT)  == "night"
    assert scheduler.phase(THURSDAY_EARLY)   == "night"


def test_the_weekend_window_outranks_the_night_window(schedule):
    """Times inside both windows resolve to weekend rather than night."""
    scheduler, _processes, _system = schedule

    assert scheduler.phase(SUNDAY_NIGHT)   == "weekend"
    assert scheduler.phase(FRIDAY_EVENING) == "weekend"


def test_transition_into_the_night_grows_the_pool_and_the_morning_shrinks_it(schedule):
    """Crossing into night widens the pool to all GPUs and the return to the workday narrows it to the weekday set."""
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)
    scheduler.tick(THURSDAY_EARLY)
    scheduler.tick(THURSDAY_WORKDAY)

    assert processes.writes == [("job1", [0, 1, 2, 3]), ("job1", [2, 3])]


def test_first_tick_records_the_phase_without_touching_the_launch_selection(schedule):
    """The first tick only records the current phase, leaving the pool the job launched with."""
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)

    assert processes.writes    == []
    assert scheduler.applied   == {"job1": {"phase": "weekday", "withheld": []}}


def test_transition_into_the_weekend_grows_the_pool(schedule):
    """Crossing from weekday into the weekend window widens the pool to the weekend GPU set."""
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(FRIDAY_EVENING)

    assert processes.writes == [("job1", [0, 1, 2, 3])]


def test_transition_back_to_the_week_shrinks_the_pool(schedule):
    """Leaving the weekend window narrows the pool to the weekday GPU set."""
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(SATURDAY)
    scheduler.tick(MONDAY_WORKDAY)

    assert processes.writes == [("job1", [2, 3])]


def test_ticks_inside_one_phase_never_rewrite_the_pool(schedule):
    """Repeated ticks within the same phase issue no pool writes."""
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(SATURDAY)
    scheduler.tick(SUNDAY_NIGHT)
    scheduler.tick(MONDAY_EARLY)

    assert processes.writes == []


def test_a_manual_resize_survives_until_the_next_transition(schedule):
    """A hand-edited pool is left alone until the next phase change overwrites it."""
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(SATURDAY)
    processes.pools["job1"]["gpus"] = [2]

    scheduler.tick(SUNDAY_NIGHT)
    assert processes.writes == []

    scheduler.tick(MONDAY_WORKDAY)
    assert processes.writes == [("job1", [2, 3])]


def test_a_gpu_held_by_another_user_is_left_out_of_the_pool(schedule):
    """A GPU running another user's process is withheld from the new pool and recorded as withheld."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                      == [("job1", [0, 2, 3])]
    assert scheduler.applied["job1"]["withheld"] == [1]


def test_our_own_processes_never_block_the_pool(schedule):
    """GPUs occupied by the local user are still handed to the job."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(2, "bard")
    system.occupy(3, "bard")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                      == [("job1", [0, 1, 2, 3])]
    assert scheduler.applied["job1"]["withheld"] == []


def test_an_unknown_occupancy_never_hands_out_a_gpu(schedule):
    """An unreadable occupancy probe suppresses the transition entirely rather than guessing."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    system.blind = True
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                      == []
    assert scheduler.applied["job1"]["phase"]    == "weekday"
    assert scheduler.applied["job1"]["withheld"] == []


def test_the_move_lands_once_the_probe_answers_again(schedule):
    """The deferred transition is applied on the next tick after the occupancy probe recovers."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    system.blind = True
    scheduler.tick(WEDNESDAY_NIGHT)

    system.blind = False
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                      == [("job1", [0, 2, 3])]
    assert scheduler.applied["job1"]["withheld"] == [1]


def test_charity_donates_nothing_while_the_occupancy_is_unknown(schedule):
    """Charity refuses to donate a GPU while the occupancy probe is blind."""
    scheduler, processes, system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague", WEDNESDAY)
    system.blind = True

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes            == []
    assert scheduler.nap_until is None


def test_a_pool_held_entirely_by_others_leaves_the_job_untouched(schedule):
    """When every candidate GPU is held by others, no write is issued and all indices are recorded as withheld."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    for index in (0, 1, 2, 3):
        system.occupy(index, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                      == []
    assert scheduler.applied["job1"]["withheld"] == [0, 1, 2, 3]


def test_a_blocked_shrink_stays_pending_until_the_weekday_pool_frees(schedule):
    """A shrink blocked by foreign processes stays pending across sweeps and lands once those GPUs free."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(SATURDAY)
    for index in (2, 3):
        system.occupy(index, "colleague")

    scheduler.tick(MONDAY_WORKDAY)

    assert processes.writes                      == []
    assert scheduler.applied["job1"]["phase"]    == "weekend"
    assert scheduler.applied["job1"]["withheld"] == [2, 3]

    scheduler.sweep(MONDAY_WORKDAY)
    assert processes.writes == []

    system.release(2)
    system.release(3)
    scheduler.tick(MONDAY_WORKDAY)

    assert processes.writes                      == [("job1", [2, 3])]
    assert scheduler.applied["job1"]["phase"]    == "weekday"
    assert scheduler.applied["job1"]["withheld"] == []


def test_a_refused_pool_write_leaves_the_transition_pending(schedule):
    """A rejected set_gpus call keeps the recorded phase unchanged so the transition retries later."""
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    processes.refuse = True
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                   == []
    assert scheduler.applied["job1"]["phase"] == "weekday"

    processes.refuse = False
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                   == [("job1", [0, 1, 2, 3])]
    assert scheduler.applied["job1"]["phase"] == "night"


def test_greedy_picks_up_a_withheld_gpu_once_it_frees(schedule):
    """The greedy sweep adds a previously withheld GPU back to the pool after its foreign process exits."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)
    assert processes.writes == [("job1", [0, 2, 3])]

    scheduler.sweep(WEDNESDAY_NIGHT)
    assert processes.writes == [("job1", [0, 2, 3])]

    system.release(1)
    scheduler.sweep(WEDNESDAY_NIGHT)

    assert processes.writes                      == [("job1", [0, 2, 3]), ("job1", [0, 1, 2, 3])]
    assert scheduler.applied["job1"]["withheld"] == []


def test_greedy_stops_sweeping_once_the_pool_is_whole(schedule):
    """Further sweeps issue no writes after the pool has reclaimed every withheld GPU."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)
    system.release(1)
    scheduler.sweep(WEDNESDAY_NIGHT)
    scheduler.sweep(WEDNESDAY_NIGHT)
    scheduler.sweep(WEDNESDAY_NIGHT)

    assert processes.writes == [("job1", [0, 2, 3]), ("job1", [0, 1, 2, 3])]


def test_greedy_off_never_reclaims_a_withheld_gpu(schedule):
    """With greedy disabled, a freed GPU is never added back to the pool."""
    scheduler, processes, system = schedule
    scheduler.settings["greedy"] = False
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)
    system.release(1)
    scheduler.sweep(WEDNESDAY_NIGHT)

    assert processes.writes == [("job1", [0, 2, 3])]


def test_greedy_adds_to_a_hand_shrunk_pool_without_undoing_the_resize(schedule):
    """The greedy sweep appends the freed GPU to a manually shrunk pool instead of restoring the full phase pool."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)
    processes.pools["job1"]["gpus"] = [0]

    system.release(1)
    scheduler.sweep(WEDNESDAY_NIGHT)

    assert processes.writes[-1] == ("job1", [0, 1])


def test_greedy_never_unparks_a_job_and_keeps_waiting_for_it(schedule):
    """A parked job keeps its withheld GPU pending until it holds a GPU again, then the sweep adds the freed one."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)
    processes.pools["job1"]["gpus"] = []

    system.release(1)
    scheduler.sweep(WEDNESDAY_NIGHT)

    assert processes.writes                      == [("job1", [0, 2, 3])]
    assert scheduler.applied["job1"]["withheld"] == [1]

    processes.pools["job1"]["gpus"] = [0]
    scheduler.sweep(WEDNESDAY_NIGHT)

    assert processes.writes[-1] == ("job1", [0, 1])


def test_charity_frees_the_gpu_a_bounced_attempt_hit(schedule):
    """A bounced attempt makes charity drop the contested GPU from the pool, start a nap and report the gift in the state payload."""
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague", WEDNESDAY)
    given = scheduler.charity(WEDNESDAY)
    state = scheduler.state()

    assert given                                 == ["job1"]
    assert processes.writes                      == [("job1", [0, 2, 3])]
    assert scheduler.applied["job1"]["withheld"] == [1]
    assert scheduler.nap_until                   == WEDNESDAY + GpuSchedule.CHARITY_NAP
    assert state["napping"]                is True
    assert state["gifted"]                 == [1]
    assert state["last_charity"]["gpu"]    == 1
    assert state["last_charity"]["users"]  == ["colleague"]
    assert state["last_charity"]["jobs"]   == ["job1"]


def test_charity_stays_quiet_while_any_gpu_is_still_free(schedule):
    """No donation happens when the machine still has a GPU outside every job pool."""
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague", WEDNESDAY)

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes             == []
    assert scheduler.nap_until          is None


def test_charity_never_leaves_a_job_without_a_gpu(schedule):
    """Charity declines to donate the last GPU a job holds."""
    scheduler, processes, system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [1]
    for index in (0, 2, 3):
        system.occupy(index, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague", WEDNESDAY)

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes             == []
    assert scheduler.nap_until          is None


def test_a_bounce_off_an_unshrinkable_pool_frees_a_gpu_from_the_biggest_one(schedule):
    """When the bounced GPU belongs to a single-GPU job, the donation is taken from the largest pool instead."""
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.add_running_fan_out("job2")
    processes.pools["job1"]["gpus"] = [0]
    processes.pools["job2"]["gpus"] = [1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(0, "colleague", WEDNESDAY)
    given = scheduler.charity(WEDNESDAY)

    assert given                                 == ["job2"]
    assert processes.writes                      == [("job2", [1, 2])]
    assert scheduler.applied["job2"]["withheld"] == [3]


def test_charity_off_leaves_the_bounce_queue_untouched(schedule):
    """With charity disabled the bounce stays queued and is honoured once charity is enabled."""
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague", WEDNESDAY)

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes             == []
    assert [attempt["gpu_index"] for attempt in scheduler.guard.bounced] == [1]

    scheduler.settings["charity"] = True

    assert scheduler.charity(WEDNESDAY) == ["job1"]
    assert processes.writes             == [("job1", [0, 2, 3])]


def test_a_bounce_older_than_the_ttl_is_discarded(schedule):
    """A bounce older than the TTL is dropped without triggering a donation."""
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague", WEDNESDAY - GpuSchedule.BOUNCE_TTL - timedelta(seconds=1))

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes             == []
    assert scheduler.guard.bounced      == []


def test_the_nap_blocks_greedy_until_it_ends_then_greedy_reclaims(schedule):
    """The charity nap suppresses greedy reclaim until it expires, after which the donated GPU returns."""
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague", WEDNESDAY)
    scheduler.charity(WEDNESDAY)

    scheduler.sweep(WEDNESDAY)
    assert processes.writes == [("job1", [0, 2, 3])]

    later = WEDNESDAY + GpuSchedule.CHARITY_NAP
    scheduler.charity(later)
    scheduler.sweep(later)

    assert scheduler.nap_until  is None
    assert processes.writes[-1] == ("job1", [0, 1, 2, 3])


def test_a_second_bounce_during_the_nap_gets_nothing(schedule):
    """A bounce arriving during the nap triggers no second donation and is discarded when the nap ends."""
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague", WEDNESDAY)
    scheduler.charity(WEDNESDAY)

    scheduler.guard.bounce(2, "rival", WEDNESDAY)

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes             == [("job1", [0, 2, 3])]
    assert [attempt["gpu_index"] for attempt in scheduler.guard.bounced] == [2]

    assert scheduler.charity(WEDNESDAY + GpuSchedule.CHARITY_NAP) == []
    assert scheduler.guard.bounced                                == []


def test_a_phase_cross_during_the_nap_keeps_the_gift_out(schedule):
    """A phase transition during the nap rebuilds the pool without the donated GPU."""
    scheduler, processes, system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [2, 3]
    system.occupy(0, "colleague")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(2, "colleague", WEDNESDAY)
    scheduler.charity(WEDNESDAY)
    assert processes.writes == [("job1", [3])]

    system.release(0)
    system.release(1)
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes[-1]                  == ("job1", [0, 1, 3])
    assert scheduler.applied["job1"]["withheld"] == [2]


def test_turning_charity_off_ends_the_nap(schedule):
    """Disabling charity through update clears the nap deadline and the gifted set."""
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague", WEDNESDAY)
    scheduler.charity(WEDNESDAY)
    assert scheduler.nap_until is not None

    result = scheduler.update({**GpuSchedule.DEFAULTS, "enabled": True, "charity": False})

    assert result["ok"]         is True
    assert scheduler.nap_until  is None
    assert scheduler.gifted     == set()


def test_a_disabled_schedule_sweeps_nothing(schedule):
    """A disabled schedule performs no greedy reclaim."""
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)
    scheduler.settings["enabled"] = False

    system.release(1)
    scheduler.sweep(WEDNESDAY_NIGHT)

    assert processes.writes == [("job1", [0, 2, 3])]


def test_a_disabled_schedule_tracks_phases_without_writing(schedule):
    """A disabled schedule still records phase changes but issues no pool writes."""
    scheduler, processes, _system = schedule
    scheduler.settings["enabled"] = False
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(SATURDAY)

    assert processes.writes  == []
    assert scheduler.applied == {"job1": {"phase": "weekend", "withheld": []}}


def test_jobs_without_a_live_pool_are_ignored(schedule):
    """Jobs whose pool is not live are never tracked or resized."""
    scheduler, processes, _system = schedule
    processes.jobs.append({"job_id": "single", "status": "running", "script": "train_backbone"})

    scheduler.tick(WEDNESDAY)
    scheduler.tick(SATURDAY)

    assert processes.writes  == []
    assert scheduler.applied == {}


def test_finished_jobs_are_forgotten(schedule):
    """A job that stops running is dropped from the applied-phase bookkeeping."""
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    assert scheduler.applied == {"job1": {"phase": "weekday", "withheld": []}}

    processes.jobs[0]["status"] = "finished"
    scheduler.tick(SATURDAY)

    assert scheduler.applied == {}


def test_update_persists_and_reloads(tmp_path):
    """Accepted settings are written to disk and restored by a freshly constructed schedule."""
    processes = StubProcesses()
    payload   = {**GpuSchedule.DEFAULTS, "enabled": True, "charity": True, "weekday_gpus": [1], "night_gpus": [1, 2], "weekend_gpus": [0, 1, 2]}

    scheduler = GpuSchedule(StubPaths(tmp_path), WebLogger(), processes, StubSystem(), StubGuard())
    result    = scheduler.update(payload)

    assert result["ok"] is True
    assert json.loads(scheduler.path.read_text())["weekday_gpus"] == [1]

    reloaded = GpuSchedule(StubPaths(tmp_path), WebLogger(), processes, StubSystem(), StubGuard())

    assert reloaded.settings["enabled"]      is True
    assert reloaded.settings["charity"]      is True
    assert reloaded.settings["weekday_gpus"] == [1]
    assert reloaded.settings["night_gpus"]   == [1, 2]
    assert reloaded.settings["weekend_gpus"] == [0, 1, 2]


def test_state_reports_the_pool_that_applies_right_now(schedule):
    """The state payload names the current phase, its GPU set and the human-readable window descriptions."""
    scheduler, _processes, _system = schedule

    state = scheduler.state()

    assert state["phase"] in ("weekday", "night", "weekend")
    assert state["gpus_now"] == state[f"{state['phase']}_gpus"]
    assert "friday 18:00 to monday 08:00" == state["window"]
    assert "20:00 to 08:00 every day"     == state["night_window"]


@pytest.mark.parametrize("payload, reason", [
    ({"enabled": "yes"},                    "true or false"),
    ({"greedy": "sure"},                    "true or false"),
    ({"charity": "why not"},                "true or false"),
    ({"weekday_gpus": []},                  "at least one GPU"),
    ({"weekend_gpus": [0, 0]},              "repeat"),
    ({"weekday_gpus": [-1]},                "non-negative"),
    ({"start_day": 9},                      "[0, 6]"),
    ({"start_hour": 24},                    "[0, 23]"),
    ({"night_gpus": []},                    "at least one GPU"),
    ({"night_start_hour": 24},              "[0, 23]"),
    ({"start_day": 4, "start_hour": 18, "end_day": 4, "end_hour": 18}, "never switch"),
    ({"night_start_hour": 20, "night_end_hour": 20}, "never switch"),
])
def test_update_rejects_an_invalid_schedule(tmp_path, payload, reason):
    """Invalid settings are refused with an explanatory error, leaving the in-memory settings and the file untouched."""
    scheduler = GpuSchedule(StubPaths(tmp_path), WebLogger(), StubProcesses(), StubSystem(), StubGuard())
    before    = dict(scheduler.settings)

    result = scheduler.update({**GpuSchedule.DEFAULTS, **payload})

    assert result["ok"] is False
    assert reason in result["error"]
    assert scheduler.settings == before
    assert not scheduler.path.exists()


def test_update_rejects_a_partial_schedule(tmp_path):
    """An update missing required keys is refused."""
    scheduler = GpuSchedule(StubPaths(tmp_path), WebLogger(), StubProcesses(), StubSystem(), StubGuard())

    result = scheduler.update({"enabled": True})

    assert result["ok"] is False
    assert "must define" in result["error"]


def test_an_unreadable_schedule_file_falls_back_to_defaults_loudly(tmp_path, capsys):
    """A corrupt settings file falls back to defaults and logs an error naming the file."""
    paths = StubPaths(tmp_path)
    paths.logs_dir.mkdir(parents=True)
    (paths.logs_dir / GpuSchedule.FILE_NAME).write_text("{not json")

    scheduler = GpuSchedule(paths, WebLogger(), StubProcesses(), StubSystem(), StubGuard())

    assert scheduler.settings == GpuSchedule.DEFAULTS

    printed = capsys.readouterr().out

    assert "ERROR" in printed
    assert "ignoring unreadable GPU schedule" in printed
    assert str(scheduler.path) in printed
