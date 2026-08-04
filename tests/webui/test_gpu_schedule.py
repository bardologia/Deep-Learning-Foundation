from __future__ import annotations

import json
from datetime import datetime

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
    def __init__(self) -> None:
        self.jobs   = []
        self.pools  = {}
        self.writes = []
        self.refuse = False

    def list_jobs(self) -> list[dict]:
        return list(self.jobs)

    def gpu_pool(self, job_id: str) -> dict:
        return self.pools.get(job_id, {"ok": True, "supported": False, "live": False})

    def set_gpus(self, job_id: str, gpus) -> dict:
        if self.refuse:
            return {"ok": False, "error": "job is not running"}

        self.writes.append((job_id, list(gpus)))
        self.pools[job_id]["gpus"] = list(gpus)
        return {"ok": True, "gpus": list(gpus), "parked": False}

    def add_running_fan_out(self, job_id: str) -> None:
        self.jobs.append({"job_id": job_id, "status": "running", "script": "train_backbone"})
        self.pools[job_id] = {"ok": True, "supported": True, "live": True, "gpus": [0]}


class StubGuard:
    def __init__(self) -> None:
        self.bounced = []

    def bounce(self, gpu: int, user: str) -> None:
        self.bounced.append({"gpu_index": gpu, "user": user, "status": "survived"})

    def take_bounced(self) -> list[dict]:
        bounced      = list(self.bounced)
        self.bounced = []
        return bounced


class StubSystem:
    def __init__(self) -> None:
        self.user    = "bard"
        self.holders = {}

    def occupy(self, index: int, user: str) -> None:
        self.holders.setdefault(index, []).append(user)

    def release(self, index: int) -> None:
        self.holders.pop(index, None)

    def gpu_occupancy(self) -> list[dict]:
        return [
            {"index": index, "procs": [{"pid": 100 + index, "owner": owner} for owner in self.holders.get(index, [])]}
            for index in range(4)
        ]


@pytest.fixture
def schedule(tmp_path):
    processes = StubProcesses()
    system    = StubSystem()
    schedule  = GpuSchedule(StubPaths(tmp_path), WebLogger(), processes, system, StubGuard())
    schedule.settings["enabled"] = True
    return schedule, processes, system


def test_default_window_spans_friday_evening_to_monday_morning():
    window = WeekWindow(**{key: GpuSchedule.DEFAULTS[key] for key in ("start_day", "start_hour", "end_day", "end_hour")})

    assert window.contains(FRIDAY_EVENING) is True
    assert window.contains(SATURDAY)       is True
    assert window.contains(SUNDAY_NIGHT)   is True
    assert window.contains(MONDAY_EARLY)   is True

    assert window.contains(FRIDAY_MORNING) is False
    assert window.contains(MONDAY_WORKDAY) is False
    assert window.contains(WEDNESDAY)      is False


def test_window_that_does_not_wrap_the_week():
    window = WeekWindow(start_day=1, start_hour=8, end_day=3, end_hour=8)

    assert window.contains(datetime(2026, 7, 21, 9, 0))  is True
    assert window.contains(datetime(2026, 7, 22, 23, 0)) is True
    assert window.contains(datetime(2026, 7, 23, 9, 0))  is False
    assert window.contains(datetime(2026, 7, 20, 9, 0))  is False


def test_window_boundaries_are_inclusive_at_the_start_and_exclusive_at_the_end():
    window = WeekWindow(**{key: GpuSchedule.DEFAULTS[key] for key in ("start_day", "start_hour", "end_day", "end_hour")})

    assert window.contains(datetime(2026, 7, 17, 18, 0))  is True
    assert window.contains(datetime(2026, 7, 17, 17, 59)) is False
    assert window.contains(datetime(2026, 7, 20, 7, 59))  is True
    assert window.contains(datetime(2026, 7, 20, 8, 0))   is False


def test_default_night_window_spans_the_evening_to_the_next_morning():
    window = NightWindow(**{key: GpuSchedule.DEFAULTS[key] for key in ("night_start_hour", "night_end_hour")})

    assert window.contains(WEDNESDAY_NIGHT)   is True
    assert window.contains(THURSDAY_EARLY)    is True
    assert window.contains(datetime(2026, 7, 22, 20, 0)) is True

    assert window.contains(WEDNESDAY)         is False
    assert window.contains(THURSDAY_WORKDAY)  is False
    assert window.contains(datetime(2026, 7, 22, 19, 59)) is False
    assert window.contains(datetime(2026, 7, 23, 8, 0))   is False


def test_phase_names_follow_the_window(schedule):
    scheduler, _processes, _system = schedule

    assert scheduler.phase(SATURDAY)         == "weekend"
    assert scheduler.phase(MONDAY_WORKDAY)   == "weekday"
    assert scheduler.phase(WEDNESDAY_NIGHT)  == "night"
    assert scheduler.phase(THURSDAY_EARLY)   == "night"


def test_the_weekend_window_outranks_the_night_window(schedule):
    scheduler, _processes, _system = schedule

    assert scheduler.phase(SUNDAY_NIGHT)   == "weekend"
    assert scheduler.phase(FRIDAY_EVENING) == "weekend"


def test_transition_into_the_night_grows_the_pool_and_the_morning_shrinks_it(schedule):
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)
    scheduler.tick(THURSDAY_EARLY)
    scheduler.tick(THURSDAY_WORKDAY)

    assert processes.writes == [("job1", [0, 1, 2, 3]), ("job1", [2, 3])]


def test_first_tick_records_the_phase_without_touching_the_launch_selection(schedule):
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)

    assert processes.writes    == []
    assert scheduler.applied   == {"job1": {"phase": "weekday", "withheld": []}}


def test_transition_into_the_weekend_grows_the_pool(schedule):
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(FRIDAY_EVENING)

    assert processes.writes == [("job1", [0, 1, 2, 3])]


def test_transition_back_to_the_week_shrinks_the_pool(schedule):
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(SATURDAY)
    scheduler.tick(MONDAY_WORKDAY)

    assert processes.writes == [("job1", [2, 3])]


def test_ticks_inside_one_phase_never_rewrite_the_pool(schedule):
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(SATURDAY)
    scheduler.tick(SUNDAY_NIGHT)
    scheduler.tick(MONDAY_EARLY)

    assert processes.writes == []


def test_a_manual_resize_survives_until_the_next_transition(schedule):
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(SATURDAY)
    processes.pools["job1"]["gpus"] = [2]

    scheduler.tick(SUNDAY_NIGHT)
    assert processes.writes == []

    scheduler.tick(MONDAY_WORKDAY)
    assert processes.writes == [("job1", [2, 3])]


def test_a_gpu_held_by_another_user_is_left_out_of_the_pool(schedule):
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                      == [("job1", [0, 2, 3])]
    assert scheduler.applied["job1"]["withheld"] == [1]


def test_our_own_processes_never_block_the_pool(schedule):
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    system.occupy(2, "bard")
    system.occupy(3, "bard")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                      == [("job1", [0, 1, 2, 3])]
    assert scheduler.applied["job1"]["withheld"] == []


def test_a_pool_held_entirely_by_others_leaves_the_job_untouched(schedule):
    scheduler, processes, system = schedule
    processes.add_running_fan_out("job1")
    for index in (0, 1, 2, 3):
        system.occupy(index, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes                      == []
    assert scheduler.applied["job1"]["withheld"] == [0, 1, 2, 3]


def test_a_blocked_shrink_stays_pending_until_the_weekday_pool_frees(schedule):
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
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague")
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
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague")

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes             == []
    assert scheduler.nap_until          is None


def test_charity_never_leaves_a_job_without_a_gpu(schedule):
    scheduler, processes, system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [1]
    for index in (0, 2, 3):
        system.occupy(index, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague")

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes             == []
    assert scheduler.nap_until          is None


def test_a_bounce_off_an_unshrinkable_pool_frees_a_gpu_from_the_biggest_one(schedule):
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.add_running_fan_out("job2")
    processes.pools["job1"]["gpus"] = [0]
    processes.pools["job2"]["gpus"] = [1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(0, "colleague")
    given = scheduler.charity(WEDNESDAY)

    assert given                                 == ["job2"]
    assert processes.writes                      == [("job2", [1, 2])]
    assert scheduler.applied["job2"]["withheld"] == [3]


def test_charity_off_drains_bounces_without_donating(schedule):
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague")

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes             == []
    assert scheduler.guard.bounced      == []


def test_the_nap_blocks_greedy_until_it_ends_then_greedy_reclaims(schedule):
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague")
    scheduler.charity(WEDNESDAY)

    scheduler.sweep(WEDNESDAY)
    assert processes.writes == [("job1", [0, 2, 3])]

    later = WEDNESDAY + GpuSchedule.CHARITY_NAP
    scheduler.charity(later)
    scheduler.sweep(later)

    assert scheduler.nap_until  is None
    assert processes.writes[-1] == ("job1", [0, 1, 2, 3])


def test_a_second_bounce_during_the_nap_gets_nothing(schedule):
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague")
    scheduler.charity(WEDNESDAY)

    scheduler.guard.bounce(2, "rival")

    assert scheduler.charity(WEDNESDAY) == []
    assert processes.writes             == [("job1", [0, 2, 3])]
    assert scheduler.guard.bounced      == []

    assert scheduler.charity(WEDNESDAY + GpuSchedule.CHARITY_NAP) == []


def test_a_phase_cross_during_the_nap_keeps_the_gift_out(schedule):
    scheduler, processes, system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [2, 3]
    system.occupy(0, "colleague")
    system.occupy(1, "colleague")

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(2, "colleague")
    scheduler.charity(WEDNESDAY)
    assert processes.writes == [("job1", [3])]

    system.release(0)
    system.release(1)
    scheduler.tick(WEDNESDAY_NIGHT)

    assert processes.writes[-1]                  == ("job1", [0, 1, 3])
    assert scheduler.applied["job1"]["withheld"] == [2]


def test_turning_charity_off_ends_the_nap(schedule):
    scheduler, processes, _system = schedule
    scheduler.settings["charity"] = True
    processes.add_running_fan_out("job1")
    processes.pools["job1"]["gpus"] = [0, 1, 2, 3]

    scheduler.tick(WEDNESDAY)
    scheduler.guard.bounce(1, "colleague")
    scheduler.charity(WEDNESDAY)
    assert scheduler.nap_until is not None

    result = scheduler.update({**GpuSchedule.DEFAULTS, "enabled": True, "charity": False})

    assert result["ok"]         is True
    assert scheduler.nap_until  is None
    assert scheduler.gifted     == set()


def test_a_disabled_schedule_sweeps_nothing(schedule):
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
    scheduler, processes, _system = schedule
    scheduler.settings["enabled"] = False
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    scheduler.tick(SATURDAY)

    assert processes.writes  == []
    assert scheduler.applied == {"job1": {"phase": "weekend", "withheld": []}}


def test_jobs_without_a_live_pool_are_ignored(schedule):
    scheduler, processes, _system = schedule
    processes.jobs.append({"job_id": "single", "status": "running", "script": "train_backbone"})

    scheduler.tick(WEDNESDAY)
    scheduler.tick(SATURDAY)

    assert processes.writes  == []
    assert scheduler.applied == {}


def test_finished_jobs_are_forgotten(schedule):
    scheduler, processes, _system = schedule
    processes.add_running_fan_out("job1")

    scheduler.tick(WEDNESDAY)
    assert scheduler.applied == {"job1": {"phase": "weekday", "withheld": []}}

    processes.jobs[0]["status"] = "finished"
    scheduler.tick(SATURDAY)

    assert scheduler.applied == {}


def test_update_persists_and_reloads(tmp_path):
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
    scheduler = GpuSchedule(StubPaths(tmp_path), WebLogger(), StubProcesses(), StubSystem(), StubGuard())
    before    = dict(scheduler.settings)

    result = scheduler.update({**GpuSchedule.DEFAULTS, **payload})

    assert result["ok"] is False
    assert reason in result["error"]
    assert scheduler.settings == before
    assert not scheduler.path.exists()


def test_update_rejects_a_partial_schedule(tmp_path):
    scheduler = GpuSchedule(StubPaths(tmp_path), WebLogger(), StubProcesses(), StubSystem(), StubGuard())

    result = scheduler.update({"enabled": True})

    assert result["ok"] is False
    assert "must define" in result["error"]


def test_an_unreadable_schedule_file_falls_back_to_defaults_loudly(tmp_path):
    paths = StubPaths(tmp_path)
    paths.logs_dir.mkdir(parents=True)
    (paths.logs_dir / GpuSchedule.FILE_NAME).write_text("{not json")

    scheduler = GpuSchedule(paths, WebLogger(), StubProcesses(), StubSystem(), StubGuard())

    assert scheduler.settings == GpuSchedule.DEFAULTS
