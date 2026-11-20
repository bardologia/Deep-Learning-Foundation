
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sequence_decisions import SequenceDecisions
from sequence_store     import SequenceStore
from web_logger         import WebLogger


LEAVES = {
    "train_backbone"   : {"paths.dataset_path": "/data/default", "paths.parameters_path": "/data/default/params.npy", "model_overrides": "{}", "training.seed": "0", "training.use_ema": "False", "trials_mode": "presence", "paths.log_base_dir": "runs/benchmark", "logdir": "runs/backbone"},
    "infer_backbone"   : {"paths.dataset_path": "/data/default", "paths.parameters_path": "/data/default/params.npy", "runs_dir": "runs"},
    "benchmark"        : {"dataset_base_path": "/data", "dataset_filter": "", "training.seed": "0"},
    "generate_dataset" : {"runs_dir": "datasets"},
    "cross_validate"   : {"paths.dataset_path": "/data/default", "paths.parameters_path": "/data/default/params.npy", "model_overrides": "{}", "run_tag": "None", "paths.log_base_dir": "runs/cross_validation", "training.batch_size": "256", "loss.use_param_l1": "True"},
    "tune"             : {"paths.dataset_path": "/data/default", "paths.parameters_path": "/data/default/params.npy", "run_tag": "None", "paths.log_base_dir": "runs/tuning", "training.batch_size": "256", "loss.use_param_l1": "True"},
}


class FakeResolver:

    def __init__(self, leaves_by_key: dict) -> None:
        self.leaves_by_key = leaves_by_key

    def resolve(self, key: str, interpreter: str) -> dict:
        leaves = self.leaves_by_key.get(key)
        if leaves is None:
            return {"ok": False, "error": "unresolvable"}
        return {"ok": True, "leaves": [{"path": path, "value": value} for path, value in leaves.items()]}


class FakeInterpreters:

    def preferred_interpreter(self, key: str = "") -> str:
        return "/usr/bin/python3"


def decision_engine(leaves: dict | None = None) -> SequenceDecisions:
    return SequenceDecisions(FakeResolver(leaves or LEAVES), FakeInterpreters(), WebLogger())


class SequencePaths:

    def __init__(self, root: Path, scripts: tuple = ("train_backbone", "infer_backbone", "benchmark", "generate_dataset", "cross_validate", "tune")) -> None:
        self.repo_root     = root
        self.runs_dir      = root / "runs"
        self.sequences_dir = root / "logs" / "sequences"
        self.scripts       = scripts

    def has_script(self, key: str) -> bool:
        return key in self.scripts


@pytest.fixture
def store(tmp_path):
    return SequenceStore(SequencePaths(tmp_path), WebLogger(), decision_engine())


def _block(block_id: str, script: str = "train_backbone", predecessors: list | None = None, **extra) -> dict:
    return {"block_id": block_id, "script": script, "title": script, "predecessors": predecessors or [], **extra}


def _definition(blocks: list, **extra) -> dict:
    return {"name": "seq", "dataset": {"dataset_path": "", "parameters_path": ""}, "detach": True, "isolate": False, "blocks": blocks, **extra}


def test_create_writes_one_json_file_with_an_empty_graph(store):
    result = store.create("my sweep")

    entry = result["entry"]
    saved = json.loads((store.directory / f"{entry['sequence_id']}.json").read_text())

    assert result["ok"]
    assert saved["name"] == "my sweep"
    assert saved["blocks"] == []
    assert saved["dataset"] == {"dataset_path": "", "parameters_path": ""}
    assert saved["isolate"] is True


def test_save_assigns_ids_to_new_blocks_and_normalises_overrides(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([{"script": "train_backbone", "overrides": {"training.seed": 7}}]))

    block = result["entry"]["blocks"][0]
    assert result["ok"]
    assert len(block["block_id"]) == 8
    assert block["overrides"] == {"training.seed": "7"}
    assert block["skip"] is False
    assert block["checkpoint"] is False


def test_save_discards_field_notes_from_the_payload(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    notes       = {"training.patch_size": "waiting", "backbone_name": "frozen"}

    result = store.save(sequence_id, _definition([{"script": "train_backbone", "field_notes": notes}]))

    assert result["ok"]
    assert result["entry"]["blocks"][0]["field_notes"] == {}


def test_a_verdict_freezes_the_blocks_after_the_deciding_one(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["0", "7"], "waiting": "1", "choice": "7"}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"]), _block("c3", predecessors=["b2"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["a1"]["overrides"] == {"training.seed": "1"}
    assert saved["a1"]["field_notes"] == {"training.seed": "waiting"}
    assert saved["b2"]["overrides"] == {"training.seed": "7"}
    assert saved["c3"]["field_notes"] == {"training.seed": "frozen"}


def test_an_undecided_axis_leaves_every_block_waiting(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["0", "7"], "waiting": "1", "choice": ""}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training.seed": "1"}
    assert saved["b2"]["field_notes"] == {"training.seed": "waiting"}


def test_a_value_matching_the_entry_default_is_stamped_without_an_override(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["0", "7"], "waiting": "1", "choice": "0"}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {}
    assert saved["b2"]["field_notes"] == {"training.seed": "frozen"}


def test_a_decision_skips_blocks_whose_entry_does_not_expose_the_path(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.use_ema", "options": ["False", "True"], "waiting": "False", "choice": "True"}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", script="benchmark", predecessors=["a1"]), _block("c3", predecessors=["b2"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {}
    assert saved["b2"]["field_notes"] == {}
    assert saved["c3"]["overrides"] == {"training.use_ema": "True"}


def test_an_empty_pre_verdict_value_stamps_nothing(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.use_ema", "options": ["False", "True"], "waiting": "", "choice": ""}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"], overrides={"training.use_ema": "True"})]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["a1"]["field_notes"] == {}
    assert saved["b2"]["field_notes"] == {}
    assert saved["b2"]["overrides"] == {"training.use_ema": "True"}


def test_a_verdict_over_a_blank_pre_verdict_value_still_freezes_downstream(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["0", "7"], "waiting": "", "choice": "7"}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["a1"]["field_notes"] == {}
    assert saved["b2"]["field_notes"] == {"training.seed": "frozen"}
    assert saved["b2"]["overrides"] == {"training.seed": "7"}


def test_the_later_decision_on_the_same_axis_wins_downstream(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    first       = [{"path": "training.seed", "options": ["1", "7"], "waiting": "1", "choice": "7"}]
    second      = [{"path": "training.seed", "options": ["7", "9"], "waiting": "7", "choice": "9"}]
    blocks      = [_block("a1", decisions=first), _block("b2", predecessors=["a1"], decisions=second), _block("c3", predecessors=["b2"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training.seed": "7"}
    assert saved["c3"]["overrides"] == {"training.seed": "9"}


def test_a_model_overrides_verdict_reaches_only_the_entries_exposing_it(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    overrides   = "{'base_channels': 32, 'depth': 3}"
    decisions   = [{"path": "model_overrides", "options": [overrides], "waiting": "", "choice": overrides}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"]), _block("c3", script="cross_validate", predecessors=["a1"]), _block("d4", script="infer_backbone", predecessors=["a1"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"]   == {"model_overrides": overrides}
    assert saved["c3"]["overrides"]   == {"model_overrides": overrides}
    assert saved["c3"]["field_notes"] == {"model_overrides": "frozen"}
    assert saved["d4"]["overrides"]   == {}
    assert saved["d4"]["field_notes"] == {}


def test_save_refuses_a_decision_it_cannot_resolve_the_entries_for(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["0"], "waiting": "1", "choice": "0"}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", script="generate_dataset", predecessors=["a1"])]

    result = SequenceStore(store.paths, WebLogger(), decision_engine({"train_backbone": LEAVES["train_backbone"]})).save(sequence_id, _definition(blocks))

    assert not result["ok"]
    assert "generate_dataset" in result["error"]


def test_decide_records_a_verdict_and_restamps_the_sequence(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["1", "7"], "waiting": "1", "choice": ""}]
    store.save(sequence_id, _definition([_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]))

    result = store.decide(sequence_id, "a1", "training.seed", "7")

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert result["ok"]
    assert saved["a1"]["decisions"][0]["choice"] == "7"
    assert saved["b2"]["overrides"] == {"training.seed": "7"}


def test_decide_adds_a_verdict_outside_the_options_to_them(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([
        _block("t1", script="tune", decisions=[{"path": "model_overrides", "options": [], "waiting": "", "choice": ""}]),
        _block("c2", script="cross_validate", predecessors=["t1"]),
    ]))

    result = store.decide(sequence_id, "t1", "model_overrides", "{'dropout': 0.3}")

    assert result["ok"], result
    blocks = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert blocks["t1"]["decisions"][0]["options"] == ["{'dropout': 0.3}"]
    assert blocks["t1"]["decisions"][0]["choice"]  == "{'dropout': 0.3}"
    assert blocks["c2"]["overrides"]["model_overrides"] == "{'dropout': 0.3}"
    assert blocks["c2"]["field_notes"]["model_overrides"] == "frozen"


def test_decide_reopens_the_question_when_the_choice_is_cleared(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["1", "7"], "waiting": "1", "choice": "7"}]
    store.save(sequence_id, _definition([_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]))

    result = store.decide(sequence_id, "a1", "training.seed", "")

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training.seed": "1"}
    assert saved["b2"]["field_notes"] == {"training.seed": "waiting"}


def test_decide_refuses_an_axis_the_block_does_not_decide(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([_block("a1")]))

    result = store.decide(sequence_id, "a1", "training.seed", "7")

    assert not result["ok"]
    assert "does not decide" in result["error"]


def test_save_rejects_an_unknown_script(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", script="missing")]))

    assert result == {"ok": False, "error": "unknown script 'missing'"}


def test_save_rejects_an_edge_to_a_missing_block(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", predecessors=["ghost"])]))

    assert not result["ok"]
    assert "missing block" in result["error"]


def test_save_rejects_a_cycle(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", predecessors=["b2"]), _block("b2", predecessors=["a1"])]))

    assert result == {"ok": False, "error": "the sequence has a cycle"}


def test_save_rejects_a_self_edge(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", predecessors=["a1"])]))

    assert not result["ok"]
    assert "links to itself" in result["error"]


def test_save_rejects_an_unknown_sequence(store):
    result = store.save("0" * 12, _definition([]))

    assert result == {"ok": False, "error": "sequence not found"}


def test_order_returns_a_topological_ordering(store):
    blocks  = [_block("c3", predecessors=["a1", "b2"]), _block("a1"), _block("b2", predecessors=["a1"])]
    cleaned = [store._clean_block(block) for block in blocks]

    ordered = store.order(cleaned)

    assert ordered.index("a1") < ordered.index("b2")
    assert ordered.index("b2") < ordered.index("c3")


def test_list_adopts_files_named_differently_from_their_id(store):
    entry = store.create("dropped")["entry"]
    source = store.directory / f"{entry['sequence_id']}.json"
    moved  = store.directory / "full_experiment_plan.json"
    source.rename(moved)

    listed = store.list()["sequences"]

    assert [e["sequence_id"] for e in listed] == [entry["sequence_id"]]
    assert not moved.exists()
    assert (store.directory / f"{entry['sequence_id']}.json").is_file()
    assert store.get(entry["sequence_id"]) is not None


def test_list_skips_unreadable_files_and_id_collisions(store):
    entry = store.create("kept")["entry"]
    (store.directory / "broken.json").write_text("{not json", encoding="utf-8")
    (store.directory / "copy.json").write_text((store.directory / f"{entry['sequence_id']}.json").read_text(encoding="utf-8"), encoding="utf-8")

    listed = store.list()["sequences"]

    assert [e["sequence_id"] for e in listed] == [entry["sequence_id"]]
    assert (store.directory / "broken.json").exists()
    assert (store.directory / "copy.json").exists()


def test_list_skips_a_file_missing_the_required_keys(store):
    entry = store.create("kept")["entry"]
    (store.directory / "no_stamp.json").write_text(json.dumps({"sequence_id": "a" * 12, "blocks": []}), encoding="utf-8")
    (store.directory / "no_id.json").write_text(json.dumps({"updated_at": "2026-08-25T10:00:00"}), encoding="utf-8")
    (store.directory / "not_a_mapping.json").write_text(json.dumps(["blocks"]), encoding="utf-8")

    listed = store.list()["sequences"]

    assert [e["sequence_id"] for e in listed] == [entry["sequence_id"]]
    assert (store.directory / "no_stamp.json").exists()
    assert (store.directory / "no_id.json").exists()


def test_list_returns_sequences_newest_first(store):
    first  = store.create("one")["entry"]["sequence_id"]
    second = store.create("two")["entry"]["sequence_id"]

    for sequence_id, stamp in ((first, "2026-08-18T10:00:00"), (second, "2026-08-18T09:00:00")):
        entry = json.loads((store.directory / f"{sequence_id}.json").read_text())
        (store.directory / f"{sequence_id}.json").write_text(json.dumps({**entry, "updated_at": stamp}))

    listed = [entry["sequence_id"] for entry in store.list()["sequences"]]

    assert listed == [first, second]


def test_get_and_delete_guard_malformed_ids(store):
    assert store.get("not-an-id") is None
    assert store.delete("not-an-id") == {"ok": False, "error": "sequence not found"}
    assert store.delete("f" * 12) == {"ok": False, "error": "sequence not found"}


def test_delete_removes_the_file(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.delete(sequence_id)

    assert result == {"ok": True}
    assert not (store.directory / f"{sequence_id}.json").exists()


def test_save_cleans_the_finalize_switches(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", finalize={"compare_trials": 1, "unknown": True}), _block("b2")], isolate=True))

    assert result["ok"], result
    assert result["entry"]["blocks"][0]["finalize"] == {"compare_seeds": False, "compare_trials": True, "mail_experiment_reports": False}
    assert result["entry"]["blocks"][1]["finalize"] == {"compare_seeds": False, "compare_trials": False, "mail_experiment_reports": False}


def test_save_keeps_a_finalize_switch_without_isolation(store):
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", finalize={"compare_seeds": True})]))

    assert result["ok"]
    assert result["entry"]["isolate"] is False
    assert result["entry"]["blocks"][0]["finalize"]["compare_seeds"] is True
