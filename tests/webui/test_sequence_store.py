"""Tests for the experiment-sequence store.

Covers creation of empty sequences, full-definition saves with block-id
assignment, validation of scripts, edges and cycles, the decision axes a block
declares and the verdicts they freeze downstream, the family namespace a verdict
lands in on an entry running several training families, newest-first listing,
lookup and deletion guards, and the topological order helper.

The config-resolver stand-in and its leaf table live here because both the
store's decision engine and the runner's dataset injection resolve entry
points; the runner and router test modules import them from this one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sequence_decisions import SequenceDecisions
from sequence_store     import SequenceStore
from web_logger         import WebLogger


LEAVES = {
    "train_backbone"            : {"paths.dataset_path": "/data/default", "paths.parameters_path": "/data/default/params.npy", "model_overrides": "{}", "training.seed": "0", "training.use_ema": "False", "trials_mode": "curriculum", "paths.log_base_dir": "runs/benchmark", "logdir": "runs/backbone"},
    "infer_backbone"            : {"paths.dataset_path": "/data/default", "paths.parameters_path": "/data/default/params.npy", "runs_dir": "runs"},
    "benchmark"                 : {"dataset_base_path": "/data", "dataset_filter": "", "training.seed": "0"},
    "generate_dataset"               : {"runs_dir": "datasets"},
    "train_jepa"                : {"paths.dataset_path": "/data/default", "logdir": "runs/jepa", "model_overrides": "{}", "training.batch_size": "256", "curriculum.enabled": "False", "embedding_loss.use_embedding_mse": "True"},
    "train_profile_autoencoder" : {"paths.dataset_path": "/data/default", "logdir": "runs/profile_autoencoder", "model_overrides": "{}", "training.batch_size": "8192", "ae_model_name": "mlp_ae", "ae_loss.use_mse_curve": "True"},
    "cross_validate"            : {"paths.dataset_path": "/data/default", "paths.parameters_path": "/data/default/params.npy", "model_overrides": "{}", "run_tag": "None", "paths.log_base_dir": "runs/cross_validation", "training_type": "backbone", "training.batch_size": "256", "curriculum.enabled": "True", "jepa.curriculum.enabled": "False", "jepa.embedding_loss.use_embedding_mse": "True", "autoencoder.ae_model_name": "mlp_ae", "autoencoder.ae_loss.use_mse_curve": "True"},
}


class FakeResolver:
    """Config-resolver stand-in returning fixed leaves with default values per entry-point key.

    Attributes:
        leaves_by_key : Leaf path-to-default mappings keyed by entry-point key.
    """

    def __init__(self, leaves_by_key: dict) -> None:
        """Stores the per-key leaf defaults."""
        self.leaves_by_key = leaves_by_key

    def resolve(self, key: str, interpreter: str) -> dict:
        """Returns the configured leaves for the key, or a failed result when absent."""
        leaves = self.leaves_by_key.get(key)
        if leaves is None:
            return {"ok": False, "error": "unresolvable"}
        return {"ok": True, "leaves": [{"path": path, "value": value} for path, value in leaves.items()]}


class FakeInterpreters:
    """Launcher stand-in exposing only the interpreter choice the decision engine needs."""

    def preferred_interpreter(self, key: str = "") -> str:
        """Returns a fixed interpreter path."""
        return "/usr/bin/python3"


def decision_engine(leaves: dict | None = None) -> SequenceDecisions:
    """Returns a decision engine resolving against the fake leaf table.

    Args:
        leaves : Leaf defaults keyed by entry-point key; the shared table when omitted.
    """
    return SequenceDecisions(FakeResolver(leaves or LEAVES), FakeInterpreters(), WebLogger())


class SequencePaths:
    """Minimal project-paths stand-in exposing a sequences directory and a script whitelist.

    Attributes:
        repo_root     : Root of the fake repository.
        runs_dir      : Root of the run-output tree the runner isolates under.
        sequences_dir : Directory the store writes one JSON file per sequence into.
        scripts       : Entry-point keys treated as present in the repository.
    """

    def __init__(self, root: Path, scripts: tuple = ("train_backbone", "infer_backbone", "benchmark", "generate_dataset", "train_jepa", "train_profile_autoencoder", "cross_validate")) -> None:
        """Points the sequences directory at ``<root>/logs/sequences`` and records the known scripts."""
        self.repo_root     = root
        self.runs_dir      = root / "runs"
        self.sequences_dir = root / "logs" / "sequences"
        self.scripts       = scripts

    def has_script(self, key: str) -> bool:
        """Returns whether the entry-point key is one of the configured scripts."""
        return key in self.scripts


@pytest.fixture
def store(tmp_path):
    """Returns a store writing into a temporary sequences directory."""
    return SequenceStore(SequencePaths(tmp_path), WebLogger(), decision_engine())


def _block(block_id: str, script: str = "train_backbone", predecessors: list | None = None, **extra) -> dict:
    """Returns a valid block payload with the given id, script and predecessors."""
    return {"block_id": block_id, "script": script, "title": script, "predecessors": predecessors or [], **extra}


def _definition(blocks: list, **extra) -> dict:
    """Returns a full save payload holding the given blocks."""
    return {"name": "seq", "dataset": {"dataset_path": "", "parameters_path": ""}, "detach": True, "isolate": False, "blocks": blocks, **extra}


def test_create_writes_one_json_file_with_an_empty_graph(store):
    """A created sequence lands on disk with no blocks and an empty dataset selection."""
    result = store.create("my sweep")

    entry = result["entry"]
    saved = json.loads((store.directory / f"{entry['sequence_id']}.json").read_text())

    assert result["ok"]
    assert saved["name"] == "my sweep"
    assert saved["blocks"] == []
    assert saved["dataset"] == {"dataset_path": "", "parameters_path": ""}
    assert saved["isolate"] is True


def test_save_assigns_ids_to_new_blocks_and_normalises_overrides(store):
    """Blocks arriving without an id get one, and override values are stringified."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([{"script": "train_backbone", "overrides": {"training.seed": 7}}]))

    block = result["entry"]["blocks"][0]
    assert result["ok"]
    assert len(block["block_id"]) == 8
    assert block["overrides"] == {"training.seed": "7"}
    assert block["skip"] is False
    assert block["checkpoint"] is False


def test_save_discards_field_notes_from_the_payload(store):
    """Field notes are never taken from the caller: they are rewritten from the declared decisions."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    notes       = {"training.patch_size": "waiting", "backbone_name": "frozen"}

    result = store.save(sequence_id, _definition([{"script": "train_backbone", "field_notes": notes}]))

    assert result["ok"]
    assert result["entry"]["blocks"][0]["field_notes"] == {}


def test_a_verdict_freezes_the_blocks_after_the_deciding_one(store):
    """A decided axis is stamped as the verdict downstream and as the pre-verdict value elsewhere."""
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
    """Without a verdict, blocks after the deciding one keep the pre-verdict value."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["0", "7"], "waiting": "1", "choice": ""}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training.seed": "1"}
    assert saved["b2"]["field_notes"] == {"training.seed": "waiting"}


def test_a_value_matching_the_entry_default_is_stamped_without_an_override(store):
    """A stamped value equal to the entry's own default leaves the note but no override."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["0", "7"], "waiting": "1", "choice": "0"}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {}
    assert saved["b2"]["field_notes"] == {"training.seed": "frozen"}


def test_a_decision_skips_blocks_whose_entry_does_not_expose_the_path(store):
    """Only blocks whose entry point exposes the axis are stamped with it."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.use_ema", "options": ["False", "True"], "waiting": "False", "choice": "True"}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", script="benchmark", predecessors=["a1"]), _block("c3", predecessors=["b2"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {}
    assert saved["b2"]["field_notes"] == {}
    assert saved["c3"]["overrides"] == {"training.use_ema": "True"}


def test_an_empty_pre_verdict_value_stamps_nothing(store):
    """A declared axis whose pre-verdict value is still blank leaves the exposing blocks' overrides untouched."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.use_ema", "options": ["False", "True"], "waiting": "", "choice": ""}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"], overrides={"training.use_ema": "True"})]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["a1"]["field_notes"] == {}
    assert saved["b2"]["field_notes"] == {}
    assert saved["b2"]["overrides"] == {"training.use_ema": "True"}


def test_a_verdict_over_a_blank_pre_verdict_value_still_freezes_downstream(store):
    """A recorded verdict freezes the blocks after the decider even while the pre-verdict value is blank."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["0", "7"], "waiting": "", "choice": "7"}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["a1"]["field_notes"] == {}
    assert saved["b2"]["field_notes"] == {"training.seed": "frozen"}
    assert saved["b2"]["overrides"] == {"training.seed": "7"}


def test_the_later_decision_on_the_same_axis_wins_downstream(store):
    """When two blocks decide one axis, the topologically later verdict governs the blocks after it."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    first       = [{"path": "training.seed", "options": ["1", "7"], "waiting": "1", "choice": "7"}]
    second      = [{"path": "training.seed", "options": ["7", "9"], "waiting": "7", "choice": "9"}]
    blocks      = [_block("a1", decisions=first), _block("b2", predecessors=["a1"], decisions=second), _block("c3", predecessors=["b2"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training.seed": "7"}
    assert saved["c3"]["overrides"] == {"training.seed": "9"}


def test_a_bare_autoencoder_axis_lands_on_the_namespaced_cross_validation_field(store):
    """An axis the autoencoder entry exposes bare reaches the field cross-validation keeps that family under."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "ae_model_name", "options": ["mlp_ae", "conv1d_ae"], "waiting": "mlp_ae", "choice": "conv1d_ae"}]
    blocks      = [_block("a1", script="train_profile_autoencoder", decisions=decisions), _block("b2", script="cross_validate", predecessors=["a1"], overrides={"training_type": "profile_autoencoder"})]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training_type": "profile_autoencoder", "autoencoder.ae_model_name": "conv1d_ae"}
    assert saved["b2"]["field_notes"] == {"autoencoder.ae_model_name": "frozen"}


def test_a_bare_jepa_axis_lands_on_the_namespaced_cross_validation_field(store):
    """An axis the JEPA entry exposes bare reaches the field cross-validation keeps the JEPA coupling under."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "embedding_loss.use_embedding_mse", "options": ["True", "False"], "waiting": "True", "choice": "False"}]
    blocks      = [_block("a1", script="train_jepa", decisions=decisions), _block("b2", script="cross_validate", predecessors=["a1"], overrides={"training_type": "jepa"})]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training_type": "jepa", "jepa.embedding_loss.use_embedding_mse": "False"}
    assert saved["b2"]["field_notes"] == {"jepa.embedding_loss.use_embedding_mse": "frozen"}


def test_a_shared_axis_keeps_resolving_bare_on_a_namespaced_block(store):
    """An axis no training family namespaces is stamped bare, whichever family the target block runs."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.batch_size", "options": ["256", "512"], "waiting": "256", "choice": "512"}]
    blocks      = [_block("a1", script="train_jepa", decisions=decisions), _block("b2", script="cross_validate", predecessors=["a1"], overrides={"training_type": "jepa"})]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training_type": "jepa", "training.batch_size": "512"}
    assert saved["b2"]["field_notes"] == {"training.batch_size": "frozen"}


def test_a_jepa_curriculum_verdict_stays_out_of_the_backbone_curriculum(store):
    """``curriculum.enabled`` is a leaf of both families, so the verdict lands under the family each target runs."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "curriculum.enabled", "options": ["False", "True"], "waiting": "False", "choice": "True"}]
    blocks      = [_block("a1", script="train_jepa", decisions=decisions), _block("b2", script="cross_validate", predecessors=["a1"], overrides={"training_type": "jepa"}), _block("c3", script="cross_validate", predecessors=["a1"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training_type": "jepa", "jepa.curriculum.enabled": "True"}
    assert saved["b2"]["field_notes"] == {"jepa.curriculum.enabled": "frozen"}
    assert saved["c3"]["overrides"] == {}
    assert saved["c3"]["field_notes"] == {"curriculum.enabled": "frozen"}


def test_a_family_axis_skips_a_block_running_another_family(store):
    """An axis only another family namespaces is no leaf of the target, so it stamps nothing there."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "ae_loss.use_mse_curve", "options": ["True", "False"], "waiting": "True", "choice": "False"}]
    blocks      = [_block("a1", script="train_profile_autoencoder", decisions=decisions), _block("b2", script="cross_validate", predecessors=["a1"], overrides={"training_type": "jepa"})]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training_type": "jepa"}
    assert saved["b2"]["field_notes"] == {}


def test_a_model_overrides_verdict_governs_only_the_family_that_decided_it(store):
    """The keys of ``model_overrides`` name one family's architecture fields, so the verdict stops at that family."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    overrides   = "{'hidden_dim': 256, 'depth': 3}"
    decisions   = [{"path": "model_overrides", "options": [overrides], "waiting": "", "choice": overrides}]
    blocks      = [_block("a1", script="train_profile_autoencoder", decisions=decisions), _block("b2", script="train_profile_autoencoder", predecessors=["a1"]), _block("c3", script="train_backbone", predecessors=["a1"]), _block("d4", script="cross_validate", predecessors=["a1"], overrides={"training_type": "profile_autoencoder"}), _block("e5", script="cross_validate", predecessors=["a1"])]

    result = store.save(sequence_id, _definition(blocks))

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"model_overrides": overrides}
    assert saved["d4"]["overrides"] == {"training_type": "profile_autoencoder", "model_overrides": overrides}
    assert saved["c3"]["overrides"] == {}
    assert saved["c3"]["field_notes"] == {}
    assert saved["e5"]["overrides"] == {}
    assert saved["e5"]["field_notes"] == {}


def test_save_refuses_a_block_running_an_unknown_training_family(store):
    """A block whose ``training_type`` has no namespace is refused by value."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "ae_model_name", "options": ["mlp_ae", "conv1d_ae"], "waiting": "mlp_ae", "choice": "conv1d_ae"}]
    blocks      = [_block("a1", script="train_profile_autoencoder", decisions=decisions), _block("b2", script="cross_validate", predecessors=["a1"], overrides={"training_type": "unrolled"})]

    result = store.save(sequence_id, _definition(blocks))

    assert not result["ok"]
    assert "'unrolled'" in result["error"]
    assert "profile_autoencoder" in result["error"]


def test_save_refuses_a_decision_it_cannot_resolve_the_entries_for(store):
    """A decision on a sequence holding an unresolvable entry point is refused by name."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["0"], "waiting": "1", "choice": "0"}]
    blocks      = [_block("a1", decisions=decisions), _block("b2", script="generate_dataset", predecessors=["a1"])]

    result = SequenceStore(store.paths, WebLogger(), decision_engine({"train_backbone": LEAVES["train_backbone"]})).save(sequence_id, _definition(blocks))

    assert not result["ok"]
    assert "generate_dataset" in result["error"]


def test_decide_records_a_verdict_and_restamps_the_sequence(store):
    """Recording a verdict through the store freezes it into the blocks after the deciding one."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["1", "7"], "waiting": "1", "choice": ""}]
    store.save(sequence_id, _definition([_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]))

    result = store.decide(sequence_id, "a1", "training.seed", "7")

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert result["ok"]
    assert saved["a1"]["decisions"][0]["choice"] == "7"
    assert saved["b2"]["overrides"] == {"training.seed": "7"}


def test_decide_reopens_the_question_when_the_choice_is_cleared(store):
    """Clearing a verdict puts the blocks after the deciding one back on the pre-verdict value."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["1", "7"], "waiting": "1", "choice": "7"}]
    store.save(sequence_id, _definition([_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]))

    result = store.decide(sequence_id, "a1", "training.seed", "")

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert saved["b2"]["overrides"] == {"training.seed": "1"}
    assert saved["b2"]["field_notes"] == {"training.seed": "waiting"}


def test_decide_refuses_an_axis_the_block_does_not_decide(store):
    """Recording a verdict for an axis the block never declared is refused."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([_block("a1")]))

    result = store.decide(sequence_id, "a1", "training.seed", "7")

    assert not result["ok"]
    assert "does not decide" in result["error"]


def test_save_rejects_an_unknown_script(store):
    """A block whose script has no entry point is refused by name."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", script="missing")]))

    assert result == {"ok": False, "error": "unknown script 'missing'"}


def test_save_rejects_an_edge_to_a_missing_block(store):
    """A predecessor id that matches no block in the payload is refused."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", predecessors=["ghost"])]))

    assert not result["ok"]
    assert "missing block" in result["error"]


def test_save_rejects_a_cycle(store):
    """A definition whose edges form a cycle is refused."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", predecessors=["b2"]), _block("b2", predecessors=["a1"])]))

    assert result == {"ok": False, "error": "the sequence has a cycle"}


def test_save_rejects_a_self_edge(store):
    """A block that lists itself as a predecessor is refused."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", predecessors=["a1"])]))

    assert not result["ok"]
    assert "links to itself" in result["error"]


def test_save_rejects_an_unknown_sequence(store):
    """Saving against an id with no file on disk is refused."""
    result = store.save("0" * 12, _definition([]))

    assert result == {"ok": False, "error": "sequence not found"}


def test_order_returns_a_topological_ordering(store):
    """The order helper puts every block after all of its predecessors."""
    blocks  = [_block("c3", predecessors=["a1", "b2"]), _block("a1"), _block("b2", predecessors=["a1"])]
    cleaned = [store._clean_block(block) for block in blocks]

    ordered = store.order(cleaned)

    assert ordered.index("a1") < ordered.index("b2")
    assert ordered.index("b2") < ordered.index("c3")


def test_list_adopts_files_named_differently_from_their_id(store):
    """A definition dropped in under any filename is renamed to its sequence id and listed."""
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
    """A malformed file or a duplicate id is reported and skipped, not fatal to the listing."""
    entry = store.create("kept")["entry"]
    (store.directory / "broken.json").write_text("{not json", encoding="utf-8")
    (store.directory / "copy.json").write_text((store.directory / f"{entry['sequence_id']}.json").read_text(encoding="utf-8"), encoding="utf-8")

    listed = store.list()["sequences"]

    assert [e["sequence_id"] for e in listed] == [entry["sequence_id"]]
    assert (store.directory / "broken.json").exists()
    assert (store.directory / "copy.json").exists()


def test_list_skips_a_file_missing_the_required_keys(store):
    """A JSON file without sequence_id or updated_at is skipped instead of breaking the listing's sort."""
    entry = store.create("kept")["entry"]
    (store.directory / "no_stamp.json").write_text(json.dumps({"sequence_id": "a" * 12, "blocks": []}), encoding="utf-8")
    (store.directory / "no_id.json").write_text(json.dumps({"updated_at": "2026-08-25T10:00:00"}), encoding="utf-8")
    (store.directory / "not_a_mapping.json").write_text(json.dumps(["blocks"]), encoding="utf-8")

    listed = store.list()["sequences"]

    assert [e["sequence_id"] for e in listed] == [entry["sequence_id"]]
    assert (store.directory / "no_stamp.json").exists()
    assert (store.directory / "no_id.json").exists()


def test_list_returns_sequences_newest_first(store):
    """Listing sorts by update stamp, newest first."""
    first  = store.create("one")["entry"]["sequence_id"]
    second = store.create("two")["entry"]["sequence_id"]

    for sequence_id, stamp in ((first, "2026-08-18T10:00:00"), (second, "2026-08-18T09:00:00")):
        entry = json.loads((store.directory / f"{sequence_id}.json").read_text())
        (store.directory / f"{sequence_id}.json").write_text(json.dumps({**entry, "updated_at": stamp}))

    listed = [entry["sequence_id"] for entry in store.list()["sequences"]]

    assert listed == [first, second]


def test_get_and_delete_guard_malformed_ids(store):
    """Lookup returns None and deletion reports not found for malformed or absent ids."""
    assert store.get("not-an-id") is None
    assert store.delete("not-an-id") == {"ok": False, "error": "sequence not found"}
    assert store.delete("f" * 12) == {"ok": False, "error": "sequence not found"}


def test_delete_removes_the_file(store):
    """Deleting an existing sequence removes its JSON file."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.delete(sequence_id)

    assert result == {"ok": True}
    assert not (store.directory / f"{sequence_id}.json").exists()


def test_save_cleans_the_finalize_switches(store):
    """The finalize switches are stored as one boolean per chain step, all defaulting to off."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", finalize={"compare_trials": 1, "unknown": True}), _block("b2")], isolate=True))

    assert result["ok"], result
    assert result["entry"]["blocks"][0]["finalize"] == {"compare_seeds": False, "compare_trials": True, "mail_experiment_reports": False}
    assert result["entry"]["blocks"][1]["finalize"] == {"compare_seeds": False, "compare_trials": False, "mail_experiment_reports": False}


def test_save_keeps_a_finalize_switch_without_isolation(store):
    """A finalize switch does not need the isolated workspace: the chain runs on the entry's own sweep directory instead."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]

    result = store.save(sequence_id, _definition([_block("a1", finalize={"compare_seeds": True})]))

    assert result["ok"]
    assert result["entry"]["isolate"] is False
    assert result["entry"]["blocks"][0]["finalize"]["compare_seeds"] is True
