from __future__ import annotations

from dataclasses import fields

import pytest

from configuration.sar.geometry_config    import GeometryConfig
from configuration.training.general.loss  import LossConfig
from equation_library                     import EquationLibrary
from flow_library                         import FlowLibrary
from physics_loss_library                 import PhysicsLossLibrary
from pipeline_library                     import PipelineLibrary
from project_paths                        import ProjectPaths
from script_catalog                       import ScriptCatalog
from script_config_resolver               import ScriptConfigResolver
from tools.loss.physical_loss             import PhysicalLoss


NODE_ROLES = {"measured", "calculated", "intermediate", "final"}
NODE_KINDS = {"scalar", "vector", "matrix", "tensor", "set"}

CONFIG_CLASSES = {"LossConfig": LossConfig, "GeometryConfig": GeometryConfig}


@pytest.fixture(scope="module")
def equations():
    return EquationLibrary().collect()


@pytest.fixture(scope="module")
def flows():
    return FlowLibrary().collect()


@pytest.fixture(scope="module")
def physics():
    return PhysicsLossLibrary().collect()


@pytest.fixture(scope="module")
def pipelines():
    return PipelineLibrary().collect()


@pytest.fixture(scope="module")
def script_keys():
    paths = ProjectPaths()
    return {entry["key"] for entry in ScriptCatalog(paths, ScriptConfigResolver(paths)).list_scripts()}


def test_equation_groups_are_named_once_and_carry_items(equations):
    names = [group["group"] for group in equations]

    assert equations
    assert sorted(names) == sorted(set(names))
    assert all(group["blurb"] and group["items"] for group in equations)


def test_every_equation_title_is_unique_inside_its_group(equations):
    pairs = [(group["group"], item["title"]) for group in equations for item in group["items"]]

    assert sorted(pairs) == sorted(set(pairs))


def test_every_equation_has_tex_and_a_legend(equations):
    for group in equations:
        for item in group["items"]:
            assert item["tex"].strip(),  f"{item['title']} has no tex"
            assert item["note"].strip(), f"{item['title']} has no note"

            for variable in item["vars"]:
                assert variable["sym"].strip(),  item["title"]
                assert variable["desc"].strip(), item["title"]


def test_no_equation_uses_a_macro_the_renderer_cannot_draw(equations):
    offenders = [item["title"] for group in equations for item in group["items"] if "\\boldsymbol" in item["tex"]]

    assert not offenders, f"MathJax renders \\mathbf, not \\boldsymbol: {offenders}"


def test_the_flow_catalog_and_the_pipeline_catalog_agree(flows, pipelines):
    assert [flow["key"] for flow in flows] == [pipeline["key"] for pipeline in pipelines]


def test_pipeline_scripts_point_at_real_entry_points(pipelines, script_keys):
    named   = [pipeline["script"] for pipeline in pipelines if pipeline["script"]]
    missing = [key for key in named if key not in script_keys]

    assert not missing, f"pipeline cards naming scripts the console cannot launch: {missing}"


def test_every_pipeline_card_is_filled_in(pipelines):
    keys = [pipeline["key"] for pipeline in pipelines]

    assert sorted(keys) == sorted(set(keys))
    for pipeline in pipelines:
        assert pipeline["name"] and pipeline["blurb"]
        assert len(pipeline["stages"]) >= 3, pipeline["key"]


def test_flow_node_ids_are_unique_within_a_flow(flows):
    for flow in flows:
        ids = [node["id"] for node in flow["nodes"]]

        assert flow["name"] and flow["blurb"]
        assert sorted(ids) == sorted(set(ids)), flow["key"]


def test_flow_nodes_use_the_known_vocabulary(flows):
    for flow in flows:
        for node in flow["nodes"]:
            assert node["role"] in NODE_ROLES, (flow["key"], node["id"], node["role"])
            assert node["kind"] in NODE_KINDS, (flow["key"], node["id"], node["kind"])
            assert node["tex"] and node["desc"] and node["shape"]


def test_every_flow_step_references_declared_nodes(flows):
    dangling = []

    for flow in flows:
        ids = {node["id"] for node in flow["nodes"]}
        for step in flow["steps"]:
            for reference in list(step["inputs"]) + list(step["outputs"]):
                if reference not in ids:
                    dangling.append((flow["key"], step["id"], reference))

    assert not dangling, f"walkthrough steps pointing at undeclared nodes: {dangling}"


def test_flow_steps_are_unique_and_produce_something(flows):
    for flow in flows:
        ids = [step["id"] for step in flow["steps"]]

        assert flow["steps"], flow["key"]
        assert sorted(ids) == sorted(set(ids)), flow["key"]

        for step in flow["steps"]:
            assert step["title"] and step["phase"] and step["note"]
            assert step["outputs"], (flow["key"], step["id"])
            assert step["lines"],   (flow["key"], step["id"])


def test_the_physics_page_has_every_section(physics):
    assert sorted(physics) == ["comparison", "config", "dataset", "intro", "operator", "terms"]
    assert physics["intro"]["points"]
    assert physics["operator"]["items"]
    assert physics["dataset"]["sources"] and physics["dataset"]["pipeline"]


def test_physics_terms_are_numbered_in_order(physics):
    keys = [term["key"] for term in physics["terms"]]

    assert sorted(keys) == sorted(set(keys))
    assert [term["index"] for term in physics["terms"]] == list(range(1, len(keys) + 1))


def test_every_physics_term_names_a_real_loss_implementation(physics):
    missing = []

    for term in physics["terms"]:
        owner, _, method = term["code"].partition(".")
        if owner != PhysicalLoss.__name__ or not hasattr(PhysicalLoss, method):
            missing.append(term["code"])

    assert not missing, f"physics terms documented against absent implementations: {missing}"


def test_the_comparison_table_lists_the_same_five_terms(physics):
    rows = physics["comparison"]["rows"]

    assert [row["term"] for row in rows] == [term["name"] for term in physics["terms"]]
    assert len(physics["comparison"]["columns"]) == 5
    assert all(sorted(row) == ["cost", "invariant", "quantity", "role", "term"] for row in rows)
    assert all(len(row) == len(physics["comparison"]["columns"]) for row in rows)


def test_every_documented_config_field_exists_on_its_dataclass(physics):
    missing = []

    for group in physics["config"]["groups"]:
        known = {field.name for field in fields(CONFIG_CLASSES[group["name"]])}
        for entry in group["fields"]:
            for name in [part.strip() for part in entry["field"].split("/")]:
                if name not in known:
                    missing.append(f"{group['name']}.{name}")

    assert not missing, f"physics config reference naming fields that no longer exist: {missing}"


def test_every_physics_term_carries_its_story_and_legend(physics):
    for term in physics["terms"]:
        for field in ("name", "tagline", "role", "role_label", "quantity", "invariant", "cost_label", "tex", "story", "caveat"):
            assert term[field], f"{term['key']} has an empty {field}"

        assert term["cost"] >= 1
        assert term["vars"]
        for variable in term["vars"]:
            assert variable["sym"] and variable["desc"]
