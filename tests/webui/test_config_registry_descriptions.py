from __future__ import annotations

import pytest

from config_registry import ConfigRegistry
from project_paths   import ProjectPaths


@pytest.fixture(scope="module")
def registry():
    return ConfigRegistry(ProjectPaths())


@pytest.fixture(scope="module")
def groups(registry):
    return registry.collect()


def _classes(groups):
    for group in groups:
        for cls in group["classes"]:
            yield group, cls


def test_collect_succeeds_and_is_non_empty(groups):
    assert groups
    assert all(group["classes"] for group in groups)


def test_every_section_has_a_general_description(groups):
    for group in groups:
        assert group.get("desc"), f"section {group['module']} is missing a general description"


def test_every_class_has_a_summary(groups):
    for _group, cls in _classes(groups):
        assert cls.get("desc"), f"{cls['module']}::{cls['name']} is missing a summary"


def test_every_field_has_a_description(groups):
    for _group, cls in _classes(groups):
        for field in cls["fields"]:
            assert field.get("desc"), f"{cls['module']}::{cls['name']}.{field['name']} has no description"


def test_no_stale_config_entries(registry):
    reachable = {}
    for _section, files in registry._section_units():
        for path in files:
            module = registry._rel_module(path)
            for cls in registry._parse_module(path):
                reachable[f"{module}::{cls['name']}"] = {field["name"] for field in cls["fields"]}

    configs       = registry.descriptions["configs"]
    stale_classes = [key for key in configs if key not in reachable]
    assert not stale_classes, f"stale config entries with no dataclass: {stale_classes}"

    stale_fields = []
    for key, entry in configs.items():
        for name in entry["fields"]:
            if name not in reachable[key]:
                stale_fields.append(f"{key}.{name}")
    assert not stale_fields, f"stale field descriptions with no field: {stale_fields}"


def test_section_map_matches_rendered_sections(registry, groups):
    sections = registry.descriptions["sections"]
    rendered = {group["module"] for group in groups}
    missing  = rendered - set(sections)
    assert not missing, f"rendered sections without a general description: {missing}"
    for key, text in sections.items():
        assert text and text.strip(), f"empty section description for {key}"


def test_descriptions_are_plain_ascii(groups):
    for _group, cls in _classes(groups):
        blobs = [cls["desc"]] + [field["desc"] for field in cls["fields"]]
        for text in blobs:
            assert all(ord(ch) < 0x2190 for ch in text), f"non-ascii symbol in {cls['name']}: {text!r}"
