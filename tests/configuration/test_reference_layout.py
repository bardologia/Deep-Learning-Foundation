"""Guards that every path default in the configuration package is rebasable.

The machine profile retargets defaults by rebasing the checkout-relative roots
named in ReferenceLayout. A default written as an absolute path, or one placed
below another root, would silently keep pointing at the checkout on every other
installation, so this test walks every constructible configuration dataclass and
refuses absolute defaults and checks that a full profile leaves no reference path
behind.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import is_dataclass

import pytest

import configuration
from configuration.reference_layout     import ReferenceLayout
from configuration.training.general.run import DEFAULT_DATASET_DIR, DEFAULT_PARAMS_RUN
from tools.runtime.config_cli           import ConfigCli
from tools.runtime.machine_profile      import MachineProfile

ROOTS = (str(ReferenceLayout.DATA_ROOT), str(ReferenceLayout.RUNS_ROOT), str(ReferenceLayout.RESULTS_ROOT))

ALL_ROOTS_PROFILE = {
    "roots"   : {"repo": "/local/repo", "data": "/local/data", "runs": "/local/runs", "results": "/local/results"},
    "dataset" : {"path": "/local/data/scene", "parameters_path": "/local/data/scene/params/run/parameters.npy"},
}


def _constructible_dataclasses() -> list[tuple[str, type]]:
    """Returns every dataclass under configuration/ that builds with no arguments."""
    found = []

    for module_info in pkgutil.walk_packages(configuration.__path__, "configuration."):
        module = importlib.import_module(module_info.name)

        for name, klass in inspect.getmembers(module, inspect.isclass):
            if klass.__module__ != module.__name__ or not is_dataclass(klass):
                continue
            try:
                klass()
            except TypeError:
                continue
            found.append((f"{module.__name__}.{name}", klass))

    return found


DATACLASSES = _constructible_dataclasses()


def _text_leaves(config) -> list[tuple[str, str]]:
    """Returns (dotted path, text) for every string-valued leaf of a config, Path leaves included."""
    return [(path, value) for path, value in ConfigCli.to_mapping(config).items() if isinstance(value, str)]


def _below_a_root(text: str) -> bool:
    """Returns whether a path text equals a reference root or lies below one."""
    return any(text == root or text.startswith(root + "/") for root in ROOTS)


def test_the_default_dataset_constants_come_from_the_reference_layout():
    """run.py exposes the example dataset and parameter run rather than its own literals."""
    assert DEFAULT_DATASET_DIR == ReferenceLayout.DATASET_DIR
    assert DEFAULT_PARAMS_RUN  == ReferenceLayout.PARAMS_RUN
    assert ReferenceLayout.PARAMETERS == ReferenceLayout.DATASET_DIR / "params" / ReferenceLayout.PARAMS_RUN / "parameters.npy"
    assert not ReferenceLayout.DATA_ROOT.is_absolute()
    assert not ReferenceLayout.RUNS_ROOT.is_absolute()
    assert not ReferenceLayout.RESULTS_ROOT.is_absolute()


@pytest.mark.parametrize("label, klass", DATACLASSES, ids=[label for label, _ in DATACLASSES])
def test_no_configuration_default_is_an_absolute_path(label, klass):
    """Every default is written against the checkout, never against one machine's filesystem."""
    absolute = [(path, text) for path, text in _text_leaves(klass()) if text.startswith("/")]

    assert absolute == [], f"{label} holds absolute defaults: {absolute}; write them below a ReferenceLayout root instead"


@pytest.mark.parametrize("label, klass", DATACLASSES, ids=[label for label, _ in DATACLASSES])
def test_a_profile_setting_every_root_rebases_every_reference_default(label, klass):
    """With all roots set, no default below a reference root still points into the checkout."""
    profile = MachineProfile(None, {"schema": MachineProfile.SCHEMA, **ALL_ROOTS_PROFILE})
    cli     = ConfigCli(klass(), profile=profile)
    stale   = [(path, text) for path, text in _text_leaves(cli.config) if _below_a_root(text)]

    assert stale == [], f"{label} keeps reference paths after rebasing: {stale}"
