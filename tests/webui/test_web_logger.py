from __future__ import annotations

import re

import pytest

from results_browser import ResultsBrowser
from web_logger      import WebLogger


@pytest.fixture
def plain():
    logger         = WebLogger()
    logger.enabled = False
    return logger


@pytest.fixture
def coloured():
    logger         = WebLogger()
    logger.enabled = True
    return logger


def _lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_every_level_prints_its_own_tag(plain, capsys):
    plain.info("started")
    plain.ok("done")
    plain.warning("slow")
    plain.error("crashed")
    plain.muted("detail")

    tags = [line.split("]")[1].strip().split(" ")[0] for line in _lines(capsys)]

    assert tags == ["INFO", "OK", "WARN", "ERROR", "MUTED"]


def test_a_plain_line_carries_a_timestamp_and_the_message(plain, capsys):
    plain.info("job 1234 started")
    line = _lines(capsys)[0]

    assert re.match(r"^\[\d{2}:\d{2}:\d{2}\] INFO  job 1234 started$", line), line


def test_colour_is_only_emitted_for_a_terminal(plain, coloured, capsys):
    plain.error("crashed")
    assert "\x1b" not in _lines(capsys)[0]

    coloured.error("crashed")
    line = _lines(capsys)[0]

    assert line.startswith(WebLogger.COLORS["ERROR"])
    assert WebLogger.RESET in line


def test_a_coloured_line_survives_the_log_reader_that_strips_it(coloured, capsys):
    coloured.warning("gpu 0 is busy")
    raw = _lines(capsys)[0]

    cleaned = ResultsBrowser(coloured)._clean_log(raw)

    assert "\x1b" not in cleaned
    assert cleaned.endswith("WARN  gpu 0 is busy")


def test_the_banner_box_is_wide_enough_for_its_widest_line(plain, capsys):
    plain.banner("Console", ["url  http://127.0.0.1:8765", "envs 2"])
    lines = _lines(capsys)

    bar = lines[0].split("] OK    ")[1]

    assert lines[0] == lines[2]
    assert set(bar) == {"="}
    assert len(bar) == len("url  http://127.0.0.1:8765") + 4
    assert lines[1].endswith("  Console")
    assert [line.split("] INFO  ")[1] for line in lines[3:]] == ["  url  http://127.0.0.1:8765", "  envs 2"]


def test_a_banner_with_no_body_still_draws_its_box(plain, capsys):
    plain.banner("Ready", [])
    lines = _lines(capsys)

    assert len(lines) == 3
    assert lines[0].endswith("=" * (len("Ready") + 4))
