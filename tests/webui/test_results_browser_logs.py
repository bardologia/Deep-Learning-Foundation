from __future__ import annotations

from pathlib import Path

from results_browser import ResultsBrowser
from web_logger      import WebLogger


def _browser(root: Path) -> ResultsBrowser:
    browser = ResultsBrowser(WebLogger())
    assert browser.tree(str(root))["ok"]
    return browser


def test_folder_buckets_log_files_with_content(tmp_path):
    (tmp_path / "inference.log").write_text("line1\nline2\n")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "weights.bin").write_bytes(b"\x00\x01")

    payload = _browser(tmp_path).folder(str(tmp_path), "")

    assert [log["name"] for log in payload["logs"]] == ["inference.log", "notes.txt"]
    assert payload["logs"][0]["text"] == "line1\nline2\n"
    assert payload["logs"][0]["size"] == len("line1\nline2\n")
    assert [entry["name"] for entry in payload["other"]] == ["weights.bin"]


def test_large_log_serves_head(tmp_path):
    (tmp_path / "big.log").write_bytes(b"THE_START" + b"x" * 300000)

    payload = _browser(tmp_path).folder(str(tmp_path), "")

    text = payload["logs"][0]["text"]
    assert text.startswith("THE_START")
    assert text.endswith("[truncated: showing the first 256 KB of 292 KB]")
    assert len(text) < 300000


def test_tree_counts_logs(tmp_path):
    (tmp_path / "a.log").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.json").write_text("{}")
    (tmp_path / "d.out").write_text("d")

    tree = _browser(tmp_path).tree(str(tmp_path))["tree"]

    assert tree["counts"]["logs"] == 3
    assert tree["counts"]["configs"] == 1
    assert tree["counts"]["other"] == 0


def test_ansi_escapes_are_stripped(tmp_path):
    raw = "\x1b[36m────\x1b[0m \x1b[1;36minference\x1b[0m\n  \x1b[1;35mStarted\x1b[0m  : \x1b[97m2026-07-04\x1b[0m\n\x1b]0;title\x07plain\n"
    (tmp_path / "console.out").write_text(raw)

    payload = _browser(tmp_path).folder(str(tmp_path), "")

    text = payload["logs"][0]["text"]
    assert "\x1b" not in text
    assert text == "──── inference\n  Started  : 2026-07-04\nplain\n"


def test_carriage_return_overwrites_keep_last_frame(tmp_path):
    raw = "start\nprogress 10%\rprogress 50%\rprogress 100%\nwindows line\r\ndone\x07\n"
    (tmp_path / "train.log").write_text(raw)

    payload = _browser(tmp_path).folder(str(tmp_path), "")

    text = payload["logs"][0]["text"]
    assert "\r" not in text and "\x07" not in text
    assert text == "start\nprogress 100%\nwindows line\ndone\n"
