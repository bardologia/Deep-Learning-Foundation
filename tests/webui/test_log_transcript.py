
from __future__ import annotations

import re

from log_transcript import AnsiTranscript


def test_plain_text_is_untouched():
    assert AnsiTranscript().flatten("line1\nline2\n") == "line1\nline2\n"


def test_sgr_colours_are_dropped():
    raw = "\x1b[36m────\x1b[0m \x1b[1;36minference\x1b[0m\n"
    assert AnsiTranscript().flatten(raw) == "──── inference\n"


def test_carriage_return_keeps_the_last_frame():
    raw = "progress 10%\rprogress 50%\rprogress 100%\n"
    assert AnsiTranscript().flatten(raw) == "progress 100%\n"


def test_cursor_up_and_erase_collapse_a_redrawn_block():
    raw = "header\nvalue 1\nvalue 2\n\x1b[2A\x1b[2Kvalue 7\n\x1b[2Kvalue 8\n"
    assert AnsiTranscript().flatten(raw) == "header\nvalue 7\nvalue 8\n"


def test_cursor_up_past_the_top_clamps():
    raw = "a\n\x1b[9A\x1b[2Kz\n"
    assert AnsiTranscript().flatten(raw) == "z\n"


def test_erase_to_end_of_line_truncates():
    raw = "keep this and drop that\x1b[14D\x1b[0K\n"
    assert AnsiTranscript().flatten(raw) == "keep this\n"


def test_erase_display_after_cursor_drops_following_lines():
    raw = "one\ntwo\nthree\n\x1b[2A\x1b[0Jrewritten\n"
    assert AnsiTranscript().flatten(raw) == "one\nrewritten\n"


def test_column_absolute_positions_the_cursor():
    raw = "abcdef\x1b[3Gxy\n"
    assert AnsiTranscript().flatten(raw) == "abxyef\n"


def test_cursor_visibility_and_osc_titles_are_ignored():
    raw = "\x1b[?25lplain\x1b]0;a title\x07 text\x1b[?25h\n"
    assert AnsiTranscript().flatten(raw) == "plain text\n"


def test_control_characters_are_dropped():
    assert AnsiTranscript().flatten("done\x07\n") == "done\n"


def test_trailing_spaces_are_trimmed_per_line():
    assert AnsiTranscript().flatten("padded      \nnext\n") == "padded\nnext\n"


def test_repeated_live_frames_collapse_to_the_last_one():
    frame  = "╭── Monitor ──╮\n│ epoch {0}/9 │\n╰─────────────╯\n"
    stream = frame.format(1) + "".join("\x1b[3A" + frame.format(step) for step in range(2, 10))

    flat = AnsiTranscript().flatten(stream)

    assert flat.count("Monitor") == 1
    assert "epoch 9/9" in flat
    assert "epoch 8/9" not in flat


def test_the_colour_only_fast_path_matches_the_emulator():
    raw       = "".join(f"\x1b[32mepoch {step}\x1b[0m loss \x1b[1m0.{step}\x1b[0m\n" for step in range(200))
    slow      = AnsiTranscript()
    slow.MOVE = re.compile(r"\x1b\[")

    assert AnsiTranscript().flatten(raw) == slow.flatten(raw)
    assert "\x1b" not in AnsiTranscript().flatten(raw)


def test_a_carriage_return_still_reaches_the_emulator():
    assert AnsiTranscript().flatten("\x1b[32mfirst\x1b[0m\rsecond\n") == "second\n"
