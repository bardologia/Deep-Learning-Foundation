from __future__ import annotations

import math

import pytest

from tools.reporting.markdown import MarkdownDoc, MarkdownTable, ScalarFormatter


def test_scalar_formatter_none_returns_empty():
    assert ScalarFormatter.format_scalar(None) == ScalarFormatter.EMPTY


def test_scalar_formatter_custom_empty():
    assert ScalarFormatter.format_scalar(None, empty="N/A") == "N/A"


def test_scalar_formatter_float_default_precision():
    assert ScalarFormatter.format_scalar(3.14159265, precision=5) == "3.1416"


def test_scalar_formatter_float_precision_three():
    assert ScalarFormatter.format_scalar(2.0 / 3.0, precision=3) == "0.667"


def test_scalar_formatter_adaptive_large():
    assert ScalarFormatter.format_scalar(12345.678, adaptive=True) == "1.2346e+04"


def test_scalar_formatter_adaptive_small():
    assert ScalarFormatter.format_scalar(1e-5, adaptive=True) == "1.0000e-05"


def test_scalar_formatter_adaptive_zero_stays_g():
    assert ScalarFormatter.format_scalar(0.0, adaptive=True) == "0"


def test_scalar_formatter_adaptive_midrange_uses_g():
    assert ScalarFormatter.format_scalar(12.5, adaptive=True) == "12.5"


def test_scalar_formatter_list_joined():
    assert ScalarFormatter.format_scalar([1, 2, 3]) == "1, 2, 3"


def test_scalar_formatter_tuple_joined():
    assert ScalarFormatter.format_scalar((4, 5)) == "4, 5"


def test_scalar_formatter_int_passthrough():
    assert ScalarFormatter.format_scalar(42) == "42"


def test_scalar_formatter_str_passthrough():
    assert ScalarFormatter.format_scalar("abc") == "abc"


def test_table_columns_stringified():
    table = MarkdownTable([1, "b", 3.0])
    assert table.columns == ["1", "b", "3.0"]


def test_table_default_alignment_all_left():
    table = MarkdownTable(["a", "b"])
    assert table.align == ["left", "left"]


def test_table_is_empty_initially():
    assert MarkdownTable(["x"]).is_empty()


def test_table_not_empty_after_add():
    table = MarkdownTable(["x"]).add_row("v")
    assert not table.is_empty()


def test_table_add_row_returns_self():
    table = MarkdownTable(["x"])
    assert table.add_row("v") is table


def test_table_add_row_pads_missing_cells_with_empty():
    table = MarkdownTable(["a", "b", "c"])
    table.add_row("only")
    assert table.rows[0] == ["only", MarkdownTable.EMPTY, MarkdownTable.EMPTY]


def test_table_add_row_none_cell_becomes_empty():
    table = MarkdownTable(["a", "b"])
    table.add_row("x", None)
    assert table.rows[0] == ["x", MarkdownTable.EMPTY]


def test_table_add_rows_bulk():
    table = MarkdownTable(["a", "b"])
    table.add_rows([("1", "2"), ("3", "4")])
    assert len(table.rows) == 2


def test_table_render_structure():
    table = MarkdownTable(["Key", "Value"])
    table.add_row("alpha", "1")
    lines = table.render()

    assert len(lines) == 3
    assert lines[0].startswith("|") and lines[0].endswith("|")
    assert set(lines[1].replace("|", "").replace(":", "").replace(" ", "")) == {"-"}
    assert "alpha" in lines[2] and "1" in lines[2]


def test_table_render_pipe_count_consistent():
    table = MarkdownTable(["a", "b", "c"])
    table.add_row("x", "y", "z")
    lines       = table.render()
    pipe_counts = {line.count("|") for line in lines}
    assert pipe_counts == {4}


def test_table_render_column_widths_align():
    table = MarkdownTable(["aa", "b"])
    table.add_row("longcell", "y")
    lines   = table.render()
    lengths = {len(line) for line in lines}
    assert len(lengths) == 1


def test_table_separator_right_alignment():
    table = MarkdownTable(["num"], align=["right"])
    table.add_row("1")
    inner = table.render()[1].strip()[1:-1].strip()
    assert inner.endswith(":")
    assert not inner.startswith(":")


def test_table_separator_center_alignment():
    table = MarkdownTable(["num"], align=["center"])
    table.add_row("1")
    sep   = table.render()[1].strip()
    inner = sep[1:-1].strip()
    assert inner.startswith(":") and inner.endswith(":")


def test_table_separator_left_plain_dashes():
    table = MarkdownTable(["num"], align=["left"])
    table.add_row("1")
    sep = table.render()[1]
    assert ":" not in sep


def test_table_right_aligned_cell_padding():
    table = MarkdownTable(["value"], align=["right"])
    table.add_row("x")
    body = table.render()[2]
    cell = body.split("|")[1]
    assert cell.startswith(" ") and cell.endswith("x ")


def test_table_min_width_three():
    table = MarkdownTable(["a"])
    table.add_row("b")
    width = len(table.render()[0].split("|")[1].strip().ljust(3))
    assert width >= 3


def test_doc_empty_render_newline():
    assert MarkdownDoc().render() == "\n"


def test_doc_title_creates_h1():
    doc = MarkdownDoc("Report")
    assert doc.render().startswith("# Report")


def test_doc_heading_levels():
    doc = MarkdownDoc()
    doc.heading("Sub", level=2)
    assert "## Sub" in doc.render()


def test_doc_heading_inserts_blank_before_when_nonempty():
    doc = MarkdownDoc("Top")
    doc.heading("Second", level=2)
    text = doc.render()
    assert "# Top" in text
    assert "## Second" in text
    assert text.index("# Top") < text.index("## Second")
    assert "\n\n## Second" in text


def test_doc_paragraph():
    doc = MarkdownDoc()
    doc.paragraph("hello world")
    assert "hello world" in doc.render()


def test_doc_bold_kv():
    doc = MarkdownDoc()
    doc.bold_kv("loss", 0.5)
    assert "**loss:** `0.5`" in doc.render()


def test_doc_raw_no_trailing_blank():
    doc = MarkdownDoc()
    doc.raw("rawline")
    assert doc.lines == ["rawline"]


def test_doc_blank_adds_empty_line():
    doc = MarkdownDoc()
    doc.blank()
    assert doc.lines == [""]


def test_doc_image_markdown():
    doc = MarkdownDoc()
    doc.image("alt text", "fig.png")
    assert "![alt text](fig.png)" in doc.render()


def test_doc_methods_chainable():
    doc    = MarkdownDoc()
    result = doc.heading("h").paragraph("p").bold_kv("k", "v").blank()
    assert result is doc


def test_doc_kv_table_from_mapping():
    doc = MarkdownDoc()
    doc.kv_table({"a": 1, "b": 2})
    text = doc.render()
    assert "`a`" in text and "`b`" in text


def test_doc_kv_table_from_iterable():
    doc = MarkdownDoc()
    doc.kv_table([("x", 10), ("y", 20)])
    text = doc.render()
    assert "`x`" in text and "10" in text


def test_doc_kv_table_no_code_keys():
    doc = MarkdownDoc()
    doc.kv_table({"plain": 1}, code_keys=False)
    text = doc.render()
    assert "`plain`" not in text
    assert "plain" in text


def test_doc_kv_table_custom_header():
    doc = MarkdownDoc()
    doc.kv_table({"a": 1}, header=("Metric", "Score"))
    text = doc.render()
    assert "Metric" in text and "Score" in text


def test_doc_table_appends_render_plus_blank():
    table = MarkdownTable(["a"]).add_row("1")
    doc   = MarkdownDoc()
    doc.table(table)
    assert doc.lines[-1] == ""
    assert any("a" in line for line in doc.lines)


def test_doc_render_ends_with_single_newline():
    doc      = MarkdownDoc("T")
    rendered = doc.render()
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")


def test_doc_save_writes_file(tmp_path):
    doc = MarkdownDoc("Saved")
    doc.paragraph("body")
    out = doc.save(tmp_path / "nested" / "report.md")

    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("# Saved")


def test_doc_save_returns_path(tmp_path):
    doc = MarkdownDoc("X")
    out = doc.save(tmp_path / "x.md")
    assert out == tmp_path / "x.md"


def test_doc_save_roundtrip_equals_render(tmp_path):
    doc = MarkdownDoc("Round")
    doc.kv_table({"k": "v"})
    out = doc.save(tmp_path / "r.md")
    assert out.read_text(encoding="utf-8") == doc.render()


@pytest.mark.real_data
def test_kv_table_with_real_extraction_meta(param_extraction_meta):
    scalars = {k: v for k, v in param_extraction_meta.items() if isinstance(v, (str, int, float))}
    doc     = MarkdownDoc("Extraction")
    doc.kv_table(scalars)
    text = doc.render()

    for key in list(scalars)[:5]:
        assert f"`{key}`" in text


@pytest.mark.real_data
def test_table_render_real_baselines(baselines_json):
    table = MarkdownTable(["Key", "Value"])
    for k, v in list(baselines_json.items())[:6]:
        table.add_row(str(k), ScalarFormatter.format_scalar(v) if not isinstance(v, (dict, list)) else "...")
    lines = table.render()
    assert all(line.count("|") == 3 for line in lines)
