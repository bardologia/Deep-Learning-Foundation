from __future__ import annotations

from pathlib import Path
from typing  import Any, Iterable, Mapping, Optional, Sequence


class ScalarFormatter:

    EMPTY = "—"

    @staticmethod
    def format_scalar(value: Any, precision: int = 5, adaptive: bool = False, empty: str = EMPTY) -> str:
        if value is None:
            return empty

        if isinstance(value, float):
            if adaptive and (abs(value) >= 1e4 or 0 < abs(value) < 1e-3):
                return f"{value:.4e}"
            return f"{value:.{precision}g}"

        if isinstance(value, (list, tuple)):
            return ", ".join(str(x) for x in value)

        return str(value)


class MarkdownTable:

    EMPTY = "—"

    def __init__(self, columns: Sequence[Any], align: Optional[Sequence[str]] = None) -> None:
        self.columns = [str(c) for c in columns]
        self.align   = list(align) if align is not None else ["left"] * len(self.columns)
        self.rows    = []

    def add_row(self, *cells: Any) -> "MarkdownTable":
        padded = list(cells) + [None] * (len(self.columns) - len(cells))
        self.rows.append([self.EMPTY if c is None else str(c) for c in padded])
        return self

    def add_rows(self, rows: Iterable[Sequence[Any]]) -> "MarkdownTable":
        for row in rows:
            self.add_row(*row)
        return self

    def is_empty(self) -> bool:
        return not self.rows

    def _widths(self) -> list[int]:
        widths = [max(3, len(c)) for c in self.columns]
        for row in self.rows:
            widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
        return widths

    def _separator(self, width: int, align: str) -> str:
        if align == "right":
            return "-" * (width - 1) + ":"
        if align == "center":
            return ":" + "-" * (width - 2) + ":"
        return "-" * width

    def _aligned(self, cell: str, width: int, align: str) -> str:
        if align == "right":
            return f"{cell:>{width}}"
        if align == "center":
            return f"{cell:^{width}}"
        return f"{cell:<{width}}"

    def _row_line(self, cells: Sequence[str], widths: Sequence[int]) -> str:
        body = " | ".join(self._aligned(c, w, a) for c, w, a in zip(cells, widths, self.align))
        return f"| {body} |"

    def render(self) -> list[str]:
        widths    = self._widths()
        separator = "| " + " | ".join(self._separator(w, a) for w, a in zip(widths, self.align)) + " |"

        lines  = [self._row_line(self.columns, widths), separator]
        lines += [self._row_line(row, widths) for row in self.rows]
        return lines


class MarkdownDoc:
    def __init__(self, title: Optional[str] = None) -> None:
        self.lines: list[str] = []
        if title is not None:
            self.heading(title, level=1)

    def heading(self, text: str, level: int = 1) -> "MarkdownDoc":
        if self.lines:
            self.lines.append("")
        self.lines.append(f"{'#' * level} {text}")
        self.lines.append("")
        return self

    def paragraph(self, text: str) -> "MarkdownDoc":
        self.lines.append(str(text))
        self.lines.append("")
        return self

    def raw(self, text: str) -> "MarkdownDoc":
        self.lines.append(str(text))
        return self

    def blank(self) -> "MarkdownDoc":
        self.lines.append("")
        return self

    def bold_kv(self, key: str, value: Any) -> "MarkdownDoc":
        self.lines.append(f"**{key}:** `{value}`")
        return self

    def kv_table(self, items: Mapping[str, Any] | Iterable[tuple], header: Sequence[str] = ("Key", "Value"), code_keys: bool = True) -> "MarkdownDoc":
        table = MarkdownTable(header)
        pairs = items.items() if isinstance(items, Mapping) else items

        for k, v in pairs:
            key = f"`{k}`" if code_keys else str(k)
            table.add_row(key, v)

        return self.table(table)

    def table(self, table: MarkdownTable) -> "MarkdownDoc":
        self.lines.extend(table.render())
        self.lines.append("")
        return self

    def image(self, alt: str, path: Any) -> "MarkdownDoc":
        self.lines.append(f"![{alt}]({path})")
        self.lines.append("")
        return self

    def render(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"

    def save(self, path: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path
