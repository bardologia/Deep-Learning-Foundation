from __future__ import annotations

from typing import Callable, Hashable

from tools.reporting.markdown import MarkdownTable, ScalarFormatter
from tools.metrics.scoring    import FiniteScalar


class MetricTableRenderer:

    DIRECTION_ARROW = {"higher": " ↑", "lower": " ↓", None: ""}

    @staticmethod
    def _best_values(rows: list, metric_columns: list, orientation: dict, group_of: Callable) -> dict:
        best: dict = {}

        for row in rows:
            group = group_of(row)
            for key, _ in metric_columns:
                direction = orientation.get(key)
                value     = FiniteScalar.coerce(row.metrics.get(key))

                if direction is None or value is None:
                    continue

                current = best.get((group, key))
                if current is None:
                    best[(group, key)] = value
                else:
                    best[(group, key)] = max(current, value) if direction == "higher" else min(current, value)

        return best

    @staticmethod
    def render(
        rows           : list,
        leading        : list[tuple[str, Callable]],
        metric_columns : list[tuple[str, str]],
        orientation    : dict,
        group_of       : Callable[[object], Hashable] | None = None,
        precision      : int = 4,
    ) -> list[str]:
        grouping = group_of if group_of is not None else (lambda row: 0)
        best     = MetricTableRenderer._best_values(rows, metric_columns, orientation, grouping)

        headers  = [header for header, _ in leading]
        headers += [f"{label}{MetricTableRenderer.DIRECTION_ARROW[orientation.get(key)]}" for key, label in metric_columns]

        table = MarkdownTable(headers)

        for row in rows:
            cells = [cell_fn(row) for _, cell_fn in leading]
            group = grouping(row)

            for key, _ in metric_columns:
                value  = row.metrics.get(key)
                cell   = ScalarFormatter.format_scalar(value, precision=precision)
                finite = FiniteScalar.coerce(value)
                mark   = best.get((group, key))

                if mark is not None and finite is not None and finite == mark:
                    cell = f"**{cell}**"

                cells.append(cell)

            table.add_row(*cells)

        return table.render()
