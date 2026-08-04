from __future__ import annotations


class RangeFormatter:
    @staticmethod
    def compact(values: list[int], max_items: int = 6) -> str:
        ranges : list[tuple[int, int]] = []
        start  = values[0]
        prev   = values[0]

        for idx in values[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            ranges.append((start, prev))
            start = idx
            prev  = idx

        ranges.append((start, prev))

        parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in ranges[:max_items]]
        if len(ranges) > max_items:
            parts.append("...")

        return ", ".join(parts)
