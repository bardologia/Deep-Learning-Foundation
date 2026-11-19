
from __future__ import annotations

import re


class AnsiTranscript:

    CSI  = re.compile(r"\x1b\[([0-9;:?]*)[ -/]*([@-~])")
    OSC  = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
    LONE = re.compile(r"\x1b[@-Z\\-_]")
    MOVE = re.compile(r"\x1b\[[0-9;:?]*[ -/]*[@-GJK]")
    CTRL = re.compile(r"[\x00-\x09\x0b-\x1f]")

    def __init__(self) -> None:
        self.lines = [[]]
        self.row   = 0
        self.col   = 0

    def _put(self, run: str) -> None:
        for char in run:
            if char == "\n":
                self.row += 1
                self.col  = 0
                while len(self.lines) <= self.row:
                    self.lines.append([])
            elif char == "\r":
                self.col = 0
            elif char == "\t":
                self.col += 8 - (self.col % 8)
            elif ord(char) >= 32:
                line = self.lines[self.row]
                while len(line) < self.col:
                    line.append(" ")
                if self.col < len(line):
                    line[self.col] = char
                else:
                    line.append(char)
                self.col += 1

    def _count(self, params: str, default: int = 1) -> int:
        head = params.split(";")[0]
        return int(head) if head.isdigit() else default

    def _erase_line(self, params: str) -> None:
        mode = self._count(params, 0)
        line = self.lines[self.row]

        if mode == 0:
            del line[self.col :]
        elif mode == 1:
            for index in range(min(self.col + 1, len(line))):
                line[index] = " "
        else:
            line.clear()

    def _erase_display(self, params: str) -> None:
        mode = self._count(params, 0)

        if mode == 0:
            del self.lines[self.row + 1 :]
            del self.lines[self.row][self.col :]
            return

        self.lines = [[]]
        self.row   = 0
        self.col   = 0

    def _move(self, params: str, final: str) -> None:
        count = self._count(params)

        if final == "A":
            self.row = max(0, self.row - count)
        elif final in ("B", "E"):
            self.row += count
            while len(self.lines) <= self.row:
                self.lines.append([])
        elif final == "C":
            self.col += count
        elif final == "D":
            self.col = max(0, self.col - count)
        elif final == "F":
            self.row = max(0, self.row - count)

        if final in ("E", "F"):
            self.col = 0
        elif final == "G":
            self.col = max(0, self._count(params) - 1)

    def _control(self, params: str, final: str) -> None:
        if params.startswith("?"):
            return

        if final == "K":
            self._erase_line(params)
        elif final == "J":
            self._erase_display(params)
        elif final in ("A", "B", "C", "D", "E", "F", "G"):
            self._move(params, final)

    def _colours_only(self, text: str) -> str | None:
        if self.MOVE.search(text) is not None:
            return None

        plain = self.CSI.sub("", text)
        if self.CTRL.search(plain) is not None:
            return None

        return "\n".join(line.rstrip() for line in plain.split("\n"))

    def flatten(self, text: str) -> str:
        text  = self.LONE.sub("", self.OSC.sub("", text))
        plain = self._colours_only(text)

        if plain is not None:
            return plain

        cursor = 0

        for match in self.CSI.finditer(text):
            self._put(text[cursor : match.start()])
            self._control(match.group(1), match.group(2))
            cursor = match.end()

        self._put(text[cursor:])

        return "\n".join("".join(line).rstrip() for line in self.lines)
