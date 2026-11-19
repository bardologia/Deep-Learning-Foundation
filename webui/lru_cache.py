
from __future__ import annotations

import threading
from collections import OrderedDict


class LruCache:

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"an LRU cache needs room for at least one entry, got {capacity}")

        self.capacity = capacity
        self.lock     = threading.Lock()
        self.entries  = OrderedDict()

    def get(self, key: str):
        with self.lock:
            if key not in self.entries:
                return None

            self.entries.move_to_end(key)
            return self.entries[key]

    def put(self, key: str, value) -> None:
        with self.lock:
            self.entries[key] = value
            self.entries.move_to_end(key)

            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()

    def size(self) -> int:
        with self.lock:
            return len(self.entries)
