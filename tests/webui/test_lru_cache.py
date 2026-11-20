
from __future__ import annotations

import threading


from lru_cache import LruCache


def test_a_missing_key_reads_as_none():
    cache = LruCache(2)

    assert cache.get("nothing") is None
    assert cache.size() == 0


def test_the_oldest_untouched_entry_is_evicted_first():
    cache = LruCache(2)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)

    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None
    assert cache.size() == 2


def test_reputting_a_key_refreshes_it_without_growing_the_cache():
    cache = LruCache(2)

    cache.put("a", 1)
    cache.put("a", 2)

    assert cache.get("a") == 2
    assert cache.size() == 1


def test_concurrent_writers_never_exceed_the_capacity():
    cache   = LruCache(8)
    workers = [threading.Thread(target=lambda index=index: [cache.put(f"{index}-{n}", n) for n in range(200)]) for index in range(8)]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert cache.size() == 8


def test_clearing_drops_every_entry_and_keeps_the_cache_usable():
    cache = LruCache(2)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.clear()

    assert cache.size() == 0
    assert cache.get("a") is None
    assert cache.get("b") is None

    cache.put("c", 3)
    assert cache.get("c") == 3
    assert cache.size() == 1
