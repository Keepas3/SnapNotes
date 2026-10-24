import time
from collections import OrderedDict, deque
from typing import Callable


class SlidingWindowLimiter:
    """In-memory per-key limiter. Fine for one server process serving one person.

    The table is bounded: expired keys are swept regularly, and if more than `max_keys` distinct keys are live at once
    (for example someone inventing a new client address on every request) the least recently seen ones are dropped.
    The worst a flood of fake keys can do is reset the counters of other clients; it cannot grow memory."""

    SWEEP_EVERY = 1024  # calls between sweeps of expired keys

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self.max_keys = max_keys
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()  # least recently seen first
        self._calls = 0

    def __len__(self) -> int:
        return len(self._hits)

    def allow(self, key: str) -> bool:
        now = self.clock()
        self._calls += 1
        if self._calls % self.SWEEP_EVERY == 0:
            self.prune()

        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self.max_keys:
                self.prune()
            while len(self._hits) >= self.max_keys:
                self._hits.popitem(last=False)
            hits = self._hits[key] = deque()
        else:
            self._hits.move_to_end(key)

        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True

    def prune(self) -> None:
        """Forget every key with no hit inside the window."""
        now = self.clock()
        for key in [k for k, h in self._hits.items() if not h or now - h[-1] >= self.window]:
            del self._hits[key]
