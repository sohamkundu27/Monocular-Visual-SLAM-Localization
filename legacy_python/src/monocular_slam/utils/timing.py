"""Lightweight runtime instrumentation.

Stage timings are accumulated with ``time.perf_counter`` around the four
front-end stages (detection, matching, pose estimation, loop closure) plus
optimization. The overhead is a pair of clock reads per call, which is
negligible next to ORB detection at ~50 ms/frame, so instrumentation is always
on rather than behind a flag.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageStats:
    """Accumulated timings for one named stage."""

    total_s: float = 0.0
    count: int = 0
    max_s: float = 0.0

    @property
    def mean_ms(self) -> float:
        return (self.total_s / self.count * 1000.0) if self.count else 0.0

    @property
    def max_ms(self) -> float:
        return self.max_s * 1000.0


@dataclass
class StageTimer:
    """Accumulates wall-clock time per named stage.

    Example
    -------
    >>> timer = StageTimer()
    >>> with timer.time("detect"):
    ...     pass
    >>> timer.stats["detect"].count
    1
    """

    stats: dict[str, StageStats] = field(default_factory=lambda: defaultdict(StageStats))
    _start: float = field(default_factory=time.perf_counter)

    @contextmanager
    def time(self, stage: str):
        """Time the enclosed block and add it to ``stage``."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - started)

    def record(self, stage: str, seconds: float) -> None:
        entry = self.stats[stage]
        entry.total_s += seconds
        entry.count += 1
        entry.max_s = max(entry.max_s, seconds)

    def reset_wall_clock(self) -> None:
        """Restart the total-runtime clock (call when real work begins)."""
        self._start = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        """Wall-clock seconds since construction or the last reset."""
        return time.perf_counter() - self._start

    def total_s(self, stage: str) -> float:
        return self.stats[stage].total_s if stage in self.stats else 0.0

    def mean_ms(self, stage: str) -> float:
        return self.stats[stage].mean_ms if stage in self.stats else 0.0

    def fps(self, n_frames: int) -> float:
        """Frames per second over the wall-clock window."""
        elapsed = self.elapsed_s
        return float(n_frames / elapsed) if elapsed > 0 else 0.0

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-stage totals in a JSON-serialisable form."""
        return {
            stage: {
                "total_s": round(entry.total_s, 4),
                "mean_ms": round(entry.mean_ms, 3),
                "max_ms": round(entry.max_ms, 3),
                "calls": entry.count,
            }
            for stage, entry in sorted(self.stats.items())
        }

    def report(self) -> str:
        """Human-readable table for the run log."""
        if not self.stats:
            return "no timing data"
        width = max(len(s) for s in self.stats)
        lines = [f"{'stage'.ljust(width)}   total_s    mean_ms     max_ms   calls"]
        for stage, entry in sorted(self.stats.items(), key=lambda kv: -kv[1].total_s):
            lines.append(
                f"{stage.ljust(width)} {entry.total_s:9.3f} {entry.mean_ms:10.3f} "
                f"{entry.max_ms:10.3f} {entry.count:7d}"
            )
        return "\n".join(lines)
