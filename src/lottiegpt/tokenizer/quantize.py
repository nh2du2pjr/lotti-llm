"""Per-family numeric (de)quantization.

Each family in schema.NUMERIC_FAMILIES owns a contiguous, disjoint range of
local token ids [0, bins). A numeric value is encoded as
`token(x) = clip(floor((x - lo) / (hi - lo) * bins), 0, bins - 1)` and decoded
from the bin *midpoint* (not the bin edge) to minimize expected reconstruction
error. "exact_int" families use bin width 1 (scale=1), so encode/decode is
lossless as long as the value falls inside [lo, hi].

Values outside [lo, hi] are clipped to the nearest edge bin rather than
raising — out-of-range inputs should be rare (tracked via `Quantizer.stats`)
and a clipped-but-valid token beats a crash during large-corpus tokenization.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lottiegpt.tokenizer.schema import NUMERIC_FAMILIES


@dataclass
class FamilyStats:
    count: int = 0
    clipped: int = 0

    @property
    def clip_rate(self) -> float:
        return self.clipped / self.count if self.count else 0.0


class Quantizer:
    def __init__(self, families: dict[str, dict] = NUMERIC_FAMILIES):
        self.families = families
        self.stats: dict[str, FamilyStats] = {name: FamilyStats() for name in families}

    def bins(self, family: str) -> int:
        spec = self.families[family]
        if spec["kind"] == "exact_int":
            return int(spec["hi"]) - int(spec["lo"]) + 1
        return spec["bins"]

    def encode(self, family: str, value: float) -> int:
        spec = self.families[family]
        lo, hi = spec["lo"], spec["hi"]
        stats = self.stats[family]
        stats.count += 1

        clipped_value = value
        if value < lo:
            clipped_value = lo
            stats.clipped += 1
        elif value > hi:
            clipped_value = hi
            stats.clipped += 1

        if spec["kind"] == "exact_int":
            return int(round(clipped_value)) - int(lo)

        bins = spec["bins"]
        span = hi - lo
        bin_idx = int((clipped_value - lo) / span * bins)
        return max(0, min(bins - 1, bin_idx))

    def decode(self, family: str, local_id: int) -> float:
        spec = self.families[family]
        lo, hi = spec["lo"], spec["hi"]

        if spec["kind"] == "exact_int":
            return float(int(lo) + local_id)

        bins = spec["bins"]
        span = hi - lo
        # bin midpoint minimizes expected quantization error vs. the edge
        return lo + (local_id + 0.5) / bins * span

    def clip_report(self) -> dict[str, float]:
        return {name: s.clip_rate for name, s in self.stats.items() if s.count}
