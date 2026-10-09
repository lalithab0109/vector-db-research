"""core/config.py — all configuration dataclasses.

Every config is frozen and can be serialised to a flat dict for result rows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


@dataclass(frozen=True)
class IndexConfig:
    """Static index construction parameters."""
    dim: int
    leaf_size: int = 800            # target vectors per posting
    branching_factor: int = 16
    # None → derive depth from branching_factor via §6.3 loop
    # int  → build exactly this many levels (leaf counted as depth 1)
    depth: int | None = None
    delta: float | None = None      # None → no overlap; relative threshold §6.4
    rng_rule: bool = False          # SPANN relative-neighbourhood rule
    seed: int = 0
    build_mode: Literal["baseline", "r5"] = "baseline"
    target_capacity: int = 800
    lambda_balance: float = 0.25
    alpha: float = 1.5
    tiny_cluster_fraction: float = 0.10
    max_iter: int = 15

    def __post_init__(self) -> None:
        if self.build_mode not in ("baseline", "r5"):
            raise ValueError("build_mode must be baseline or r5")
        if self.build_mode == "r5":
            import math
            if self.delta is not None or self.rng_rule:
                raise ValueError("R5 currently requires delta=None and rng_rule=False")
            if not isinstance(self.target_capacity, int) or self.target_capacity <= 0:
                raise ValueError("target_capacity must be a positive integer")
            if not math.isfinite(self.alpha) or self.alpha <= 0:
                raise ValueError("alpha must be finite and positive")
            if not math.isfinite(self.alpha * self.target_capacity):
                raise ValueError("alpha * target_capacity must be finite")
            if self.branching_factor < 2 or (self.depth is not None and self.depth < 1):
                raise ValueError("R5 requires branching_factor >= 2 and depth >= 1")
            if not math.isfinite(self.lambda_balance) or self.lambda_balance < 0:
                raise ValueError("lambda_balance must be finite and nonnegative")
            if not math.isfinite(self.tiny_cluster_fraction) or not 0 <= self.tiny_cluster_fraction <= 1:
                raise ValueError("tiny_cluster_fraction must lie in [0, 1]")
            if not isinstance(self.max_iter, int) or self.max_iter <= 0:
                raise ValueError("max_iter must be a positive integer")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QueryConfig:
    """Per-query search parameters."""
    k: int = 10
    # per-level survivor counts; last value repeats if shorter than depth
    beam_widths: tuple[int, ...] = (8,)
    n_postings: int = 1             # postings fetched per searchable unit
    overfetch: float = 0.2         # k' = ceil(k * (1 + overfetch))
    global_beam: bool = False       # use descend_global instead of descend

    def to_dict(self) -> dict:
        d = asdict(self)
        d["beam_widths"] = list(self.beam_widths)  # JSON/Parquet friendly
        return d


@dataclass(frozen=True)
class MaintainConfig:
    """Index maintenance parameters."""
    strategy: Literal["inplace", "outofplace"]
    memtable_limit: int = 10_000       # out-of-place: flush threshold
    compaction_threshold: int = 4      # out-of-place: compact at m sealed segments
    split_threshold: int = 1_600       # in-place: 2 × leaf_size
    n_reassign_neighbours: int = 8     # in-place: neighbourhood reassignment width
    tombstone_rebuild_frac: float = 0.2  # in-place: rebuild posting fraction

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration (all three sub-configs combined)."""
    index: IndexConfig
    query: QueryConfig
    maintain: MaintainConfig
    dataset: str = "synthetic"         # dataset name / path key

    def to_dict(self) -> dict:
        return {
            **{f"index_{k}": v for k, v in self.index.to_dict().items()},
            **{f"query_{k}": v for k, v in self.query.to_dict().items()},
            **{f"maintain_{k}": v for k, v in self.maintain.to_dict().items()},
            "dataset": self.dataset,
        }
