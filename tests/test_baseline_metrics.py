"""Derived baseline metrics and opt-in harness persistence."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from vecstudy.core.config import ExperimentConfig, IndexConfig, MaintainConfig, QueryConfig
from vecstudy.harness import runner
from vecstudy.harness.results import make_result_row


def _row(mean=10, p99=25, maximum=40, scans=90, queries=3):
    return make_result_row(config_dict={}, counters_dict={"vectors_scanned": scans},
                           recall=0.5, bloat=0, posting_size_dist={
                               "mean": mean, "p50": 8, "p99": p99, "max": maximum},
                           depth=1, unit_count=1, dataset_checksum="test",
                           query_count=queries)


def test_posting_size_p99_over_mean():
    assert _row()["posting_size_p99_over_mean"] == 2.5


def test_posting_size_max_over_mean():
    assert _row()["posting_size_max_over_mean"] == 4


def test_vectors_scanned_per_query_preserves_total():
    row = _row(scans=91)
    assert row["vectors_scanned"] == 91
    assert row["vectors_scanned_per_query"] == pytest.approx(91 / 3)


def test_zero_denominators():
    row = _row(mean=0, p99=0, maximum=0, scans=0, queries=0)
    for key in ("posting_size_p99_over_mean", "posting_size_max_over_mean",
                "vectors_scanned_per_query"):
        assert row[key] == 0


@pytest.mark.parametrize("query_count", [0, 2])
def test_harness_diagnostics_preserve_behavior_and_persist(monkeypatch, tmp_path, query_count):
    base = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    queries = base[:query_count]
    gt = np.arange(query_count, dtype=np.int32)[:, None]
    monkeypatch.setattr(runner, "_load_dataset", lambda *args: (base, queries, gt))
    exp = ExperimentConfig(index=IndexConfig(dim=2, leaf_size=2, delta=None),
                           query=QueryConfig(k=1, n_postings=2),
                           maintain=MaintainConfig(strategy="inplace"))
    plain = runner.run_experiment(exp, tmp_path)
    diagnosed = runner.run_experiment(exp, tmp_path, diagnostics=True)
    assert "geometry_disagreement_rate" not in plain
    for key, value in plain.items():
        if key not in ("timestamp", "build_time_seconds"):
            assert diagnosed[key] == value
    assert diagnosed["evaluated_vectors"] == 4
    assert diagnosed["owner_vs_ip_mismatch_count"] == 0
    assert diagnosed["vectors_scanned_per_query"] == (4 if query_count else 0)
    # Statistics remain based on owned counts (two owners per posting).
    assert diagnosed["posting_size_mean"] == 2
    rows = pq.read_table(tmp_path / "results.parquet").to_pylist()
    assert len(rows) == 2
    assert rows[0]["geometry_disagreement_count"] is None
    assert rows[1]["geometry_disagreement_count"] == diagnosed["geometry_disagreement_count"]
