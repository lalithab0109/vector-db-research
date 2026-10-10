"""scripts/analyze_boundary.py — Boundary statistic r = d2/d1 - 1 analysis and visualization.

Analyzes SIFT1M vectors against the 1,250 R5 balanced centroids:
  1. Computes d1, d2, r = d2/d1 - 1, c1, and c2 for all 1,000,000 vectors in chunks.
  2. Plots and saves histogram and empirical CDF curves.
  3. Evaluates quantiles (mean, median, p1..p99) and boundary proximity counts.
  4. Analyzes primary ownership (how often R5 assigned vectors to c1 vs c2).
  5. Exports a Markdown summary to reports/BOUNDARY_ANALYSIS.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vecstudy.harness.datasets import fetch_sift, read_fvecs, l2_normalize
from vecstudy.core.boundary import compute_boundary_statistics, summarize_boundary_distribution
from vecstudy.core.config import IndexConfig
from vecstudy.core.metrics import Counters
from vecstudy.index.segment import Segment


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze boundary statistic r on SIFT1M with R5 centroids")
    parser.add_argument("--cache-segment", default="data/sift_r5_segment.pkl",
                        help="Path to cached R5 segment (default: data/sift_r5_segment.pkl)")
    parser.add_argument("--geometry", choices=["l2", "ip"], default="l2",
                        help="Distance geometry: l2 or ip (default: l2)")
    parser.add_argument("--chunk-size", type=int, default=32768,
                        help="Chunk size for memory-safe evaluation (default: 32768)")
    parser.add_argument("--plots-dir", default="reports/plots",
                        help="Directory to save generated plots (default: reports/plots)")
    parser.add_argument("--out-md", default="reports/BOUNDARY_ANALYSIS.md",
                        help="Output report markdown path (default: reports/BOUNDARY_ANALYSIS.md)")
    return parser.parse_args()


def load_data_and_centroids(cache_path: Path):
    print("Loading SIFT1M dataset ...", flush=True)
    sift_dir = fetch_sift()
    base = l2_normalize(read_fvecs(sift_dir / "sift_base.fvecs"))
    print(f"Loaded {len(base)} base vectors (dim={base.shape[1]}).", flush=True)

    if cache_path.exists():
        print(f"Loading cached R5 segment from {cache_path} ...", flush=True)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        seg = cached["segment"]
    else:
        print(f"Segment cache not found at {cache_path}, building R5 segment ...", flush=True)
        cfg = IndexConfig(
            dim=128, branching_factor=16, depth=1, seed=0,
            build_mode="r5", target_capacity=800, lambda_balance=0.25,
            alpha=1.5, tiny_cluster_fraction=0.10, max_iter=15,
        )
        counters = Counters()
        vids = np.arange(len(base), dtype=np.int64)
        seg = Segment.build(vids, base, cfg, counters)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"segment": seg, "config": cfg}, f)

    centroids = np.asarray(seg.levels[-1].centroids, dtype=np.float32)
    print(f"Centroids: {centroids.shape} ({len(centroids)} centroids).", flush=True)

    # Reconstruct authoritative owner IDs per vector
    owners = np.empty(len(base), dtype=np.int64)
    for p in seg.postings:
        owners[p.owned] = p.id

    return base, centroids, owners, seg


def plot_histogram_and_cdf(r: np.ndarray, summary: dict, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    r_finite = r[np.isfinite(r)]

    # 1. Histogram of r
    plt.figure(figsize=(9, 5.5), dpi=150)
    # Clip extreme tail at p99.5 for clear visual presentation
    cap_val = float(np.percentile(r_finite, 99.5))
    r_vis = r_finite[r_finite <= cap_val]

    counts, bins, patches = plt.hist(r_vis, bins=100, density=True, color="#2b5c8f",
                                     edgecolor="white", alpha=0.85, label="Empirical Density")
    plt.axvline(summary["mean"], color="#d9534f", linestyle="--", linewidth=2,
                label=f"Mean r = {summary['mean']:.4f}")
    plt.axvline(summary["median"], color="#f0ad4e", linestyle="-.", linewidth=2,
                label=f"Median r = {summary['median']:.4f}")
    plt.axvline(summary["quantiles"]["p10"], color="#5cb85c", linestyle=":", linewidth=1.5,
                label=f"p10 = {summary['quantiles']['p10']:.4f}")
    plt.axvline(summary["quantiles"]["p90"], color="#5bc0de", linestyle=":", linewidth=1.5,
                label=f"p90 = {summary['quantiles']['p90']:.4f}")

    plt.title(f"SIFT1M Boundary Statistic Distribution: $r = d_2 / d_1 - 1$\n(Mean = {summary['mean']:.4f}, Median = {summary['median']:.4f})", fontsize=12, fontweight="bold")
    plt.xlabel("Boundary Distance Ratio $r$", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    hist_path = plots_dir / "r_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved histogram to {hist_path}", flush=True)

    # 2. Empirical CDF
    plt.figure(figsize=(9, 5.5), dpi=150)
    sorted_r = np.sort(r_finite)
    cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)

    # Zoom in on r in [0, 0.40] where replication decisions matter
    plt.plot(sorted_r, cdf * 100.0, color="#1f77b4", linewidth=2.5, label=r"Empirical CDF: $P(r \leq \delta)$")

    key_thresholds = [0.02, 0.05, 0.10, 0.15, 0.20]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    for t, c in zip(key_thresholds, colors):
        frac = float(np.mean(r <= t)) * 100.0
        plt.scatter([t], [frac], color=c, s=50, zorder=5)
        plt.plot([t, t], [0, frac], color=c, linestyle=":", alpha=0.7)
        plt.plot([0, t], [frac, frac], color=c, linestyle=":", alpha=0.7)
        plt.annotate(f"$\\delta$={t:.2f}: {frac:.1f}%", xy=(t, frac),
                     xytext=(t + 0.012, frac - 2.5),
                     fontsize=9, fontweight="bold", color=c)

    plt.xlim(0.0, 0.35)
    plt.ylim(0.0, 100.0)
    plt.title("Empirical CDF of Boundary Statistic $r$ (Predicted Replication Bloat)", fontsize=12, fontweight="bold")
    plt.xlabel(r"Overlap Threshold $\delta$ (where $d_2 \leq d_1 (1 + \delta)$)", fontsize=11)
    plt.ylabel("Cumulative Boundary Vectors (% of Dataset = Predicted Bloat %)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    cdf_path = plots_dir / "r_empirical_cdf.png"
    plt.savefig(cdf_path)
    plt.close()
    print(f"Saved CDF plot to {cdf_path}", flush=True)


def main():
    args = parse_args()
    base, centroids, owners, seg = load_data_and_centroids(REPO_ROOT / args.cache_segment)

    print(f"\nComputing boundary statistics (geometry={args.geometry}, chunk_size={args.chunk_size}) ...", flush=True)
    t0 = time.time()
    d1, d2, r, c1, c2 = compute_boundary_statistics(
        base, centroids, geometry=args.geometry, chunk_size=args.chunk_size
    )
    elapsed = time.time() - t0
    print(f"Computed boundary stats for {len(base):,} vectors in {elapsed:.2f}s ({len(base)/elapsed:,.0f} vec/s).", flush=True)

    summary = summarize_boundary_distribution(
        r, thresholds=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    )

    # Primary ownership assignment analysis under R5
    c1_match = np.count_nonzero(owners == c1)
    c2_match = np.count_nonzero(owners == c2)
    other_match = len(base) - c1_match - c2_match

    # Generate plots
    plots_dir = REPO_ROOT / args.plots_dir
    plot_histogram_and_cdf(r, summary, plots_dir)

    # Print distribution table
    print("\n" + "=" * 80)
    print("BOUNDARY STATISTIC DISTRIBUTION SUMMARY (r = d2/d1 - 1)")
    print("=" * 80)
    print(f"  Total Vectors:        {summary['count']:,}")
    print(f"  Finite R Count:       {summary['finite_count']:,}")
    print(f"  Mean r:               {summary['mean']:.4f}")
    print(f"  Median r (p50):       {summary['median']:.4f}")
    print(f"  Standard Deviation:   {summary['std']:.4f}")
    print(f"  Min r:                {summary['min']:.6f}")
    print(f"  Max r:                {summary['max']:.4f}")
    print("\n  Important Quantiles:")
    for k, v in summary["quantiles"].items():
        print(f"    {k:>3}: {v:.4f}")

    print("\n" + "-" * 80)
    print("PROXIMITY TO CLUSTER BOUNDARIES & PREDICTED REPLICATION BLOAT")
    print("-" * 80)
    print(f"  {'Threshold delta':>16} | {'Boundary Vectors Count':>22} | {'% of Dataset (Predicted Bloat)':>30}")
    print("  " + "-" * 74)
    for row in summary["thresholds"]:
        print(f"  {row['threshold']:>16.2f} | {row['count']:>22,} | {row['percentage']:>29.2f}%")

    print("\n" + "-" * 80)
    print("R5 PRIMARY CLUSTER ASSIGNMENT ANALYSIS")
    print("-" * 80)
    print(f"  Assigned to nearest centroid (c1):        {c1_match:,} ({c1_match/len(base)*100:.2f}%)")
    print(f"  Assigned to 2nd nearest centroid (c2):     {c2_match:,} ({c2_match/len(base)*100:.2f}%)")
    print(f"  Assigned to other centroid (> 2nd):       {other_match:,} ({other_match/len(base)*100:.2f}%)")
    print("=" * 80 + "\n")

    # Save Markdown report
    out_md = REPO_ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md_content = f"""# SIFT1M Boundary Statistic Analysis ($r = d_2 / d_1 - 1$)

- **Dataset:** SIFT1M (1,000,000 vectors, 128-D)
- **Centroids:** {len(centroids):,} R5 balanced leaf centroids ($C_{{\\text{{target}}}} = 800$, $\\alpha = 1.5$)
- **Geometry:** `{args.geometry}` (Euclidean distance)
- **Evaluation Time:** {elapsed:.2f}s

---

## 1. Boundary Statistic ($r$) Distribution

| Metric | Measured Value | Description / Interpretation |
| :--- | :---: | :--- |
| **Mean $r$** | **{summary['mean']:.4f}** | Average relative distance buffer to the 2nd centroid |
| **Median ($p50$)** | **{summary['median']:.4f}** | 50% of vectors have $r \le {summary['median']:.4f}$ |
| **Std Dev** | **{summary['std']:.4f}** | Spread of the boundary distance ratio |
| **Min $r$** | **{summary['min']:.6f}** | Nearest boundary vector ($d_1 \\approx d_2$) |
| **Max $r$** | **{summary['max']:.4f}** | Most interior / isolated vector |
| **$p1$** | **{summary['quantiles']['p1']:.4f}** | Top 1% closest vectors to boundary |
| **$p5$** | **{summary['quantiles']['p5']:.4f}** | Top 5% closest vectors to boundary |
| **$p10$** | **{summary['quantiles']['p10']:.4f}** | Top 10% closest vectors to boundary |
| **$p25$** | **{summary['quantiles']['p25']:.4f}** | Lower quartile |
| **$p75$** | **{summary['quantiles']['p75']:.4f}** | Upper quartile |
| **$p90$** | **{summary['quantiles']['p90']:.4f}** | 90th percentile interior vectors |
| **$p95$** | **{summary['quantiles']['p95']:.4f}** | 95th percentile interior vectors |
| **$p99$** | **{summary['quantiles']['p99']:.4f}** | 99th percentile deep interior vectors |

---

## 2. Vectors Close to Cluster Boundaries (Predicted Replication Bloat)

For relative overlap threshold $\\delta$, a vector is replicated to its 2nd nearest centroid iff $d_2 \\le d_1 (1 + \\delta) \\iff r \\le \\delta$:

| Overlap Threshold $\\delta$ | Boundary Vector Count ($r \\le \\delta$) | Cumulative % (Predicted Bloat) |
| :---: | :---: | :---: |
"""
    for row in summary["thresholds"]:
        md_content += f"| **{row['threshold']:.2f}** | {row['count']:,} | **{row['percentage']:.2f}%** |\n"

    md_content += f"""
---

## 3. R5 Primary Ownership Concordance

Under R5 capacity constraints ($C_{{\\max}} = 1,200$), when a cluster is full, vectors can be guided to an underfilled neighbor:

| Assignment Target | Vector Count | Percentage |
| :--- | :---: | :---: |
| **Nearest Centroid ($c_1$)** | **{c1_match:,}** | **{c1_match/len(base)*100:.2f}%** |
| **2nd Nearest Centroid ($c_2$)** | **{c2_match:,}** | **{c2_match/len(base)*100:.2f}%** |
| **3rd+ Nearest Centroid ($> 2$)** | **{other_match:,}** | **{other_match/len(base)*100:.2f}%** |

---

## 4. Generated Plots
- **Histogram of $r$:** [`reports/plots/r_histogram.png`](file:///{REPO_ROOT.as_posix()}/reports/plots/r_histogram.png)
- **Empirical CDF Curve:** [`reports/plots/r_empirical_cdf.png`](file:///{REPO_ROOT.as_posix()}/reports/plots/r_empirical_cdf.png)
"""
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Markdown report to {out_md}", flush=True)


if __name__ == "__main__":
    main()
