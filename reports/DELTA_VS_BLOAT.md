# Experimental Verification: Bloat $\approx F_R(\delta)$

- **Dataset:** SIFT1M (1,000,000 vectors, 128-D)
- **Centroids:** 1,250 R5 Leaf Centroids
- **Replication Rule:** Top-2 assignment with relative threshold $\delta$ (at most one replica per vector)

---

## 1. Experimental Results vs. Empirical CDF

| $\delta$ | Empirical CDF $F_R(\delta)$ | Replicated Vectors ($N_{\text{rep}}$) | Measured Index Bloat | Absolute Difference $|\text{Bloat} - F_R(\delta)|$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01** | **14.468%** | 144,679 | **14.468%** | 0.0000% | **Exact Match** |
| **0.02** | **26.826%** | 268,263 | **26.826%** | 0.0000% | **Exact Match** |
| **0.05** | **53.784%** | 537,843 | **53.784%** | 0.0000% | **Exact Match** |
| **0.10** | **77.673%** | 776,735 | **77.673%** | 0.0000% | **Exact Match** |
| **0.15** | **88.577%** | 885,770 | **88.577%** | 0.0000% | **Exact Match** |
| **0.20** | **93.730%** | 937,303 | **93.730%** | 0.0000% | **Exact Match** |

---

## 2. Mathematical Proof of Equivalence

1. Under the top-2 relative overlap rule (`vecstudy.index.postings.assign`):
   - Every vector $x_i$ is assigned to its primary centroid $c_1$ ($N$ owned vectors).
   - If $d_2 \le d_1 (1 + \delta) \iff r_i \le \delta$, vector $x_i$ is additionally assigned as a replica to $c_2$.
2. The total number of replicas added across the dataset is:
   $$N_{\text{rep}} = \sum_{i=1}^N \mathbb{I}(r_i \le \delta)$$
3. The realised storage bloat is formally defined as:
   $$\text{Bloat} = \frac{\text{Total Members}}{N} - 1 = \frac{N + N_{\text{rep}}}{N} - 1 = \frac{N_{\text{rep}}}{N}$$
4. The empirical cumulative distribution function (CDF) of $r$ is:
   $$F_R(\delta) = \frac11000000 \sum_{i=1}^N \mathbb{I}(r_i \le \delta) = \frac{N_{\text{rep}}}{N}$$
5. Therefore, with one extra replica:
   $$\mathbf{\text{Bloat} \equiv F_R(\delta)}$$

The experimental results confirm this identity with **zero discrepancy** across all tested $\delta$ thresholds.

---

## 3. Verification Plot
- **Plot:** [`reports/plots/delta_vs_bloat_verification.png`](file:///C:/Users/viral/OneDrive/Desktop/Hiya/IITG/VectorDB/vector-db-research/reports/plots/delta_vs_bloat_verification.png)
