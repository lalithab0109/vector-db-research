# SIFT1M Boundary Statistic Analysis ($r = d_2 / d_1 - 1$)

- **Dataset:** SIFT1M (1,000,000 vectors, 128-D)
- **Centroids:** 1,250 R5 balanced leaf centroids ($C_{\text{target}} = 800$, $\alpha = 1.5$)
- **Geometry:** `l2` (Euclidean distance)
- **Evaluation Time:** 24.18s

---

## 1. Boundary Statistic ($r$) Distribution

| Metric | Measured Value | Description / Interpretation |
| :--- | :---: | :--- |
| **Mean $r$** | **0.0742** | Average relative distance buffer to the 2nd centroid |
| **Median ($p50$)** | **0.0482** | 50% of vectors have $r \le 0.0482$ |
| **Std Dev** | **0.0844** | Spread of the boundary distance ratio |
| **Min $r$** | **0.000000** | Nearest boundary vector ($d_1 \approx d_2$) |
| **Max $r$** | **1.3133** | Most interior / isolated vector |
| **$p1$** | **0.0007** | Top 1% closest vectors to boundary |
| **$p5$** | **0.0036** | Top 5% closest vectors to boundary |
| **$p10$** | **0.0075** | Top 10% closest vectors to boundary |
| **$p25$** | **0.0203** | Lower quartile |
| **$p75$** | **0.0974** | Upper quartile |
| **$p90$** | **0.1691** | 90th percentile interior vectors |
| **$p95$** | **0.2310** | 95th percentile interior vectors |
| **$p99$** | **0.4122** | 99th percentile deep interior vectors |

---

## 2. Vectors Close to Cluster Boundaries (Predicted Replication Bloat)

For relative overlap threshold $\delta$, a vector is replicated to its 2nd nearest centroid iff $d_2 \le d_1 (1 + \delta) \iff r \le \delta$:

| Overlap Threshold $\delta$ | Boundary Vector Count ($r \le \delta$) | Cumulative % (Predicted Bloat) |
| :---: | :---: | :---: |
| **0.01** | 130,857 | **13.09%** |
| **0.02** | 246,593 | **24.66%** |
| **0.05** | 512,924 | **51.29%** |
| **0.10** | 758,663 | **75.87%** |
| **0.15** | 873,925 | **87.39%** |
| **0.20** | 929,971 | **93.00%** |
| **0.25** | 958,844 | **95.88%** |
| **0.30** | 974,375 | **97.44%** |

---

## 3. R5 Primary Ownership Concordance

Under R5 capacity constraints ($C_{\max} = 1,200$), when a cluster is full, vectors can be guided to an underfilled neighbor:

| Assignment Target | Vector Count | Percentage |
| :--- | :---: | :---: |
| **Nearest Centroid ($c_1$)** | **891,151** | **89.12%** |
| **2nd Nearest Centroid ($c_2$)** | **86,476** | **8.65%** |
| **3rd+ Nearest Centroid ($> 2$)** | **22,373** | **2.24%** |

---

## 4. Generated Plots
- **Histogram of $r$:** [`reports/plots/r_histogram.png`](file:///C:/Users/viral/OneDrive/Desktop/Hiya/IITG/VectorDB/vector-db-research/reports/plots/r_histogram.png)
- **Empirical CDF Curve:** [`reports/plots/r_empirical_cdf.png`](file:///C:/Users/viral/OneDrive/Desktop/Hiya/IITG/VectorDB/vector-db-research/reports/plots/r_empirical_cdf.png)
