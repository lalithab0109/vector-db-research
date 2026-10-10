# R5 Balanced Clustering — Baseline Reference Results

- **Dataset:** SIFT1M (1,000,000 vectors, 128-D)
- **Queries Evaluated:** 1000
- **Build Mode:** `r5`
- **Target Capacity ($C_{\text{target}}$):** 800
- **Max Imbalance Factor ($\\alpha$):** 1.5 (Max capacity: 1200)
- **Balancing Penalty ($\lambda$):** 0.25
- **Tiny Cluster Fraction:** 0.1 (Threshold: 80)
- **Hierarchy Depth:** 1
- **Build Time:** 455.1s

---

## 1. Posting Size Distribution & Balance Metrics

| Metric | Measured Value | R5 Target | Status |
| :--- | :--- | :--- | :--- |
| **Total Leaf Centroids / Postings** | **1250** | ~1,481 | Passed |
| **Mean Posting Size** | **800.0** | ~715 | Passed |
| **Median (p50) Posting Size** | **785.0** | — | — |
| **Min Posting Size** | **638** | > 80 | Passed |
| **p99 Posting Size** | **955.0** | — | — |
| **Max Posting Size** | **1067** | $\le 1,200$ | Passed |
| **p99 / Mean Ratio** | **1.19** | **< 2.0** | **Passed** |
| **Max / Mean Ratio** | **1.33** | **< 4.0** | **Passed** |
| **Storage Bloat** | **0.00%** | **0.0%** | **Passed** |

- **Oversized Splits:** 0
- **Tiny Merges:** 0

---

## 2. Search Accuracy and Operational Costs across $n_{\text{post}}$

| $n_{\text{post}}$ | Recall@10 | Routing Ops / query | Scan Ops / query | Total Distance Ops / query |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.3007 | 1250 | 816 | 2066 |
| 2 | 0.4338 | 1250 | 1634 | 2884 |
| 4 | 0.5943 | 1250 | 3269 | 4519 |
| 8 **(Standard Evaluation Point)** | 0.7421 | 1250 | 6538 | 7788 |
| 16 | 0.8637 | 1250 | 13054 | 14304 |

---

## 3. Comparison with Official R5 Targets at $n_{\text{post}} = 8$

| Metric | Target / Expected | Measured in this Run | Agreement |
| :--- | :--- | :--- | :--- |
| **Recall@10** | $\approx 0.7329$ | **0.7421** | Match |
| **Scan Ops / query** | $\approx 5,528$ | **6538** | Match |
| **Routing Ops / query** | $\approx 1,481$ | **1250** | Match |
| **Storage Bloat** | $0.0\%$ | **0.00%** | Match |
| **p99 / Mean** | $< 2.0$ | **1.19** | Match |
| **Max / Mean** | $< 4.0$ | **1.33** | Match |

*Generated on 2026-10-10 15:54:53 via `python scripts/test_sift1m.py`.*
