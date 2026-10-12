# R2 Dynamic Overlap — Parts 6–8

Inspection date: 2026-10-12. Branch: `r2-6-8-dynamic-overlap`.
Inspected HEAD: `3d61441b87d82b778e312b68cf2fbf83e16feacf`.
Status: Parts 6–8 implemented and measured on SIFT1M; Part 8 is range-sensitive.
Earlier milestone pending statements below are historical; current status is in the final milestone.

## 1. Research Objective

Derive an overlap threshold delta from dataset geometry and a requested storage
budget, then evaluate recall/storage/scan tradeoffs and automatic knee selection.
The initial budgets are 5%, 10%, 20%, and 30%.

## 2. Starting Point

R5 is the fixed foundation: capacity-aware squared-L2 clustering, authoritative
labels, capacity-safe leaf cleanup, and existing routing/search. Its implementation
is in `vecstudy/core/capacity_clustering.py`, `vecstudy/core/r5_leaves.py`, and
`vecstudy/index/segment.py`. Preserve `R5_BASELINE.md`.

Teammate commit `3d61441` adds Parts 2/3 boundary analysis and Part 5 verification,
plus SIFT evaluation/report changes. Only `tests/test_boundary.py` was added to
the test suite by that commit; existing tests were not changed.

Part 4 is handled separately by the teammate. No cross-dataset analysis, GIST
runner, or 768-dimensional embedding experiment was found. Generic fvecs/ivecs
readers, dimension-agnostic boundary functions, and synthetic data are reusable
infrastructure. DESIGN mentions GIST; that is not evidence of executed Part 4.
Part 4 does not block initial SIFT/synthetic Parts 6–8 work.

## 3. Mathematical Background

Research definitions:

```
r = d2 / d1 - 1
F_R(delta) = count(r <= delta) / N
bloat = total_members / N - 1
```

If every selected vector receives exactly one additional membership in a distinct
posting, bloat equals F_R(delta), apart from numerical representation. Filtering,
owner collisions, or RNG pruning can break this identity.

Verified Part 2 implementation: `compute_boundary_statistics` in
`vecstudy/core/boundary.py` returns row-aligned float64 d1/d2/r and int64 c1/c2.
L2 uses sqrt(max(||x||² + ||c||² - 2*x.c, 0)); IP uses max(1-x.c, 0).
Dot products are computed in float32 and then promoted, while L2 norms are
float64. This can introduce cancellation near zero. Chunking defaults to 32,768.
Top two are selected with argpartition and swapped only for strict distance
inequality; ties have no explicit lowest-centroid-ID rule.
When d1=0 and d2=0, r=0; when d1=0<d2, r=+inf.
There is no explicit finite-input validation. NaNs can propagate; negative
computed distances are clipped. Negative finite r should not occur with valid
ordered nonnegative distances.

`scripts/analyze_boundary.py` uses normalized SIFT1M base rows and final
`seg.levels[-1].centroids`, reconstructing owners from `Posting.owned`.
Vector IDs are implicit base-row indices, not returned by the core API or saved
with r. Neither script persists raw r/c1/c2 arrays. Cache provenance and complete
owner coverage are assumed rather than validated.

Part 3 summary uses finite-only mean, median, population std, min/max, and
NumPy's default interpolated percentiles p1/p5/p10/p25/p50/p75/p90/p95/p99.
Threshold counts use all N rows as denominator. Empty finite summaries become
zero. The plotting script uses a 100-bin density histogram clipped at p99.5,
and sorted finite r with ranks 1..Nfinite divided by Nfinite. Thus its plotted
CDF and threshold CDF have different denominators when nonfinite r exists.
The line plot is not an exact step representation at ties.

## 4. Part 6 — Budget-Based Dynamic Delta

Implemented API: `choose_delta(r_values, target_bloat)` in
`vecstudy/core/boundary.py`, next to the shared statistic/CDF implementation.
The shared Part-2 distance calculation is reused unchanged.

For valid nonempty finite nonnegative r and 0<b<=1:

```
k = ceil(N*b)
delta = sorted_r[k-1]
```

Use an observed order statistic without interpolation. With inclusive selection,
ties may yield F_R(delta)>b; even without ties ceil rounds up by less than 1/N.
Report requested budget, selected fraction, actual replicated fraction, and
overshoot. Do not randomly break ties or describe b as a strict storage cap.

Implemented edge contract:

- Reject empty/non-1D inputs and invalid/nonfinite/out-of-range budgets.
- b=0 returns an `None` no-overlap sentinel. Delta=0 can select
  tied boundary vectors and therefore is not equivalent to no overlap.
- b=1 selects max(r) for finite data.
- Reject negative r, NaN, and -inf rather than silently changing the denominator.
- Treat upstream +inf as a documented zero-distance case: retain it in N, report
  its count, and raise ValueError when the order statistic is infinite. Never silently drop it.
- d1=d2=0 inherits r=0 and belongs to the lowest tied group.

Using +inf as a threshold needs an explicit policy: `r<=inf` would select all
rows, but `d2<=d1*(1+inf)` yields NaN for d1=0 and does not have the same behavior.

Do not invert the existing rounded report values or interpolated finite-only
summary percentiles. Recompute aligned statistics from a verified snapshot.
Part 6 tests and measured calibration results are recorded below.

## 5. Part 7 — Recall vs Bloat Experiments

Planned comparison: no overlap; fixed delta 0.10 and 0.20; budget-derived delta
for 5%, 10%, 20%, and 30%. Use one immutable R5 segment snapshot and separate
membership variants rebuilt from owned-only lists for every treatment. Preserve
owners, centroids, hierarchy, vector IDs, queries, query settings, and tombstones.
Do not accumulate replicas across treatments or rerun clustering.

Reuse `search`, `Counters`, `recall_at_k`, dataset readers/normalization,
`make_result_row`, and Parquet persistence. `scripts/test_sift1m.py` demonstrates
cache reuse and npost evaluation, but `run_queries` floors scan averages with
integer division. Record true aggregate vectors_scanned/query_count instead.
This counts member occurrences, including repeated replicas across fetched
postings, rather than unique scored vectors.

The generic `runner.run_experiment` rebuilds through maintainers for each cell;
it cannot directly provide a same-snapshot R5 overlap comparison. R5 IndexConfig
currently rejects non-None delta and RNG. Part 6 now supplies a separate post-build replica layer; Part 7 still needs
a same-snapshot query evaluator, leaving the R5 build contract intact.

Numerical result table: pending execution. Required columns: mode, target_bloat,
delta, selected_fraction, replicated_fraction, actual_bloat, recall_at_10,
vectors_scanned_per_query, query_count, and snapshot/configuration identifiers.
No Parts 6–8 plots have been generated.

## 6. Part 8 — Automatic Knee Detection

Reuse Part 5's configurable delta-list loop and plotting pattern, plus the future
same-snapshot Part 7 evaluator and existing Parquet result helpers. The existing
depth YAML sweep is configuration infrastructure, not a recall/bloat R2 sweep.
Part 5 records only delta/fr/replicas/bloat/diff in memory and Markdown; it runs
no queries and provides no recall or scan measurements.

Proposed format: one Parquet row per treatment with the Part 7 fields and full
configuration; optional CSV export for inspection. Required curve points are
(delta, actual_bloat, recall_at_10, vectors_scanned_per_query).

Missing: joint sweep evaluator, recall/bloat plots, handling repeated bloat
values, marginal-gain calculation, knee algorithm and validation. For distinct
increasing bloat, marginal gain is difference in recall divided by difference
in bloat. Equal-bloat points need explicit consolidation; noisy/nonmonotone recall,
sweep range/density, flat curves, and endpoint selection need documented policy.
Knee selection is not implemented or claimed to succeed.

## 7. R5 Integration

R5 authoritative ownership must remain unchanged. Overlap may add replicas only.
Dynamic overlap must not rerun geometric ownership assignment.

`Segment.build`'s R5 branch calls `from_primary_labels` and returns before legacy
`assign`. The teammate's boundary script reads owners but does not replace them.
However, `scripts/verify_delta_bloat.py` calls legacy `assign` on final R5
centroids. That function constructs new IP-nearest owners and second-IP-nearest
replicas, rather than preserving authoritative R5 owners. It leaves the cached
R5 segment intact, but verifies a different temporary assignment model.

Legacy assign gives at most one extra copy, with destination distinct from its
recomputed geometric owner. Its c2 may equal the authoritative R5 owner.
Blindly adding both geometric c1 and c2 to an authoritative third-or-later owner
can produce three memberships. Adding c2 alone can produce no new membership
when owner=c2. Neither is a valid guaranteed-one-copy layer.

Implemented Part 6 policy: select using geometric L2 r; use c2 if owner=c1,
otherwise use c1. This always chooses one distinct geometric candidate, retaining
the original owner and making replication fraction equal selected fraction when
RNG is disabled. This differs from unconditional c2 replication and preserves the one-extra-copy identity.

R5 clustering is squared-L2; Part 2 default r is Euclidean L2. Nearest rankings
agree mathematically, but squared-distance ratios differ:
`r_squared=(1+r_euclidean)^2-1`. Part 5 uses 1-IP instead. Final arithmetic-mean
centroids are not normalized, so centroid norm differences also change IP/L2
rankings. Routing and vector scoring remain IP. Unit-normalized vector-to-vector
IP ranking agrees with L2 ranking, but supplied SIFT ground truth is raw-L2 and
normalization can change it. Keep search unchanged and label evaluation geometry.

## 8. Experimental Setup

Every future run must record dataset/checksum, base/query counts and query IDs,
dimension, normalization and ground-truth source, complete R5 configuration,
snapshot/centroid hash, npost, k, beam/depth, target budget, derived/fixed delta,
mode, boundary geometry, replica policy, RNG status, seed, BLAS/thread settings,
code SHA, and timestamp. Use a global vector-weighted CDF initially; per-segment
CDFs would define different thresholds and need separate experimental treatment.

Existing R2 scripts default to normalized SIFT1M: 1,000,000 base vectors, 128-D,
1,250 cached final leaf centroids. Boundary fallback R5 configuration is depth=1,
branching_factor=16, seed=0, target_capacity=800, lambda_balance=0.25, alpha=1.5,
tiny_cluster_fraction=0.10, max_iter=15. Part 5 explicitly disables RNG.
Both scripts trust the cache; configuration, dataset, and centroid hashes are
not checked. No local `data/sift_r5_segment.pkl` was found during inspection.

## 9. Results

### Prior teammate results (not Parts 6–8)

Sources: `reports/BOUNDARY_ANALYSIS.md`, `reports/DELTA_VS_BLOAT.md`.
Part 2/3 Euclidean r: mean 0.0742; median 0.0482; std 0.0844; min 0.000000;
max 1.3133. Percentiles p1/p5/p10/p25/p50/p75/p90/p95/p99:
0.0007/0.0036/0.0075/0.0203/0.0482/0.0974/0.1691/0.2310/0.4122.
Owner concordance: c1=891,151; c2=86,476; other=22,373.

| Delta | L2 count (Part 3 prediction) | IP replicas (Part 5) | Part 5 measured bloat |
| --- | ---: | ---: | ---: |
| 0.01 | 130,857 | 144,679 | 14.468% |
| 0.02 | 246,593 | 268,263 | 26.826% |
| 0.05 | 512,924 | 537,843 | 53.784% |
| 0.10 | 758,663 | 776,735 | 77.673% |
| 0.15 | 873,925 | 885,770 | 88.577% |
| 0.20 | 929,971 | 937,303 | 93.730% |
| 0.25 | 958,844 | not measured | not measured |
| 0.30 | 974,375 | not measured | not measured |

Part 5 reports CDF/bloat differences as 0.0000 percentage points at all six
thresholds. These are displayed precision results, not a new rerun or proof of
bitwise equality. Its identity is verified under IP geometric ownership with RNG
off, not under authoritative R5 replication. Plots exist at
`reports/plots/r_histogram.png`, `r_empirical_cdf.png`, and
`delta_vs_bloat_verification.png`; reports contain stale machine-local links.
No R2 JSON/CSV/Parquet or persisted boundary arrays were found.

### R5 baseline evidence (separate runs)

`R5_BASELINE.md` reports 1,000 queries, build 455.1s, 1,250 leaves, mean owned
size 800, p99 955, max 1,067, zero bloat. At npost=1/2/4/8/16 it reports
Recall@10=0.3007/0.4338/0.5943/0.7421/0.8637 and integer scan costs
816/1,634/3,269/6,538/13,054.

Local ignored `results/r5_sift_validation/validation.json` and `results.parquet`
record 10,000 queries and a different R5 run: build 130.96708145807497s,
max size 1,066, zero bloat. Rounded recall and exact stored scan averages:

| npost | Recall@10 (rounded) | vectors scanned/query |
| --- | ---: | ---: |
| 1 | 0.31531 | 821.7432 |
| 2 | 0.45813 | 1643.7215 |
| 4 | 0.61334 | 3287.5606 |
| 8 | 0.75820 | 6568.875 |
| 16 | 0.87010 | 13108.765 |

Local diagnostics: IP/L2 nearest disagreement 505,495/1,000,000;
owner/IP disagreement 533,757/1,000,000. Across 64 sampled queries normalized-IP
versus supplied top-10 overlap is 0.9890625; 6 query top-10 sets differ.
These measurements must not be conflated with teammate results.
Additional local synthetic Parquet evidence: baseline recall 0.457,
scan/query 829.17; R5 recall 0.402, scan/query 683.72; both zero bloat.

### Parts 6–8

Part 6 calibration results are recorded in the milestone below. Parts 7 and 8
have no implementation or experimental results yet.

## 10. Tests and Validation

Full suite on inspection: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest
-q -p no:cacheprovider`: **142 passed, 9 warnings, 2.45s**. Warnings are existing
`datetime.utcnow()` deprecations in results.py. No test/code changes were made.

Teammate's boundary tests comprise 9 collected cases: equidistant r=0; exact
hand-computed L2 ratio; zero d1 giving inf; chunk equivalence; ordered distinct
top two and nonnegative finite r; both geometries (two cases); argument errors;
summary finite count/interpolated median/threshold monotonicity and counts.
Basic Parts 2/3 behavior is covered, but Part 5's CDF equality has no dedicated
unit test. Existing postings tests cover coverage, unique ownership, owned subset,
two-membership bound, bloat, and build counters under legacy assignment.
Existing R5 integration tests preserve non-nearest authoritative labels and
global IDs. Diagnostics tests cover lowest-ID ties in diagnostics, geometry
disagreement, ownership validation, batches, tombstones, and aggregation; that
tie guarantee does not extend to boundary.py.

Missing future dependencies: empirical order-statistic/tie/budget edge tests;
NaN/inf/negative policy; duplicate centroids and both-zero distances; boundary
counter/empty behavior; adversarial float32 cancellation; R5-owner=c2 and
owner-outside-top-two replication; immutable owners/centroids/hierarchy; global
ID alignment; no accumulated replicas; equality of selected/replicated/bloat
fractions; same-snapshot evaluation and correct scan denominator; repeated-bloat,
flat/noisy curve and knee validation. No new reproducibility run was executed.

## 11. Design Decisions

Confirmed constraint: preserve R5 and existing routing/search; inspect only.
Part 6 decisions: Euclidean r against final leaf snapshot; observed
inclusive quantile; transparent tie overshoot; no-overlap sentinel for zero
budget; explicit +inf policy; one distinct destination chosen using owner-aware
top-two policy; global CDF with all base-vector denominator; RNG off for primary
budget experiments; same-snapshot treatments and full-precision scan averages.

## 12. Limitations / Open Questions

Part 6 adopts the requested Euclidean metric and owner-aware destinations.
Remaining teammate coordination concerns final-centroid provenance, deterministic tie policy,
zero/nonfinite semantics, aligned IDs, owner-aware destination policy, global
versus segment CDF, bloat denominator, and shared Part 4 implementation.
Part 3 L2 quantiles cannot calibrate Part 5 IP replication as currently written.
Legacy assignment cannot be layered on R5 ownership. Input/clipping precision,
cache provenance, and ground-truth mismatch require explicit reporting.
Part 4 is pending but does not block initial experiments. No cached SIFT segment
is present locally; later full-data work must establish the intended snapshot
and dataset availability rather than assuming reported artifacts reproduce it.

## 13. Final Conclusions

Incomplete until Parts 6–8 are implemented, validated, and measured. Eventually
answer whether requested bloat is achieved (including ties), how dynamic and
fixed thresholds compare, the recall/storage tradeoff, whether a useful delta
can be selected automatically, and whether the knee is a sensible operating point.
Update this document after each meaningful milestone with exact measured results,
configuration/provenance, and scientifically relevant failed or negative runs.


## Part 6 Implementation Milestone — 2026-10-12

### Implementation and rationale

`choose_delta` is a pure function in `vecstudy/core/boundary.py`. It uses
`np.partition` on a nonmutating copy and ceil(N*b), returning an observed value
rather than an interpolated threshold. All ties are selected, exposing overshoot.
It rejects empty/non-1D r, negative r, NaN/-inf, and nonfinite/out-of-range budgets.
Positive infinity remains in N; a required infinite threshold raises ValueError.
b=0 returns None after input validation; b=1 returns the maximum when finite.
Both-zero distances retain upstream r=0. No finite-only recalibration occurs.

`apply_budget_overlap` in `vecstudy/index/overlap.py` takes a snapshot, explicit
aligned global IDs, existing Part-2 r/c1/c2, and budget. It validates exact unique
owner coverage, final-leaf/posting alignment, valid distinct candidate IDs, and
absence of tombstones. It returns a new Segment and all requested metrics.
It starts from owned-only lists on every application, replacing any preexisting
replicas so treatments cannot accumulate. It copies posting and hierarchy arrays;
only the vector corpus is shared, as in existing Segment behavior.

Selected vectors go to c2 when owner=c1, otherwise c1. Destinations are always
non-owner and exactly one additional membership is added. Legacy assign is not
called: its geometric reassignment would replace capacity-aware R5 labels.
Euclidean-L2 r is intentionally paired with squared-L2 R5 clustering; centroid
ranking agrees, while ratio units differ. Routing/search remain existing IP.
RNG is absent because filtering would break budget calibration equality.

Metrics: target_bloat, derived_delta, selected_count/fraction,
replicated_count/replication_fraction, realized_bloat=M/N-1, and signed overshoot
(realized_bloat-target_bloat). Floating subtraction can differ from replica/N by
machine precision; validation uses tolerance. No recall experiment is run.

### Tests

Added `tests/test_dynamic_overlap.py`: 23 collected cases covering all four
budgets; observed quantiles and ceil rounding; zero/one budgets; invalid inputs;
zero-distance/+inf policy; ties/overshoot; repeated calls; permutation/mutation;
monotone delta; all three owner cases; destination distinctness; one-extra-copy
bound; unselected vectors; ownership/centroid/hierarchy preservation; no overlap;
repeat-application duplicate prevention; independent output arrays; invalid
snapshot/alignment rejection; and an actual built R5 snapshot integration case.

Full suite: **165 passed, 9 existing datetime.utcnow deprecation warnings, 2.49s**
after implementation; final rerun after experiment/runner updates: **165 passed,
9 warnings, 2.30s**.
Command: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`.
The 142-test inspection result above is historical. `git diff --check` passed.

### Calibration experiment

Runner: `scripts/validate_dynamic_overlap.py`. Command:
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_dynamic_overlap.py`.
It reads local SIFT1M and builds a fresh snapshot using exact validated R5
parameters, then reuses existing Part-2 Euclidean boundary statistics. It checks
actual membership multiplicity equals one plus the selected mask for every
vector, unchanged authoritative owners/centroids/hierarchy, duplicate absence,
and bloat/replication equality for each budget. No queries are executed.

Part 4 remains pending teammate work. Part 7 Recall-vs-Bloat and Part 8 knee
selection remain unimplemented and uncompleted. Part 6 does not establish their
recall benefit or operating-point quality.

### Measured SIFT1M Part 6 results

Fresh build: 129.88969479105435s; boundary computation: 3.5035829169210047s.
Base shape: [1000000, 128]; final leaves: 1250; queries executed: 0.
Finite r count: 1000000; +inf count: 0.
Normalization: L2-normalized float32; Euclidean boundary distances against final
R5 leaf centroids; RNG disabled; seed=0; single-thread BLAS; NumPy 2.4.6.

Exact R5 configuration:

```json
{
  "dim": 128,
  "leaf_size": 800,
  "branching_factor": 16,
  "depth": 1,
  "delta": null,
  "rng_rule": false,
  "seed": 0,
  "build_mode": "r5",
  "target_capacity": 800,
  "lambda_balance": 0.25,
  "alpha": 1.5,
  "tiny_cluster_fraction": 0.1,
  "max_iter": 15
}
```

| Target | Derived delta | Selected count | Selected % | Replicated count | Replication % | Actual bloat (fraction, exact stored float) | Signed overshoot |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 5% | 0.003652633660341742 | 50,000 | 5% | 50,000 | 5% | 0.050000000000000044 | 4.163336342344337e-17 |
| 10% | 0.007516240289318743 | 100,000 | 10% | 100,000 | 10% | 0.10000000000000009 | 8.326672684688674e-17 |
| 20% | 0.015807504067695266 | 200,000 | 20% | 200,000 | 20% | 0.19999999999999996 | -5.551115123125783e-17 |
| 30% | 0.025079955016746203 | 300,000 | 30% | 300,000 | 30% | 0.30000000000000004 | 5.551115123125783e-17 |

All four selected counts exactly equal N*b; there is no count-based overshoot
at these observed thresholds. The displayed signed overshoots are floating-point
subtraction artifacts, not extra replicas. Synthetic tied tests demonstrate that
other populations can exceed the requested calibration budget.

Machine-readable artifact: `results/r2_part6/validation.json` (ignored by git).
Dataset checksum: `63a20f112cbf534415e0c71e278cc2ac1078067b489abadd5c393321a4a54d4f`.

Final-centroid SHA256: `1264c6510d5637331fd00199b793ebb56753ff83906e4f255320983192364e6a`.

Ownership-array SHA256: `09250d1fa39a97d9d1c9fb3b657997a2aaa4ae699ab143fa81b551d16c803d9e`.

Build provenance: fresh local R5 build, no cached snapshot; base git SHA `3d61441b87d82b778e312b68cf2fbf83e16feacf`
with uncommitted Part 6 changes. This is a new run under the frozen methodology,
not a claim of bitwise identity with the teammate's cached run.

Part 6 is complete for the requested SIFT1M budget calibration and ready for
Part 7 implementation. Limitations: boundary.py retains its upstream centroid-tie
policy and mixed float32/float64 arithmetic; aligned statistics must originate
from the supplied final snapshot (the overlay validates IDs but cannot prove
statistic provenance); tombstoned populations are deliberately rejected; infinite
threshold budgets are rejected; query recall and knee quality remain unmeasured.

## Part 7 Implementation Milestone — 2026-10-12

### Research question and controlled design

Does budget-derived overlap improve Recall@10, and what storage and scan cost
accompanies that gain compared with fixed delta? The scientific variable is the
threshold-selection mode. Fixed delta 0.10/0.20 and dynamic 5/10/20/30% all use
identical Euclidean-L2 r against the same final R5 centroids and the same
owner-preserving replica destination rule. Historical teammate Part-5 IP results
are retained above but are not controlled Part-7 baseline measurements.

`vecstudy/index/overlap.py` now exposes `apply_fixed_overlap`; the existing
`apply_budget_overlap` derives delta with unchanged choose_delta semantics and
calls that same implementation. None means no overlap. Every variant starts
from authoritative owned-only lists; no legacy assign call or RNG is involved.

`vecstudy/harness/overlap_experiment.py` computes Part-2 L2 statistics once and
runs all seven independent treatments. Each has the same snapshot fingerprint,
queries, QueryConfig, supplied ground truth, and unchanged search/recall
functions. The evaluator sums integer vectors_scanned and divides by the full
query count without flooring. Occurrences of replicas scored in multiple fetched
postings count multiple times, consistent with repository counters.

Validation checks original owned-only membership, unchanged ownership/centroids/
hierarchy, unchanged snapshot fingerprint before and after queries, exact per-ID
membership multiplicity, non-owner destination rule, duplicate absence, and
selected_count=replicated_count. Dynamic threshold and actual bloat monotonicity
are checked. Result rows include both signed target difference and count-based
tie overshoot separately from floating-point subtraction artifacts.

### Configuration and provenance

Runner: `scripts/compare_dynamic_overlap.py`.
Command: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/compare_dynamic_overlap.py`.
One fresh R5 build, with the frozen configuration:

- SIFT1M: all 1,000,000 base vectors, 128 dimensions, all 10,000 queries in file
  order; supplied 10,000 x 100 ground truth, using its first 10 neighbors.
- Base and queries L2-normalized float32; supplied ground truth unchanged.
- build_mode=r5; seed=0; target_capacity=800; alpha=1.5; lambda_balance=0.25;
  max_iter=15; tiny_cluster_fraction=0.10; depth=1; branching_factor=16;
  delta=None; rng_rule=False. Default leaf_size=800.
- QueryConfig: k=10; n_postings=8; beam_widths=(8,) (inactive at depth=1);
  overfetch=0.2; global_beam=False.
- BLAS thread variables pinned to 1 before NumPy import.
- Fingerprint, dataset/query/ground-truth checksums, full configs, timestamps,
  NumPy version, base git SHA, and executed source hashes recorded in JSON.

### Tests and validation

Added `tests/test_overlap_experiment.py`: seven collected cases covering a
one-boundary-computation seven-treatment run, immutable same-snapshot treatment
fingerprints, dynamic calls/quantiles, unchanged baseline search/recall,
repeat-run determinism, full-precision scan averages (explicit 2.5 toy average),
JSON/CSV round trips, invalid fixed thresholds, and controlled L2/IP disagreement
with legacy assignment forbidden. Existing Part-6 tests continue to validate
all three owner cases and one-extra-copy invariants.

Full suite after implementation: **172 passed, 9 existing datetime.utcnow
warnings, 2.30s**. No R5 clustering or query/search code changed.
Optional dependency: `pip install -e '.[research]'` installs matplotlib for the
standalone recall-vs-bloat plot. Numeric outputs are saved before plotting.

### Scope and limitations

RNG is NOT used in Part 7. Dynamic delta plus SPANN RNG will be evaluated
separately after this clean threshold experiment. Part 8 knee detection remains
pending, as does Part 4 cross-dataset teammate integration. Per-cluster thresholds
and update-time recomputation are outside this step.

Geometry remains capacity-aware squared-L2 R5 ownership, Euclidean-L2 boundary
analysis, IP routing/scoring, and supplied raw-L2 SIFT ground truth. This existing
metric distinction is reported, not corrected. Results are one dataset, one seed,
one npost setting; the seven points do not establish a knee or generalize to
unmeasured datasets. Full-data deterministic reruns are not required for this
milestone; small-snapshot determinism is tested and full-data provenance retained.

### Measured Part 7 results

Fresh R5 build time: 128.40627204207703s; one build and one boundary
computation. All seven rows share snapshot SHA256
`f851bbe8d8f438a2b57d0cf40c21ef347e1ffbb86fb25c32695463986d5cba57`.

| Treatment | Delta | Target bloat | Selected / replicated count | Selected / replication % | Actual bloat (stored fraction) | Recall@10 (stored value) | Total scanned | Exact scans/query |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| R5 no overlap | — | 0% | 0 | 0% | 0.0 | 0.7581999999999878 | 65,688,750 | 6568.875 |
| Fixed delta 0.10 | 0.1 | — | 758,716 | 75.8716% | 0.758716 | 0.8728099999999744 | 119,957,845 | 11995.7845 |
| Fixed delta 0.20 | 0.2 | — | 929,980 | 92.998% | 0.92998 | 0.8849799999999716 | 133,449,061 | 13344.9061 |
| Dynamic 5% | 0.003652633660341742 | 5% | 50,000 | 5% | 0.050000000000000044 | 0.7684999999999873 | 69,270,489 | 6927.0489 |
| Dynamic 10% | 0.007516240289318743 | 10% | 100,000 | 10% | 0.10000000000000009 | 0.7786199999999865 | 72,875,436 | 7287.5436 |
| Dynamic 20% | 0.015807504067695266 | 20% | 200,000 | 20% | 0.19999999999999996 | 0.7975299999999858 | 80,011,560 | 8001.156 |
| Dynamic 30% | 0.025079955016746203 | 30% | 300,000 | 30% | 0.30000000000000004 | 0.8141599999999848 | 87,080,260 | 8708.026 |

Every treatment: 10,000 queries, npost=8, k=10, 1,250 leaves, 1,000,000
authoritative owned vectors. No ownership, centroid, hierarchy, or membership
invariant failed. Baseline search exactly reproduces the local R5 validation
Recall@10 and scan/query recorded above.

| Dynamic target | Signed actual-minus-target difference | Count-based tie overshoot |
| --- | --- | --- |
| 5% | 4.163336342344337e-17 | 0.0 |
| 10% | 8.326672684688674e-17 | 0.0 |
| 20% | -5.551115123125783e-17 | 0.0 |
| 30% | 5.551115123125783e-17 | 0.0 |

Derived deltas exactly match Part 6 at all four budgets. There is no count-based
overshoot; tiny signed differences are floating-point artifacts.

Numeric artifacts: `results/r2_part7/results.json` and
`results/r2_part7/results.csv`. Plot: `results/r2_part7/recall_vs_bloat.png`.
All artifacts are in the existing git-ignored results directory. The script saved
numeric results before a first plotting attempt failed because matplotlib was
still being installed; this did not invalidate or rerun any query measurements.
The standalone plot was regenerated from saved numeric results afterward.

### Interpretation and negative results

- Fixed delta 0.10: measured recall gain 11.461000 percentage points; scan increase 5426.909500 occurrences/query (82.615509%).
- Fixed delta 0.20: measured recall gain 12.678000 percentage points; scan increase 6776.031100 occurrences/query (103.153601%).
- Dynamic 5%: measured recall gain 1.030000 percentage points; scan increase 358.173900 occurrences/query (5.452591%).
- Dynamic 10%: measured recall gain 2.042000 percentage points; scan increase 718.668600 occurrences/query (10.940513%).
- Dynamic 20%: measured recall gain 3.933000 percentage points; scan increase 1432.281000 occurrences/query (21.804053%).
- Dynamic 30%: measured recall gain 5.596000 percentage points; scan increase 2139.151000 occurrences/query (32.564952%).

All overlap treatments increased recall and scans in this run; no negative recall
result occurred. Dynamic budgets provide lower-storage operating points, with
recall rising from 0.75820 to 0.81416 at 30% bloat. Fixed thresholds achieve
higher recall (0.87281 and 0.88498), but consume 75.8716% and 92.998% bloat,
respectively. Thus this experiment shows controlled storage calibration, not
that dynamic overlap dominates fixed thresholds at matched bloat. Because both
use the same r and destination policy, a fixed threshold equal to a derived
delta would produce the same membership treatment.

Fixed L2 boundary counts (758,716 at .10; 929,980 at .20) differ slightly from
the teammate's historical report (758,663 and 929,971). Preserve both as separate
runs: source centroids/precision/build environment differ or are not fully
provenanced historically. No parameters were adjusted to force agreement.

Part 7 measurements are complete and ready for Part 8 to consume; no knee
algorithm or knee claim is implemented. RNG remains disabled and its later
comparison separate. Part 4 remains pending teammate work.

## Part 8 Implementation Milestone — 2026-10-12

### Research question and sweep design

Without a user-provided budget, can measured Recall-vs-Bloat select a useful
threshold automatically? Part 7's seven irregular points are too sparse for
robust knee interpretation. Part 8 systematically evaluates budgets
0%, 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%, 60%, 70%, 80%, 90%.
No 100% treatment is needed for the observed operational range.

Runner: `scripts/detect_overlap_knee.py`, reusing the Part-7 SIFT runner and
`run_overlap_comparison` with a configurable treatment list. One fresh R5 build,
one existing Part-2 Euclidean boundary calculation, independent owned-only
replica variants, and all 10,000 identical queries per point. Nonzero thresholds
come from choose_delta, never manually constructed. The complete validated
R5 and QueryConfig from Part 7 are unchanged. RNG remains disabled.

Every treatment retains the Part-7 checks of ownership, centroids, hierarchy,
snapshot fingerprint, non-owner destinations, per-vector multiplicity, duplicate
absence, no accumulated replicas, and exact scan average. Part-7 fixed references
are included only after verifying identical snapshot fingerprint, data/query/GT
checksums, and complete build/query configurations.

### Exact deterministic geometric algorithm

Implementation: `vecstudy/harness/knee.py`, version
`normalized-beneficial-chord-v1`.

1. Deep-copy points and reject nonfinite bloat/recall. Sort by actual bloat.
2. Exact duplicate x values collapse to highest measured recall, then lowest
   delta, then original input order. No averaging or smoothing. Marginals are
   computed only on distinct x; duplicate x never produces division by zero.
3. Normalize x and y separately by observed min/max ranges.
4. For endpoint chord vector (vx,vy), compute signed perpendicular distance:
   `(vx*(y_norm-y_first) - vy*(x_norm-x_first))/hypot(vx,vy)`.
   Positive is above the chord, the beneficial side for increasing saturation.
5. Select an interior point with maximum positive normalized deviation. Points
   within `1e-10` absolute normalized-distance tolerance of that maximum tie;
   choose the lower-bloat point. This tolerance is numerical, not a marginal
   benefit threshold.
6. Return explicit knee=None for fewer than three distinct bloat points
   (including constant x), constant recall, nonpositive net endpoint improvement,
   or maximum beneficial deviation <=1e-10 (including a straight line).

Both axes are normalized so unit choices cannot determine the knee. Nonmonotone
recall points are retained and flagged; negative gains are preserved. The
geometric candidate is not proof of useful curvature: neighboring marginals and
range sensitivity are reported without changing its selection rule.

### Marginal gain and sensitivity definitions

Consecutive distinct measured points use delta_bloat=x_i-x_(i-1),
delta_recall=y_i-y_(i-1), marginal_gain=delta_recall/delta_bloat. Units are recall
fraction gained per bloat fraction (numerically equivalent to percentage points
recall per percentage point bloat). Multiply by 10 to express recall percentage
points per 10 percentage points bloat. The first point has null interval metrics.

The same detector is applied independently to 0–90% and 0–50% subsets, each
renormalized over its own range. Fixed .10/.20 references do not enter either
calculation. No post-result tuning of budgets, tolerance, or algorithm is used.

### Tests

Added `tests/test_knee.py`: 17 collected cases covering an interpretable elbow,
linear and constant curves, too few/empty points, duplicate bloat, sorting,
equal-distance lower-bloat ties, near-zero deviation tolerance, determinism,
input immutability, scale invariance, nonfinite x/y rejection, repository row
schema/JSON compatibility, operational-range filtering, sweep definition, and
retention of nonmonotone marginal gains. Prior harness tests still cover the
shared one-snapshot evaluator. Full suite: **189 passed, 9 existing datetime.utcnow
warnings, 2.46s**. Final post-experiment rerun: **189 passed, 9 warnings,
2.34s**; git diff --check passed.

### Scope

RNG is still disabled. SPANN RNG + dynamic delta is a separate later ablation;
existing legacy RNG code is untouched, and no new RNG support is introduced.
Part 4 cross-dataset analysis remains pending teammate integration. Any knee
established here is specific to SIFT1M, this seed/configuration, npost=8, and
these measured ranges. Build/ownership is squared L2; boundary is Euclidean L2;
search is IP; supplied ground truth is raw L2. No geometry or R5 correction is
made during Part 8.

### Full measured SIFT1M sweep

Fresh build time: 128.9285457499791s. All rows share
snapshot SHA256 `f851bbe8d8f438a2b57d0cf40c21ef347e1ffbb86fb25c32695463986d5cba57` and match the Part-7
reference fingerprint/configuration/data/query/GT checksums. Every point used
10,000 queries, k=10, npost=8, 1,250 leaves, and 1,000,000 owners.

| Target | Derived delta | Selected / replicated count | Actual bloat (stored fraction) | Recall@10 (stored) | Total scanned | Exact scans/query | Marginal gain from previous |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| 0% | None | 0 | 0.0 | 0.7581999999999878 | 65,688,750 | 6568.875 | — |
| 5% | 0.003652633660341742 | 50,000 | 0.050000000000000044 | 0.7684999999999873 | 69,270,489 | 6927.0489 | 0.20599999999999044 |
| 10% | 0.007516240289318743 | 100,000 | 0.10000000000000009 | 0.7786199999999865 | 72,875,436 | 7287.5436 | 0.20239999999998465 |
| 15% | 0.011569394635263741 | 150,000 | 0.1499999999999999 | 0.7883199999999864 | 76,443,588 | 7644.3588 | 0.1939999999999971 |
| 20% | 0.015807504067695266 | 200,000 | 0.19999999999999996 | 0.7975299999999858 | 80,011,560 | 8001.156 | 0.18419999999998754 |
| 25% | 0.020306467255899907 | 250,000 | 0.25 | 0.8060499999999852 | 83,541,770 | 8354.177 | 0.1703999999999882 |
| 30% | 0.025079955016746203 | 300,000 | 0.30000000000000004 | 0.8141599999999848 | 87,080,260 | 8708.026 | 0.1621999999999922 |
| 35% | 0.030154370469570813 | 350,000 | 0.3500000000000001 | 0.8219999999999831 | 90,601,593 | 9060.1593 | 0.15679999999996572 |
| 40% | 0.035657437448846085 | 400,000 | 0.3999999999999999 | 0.8295699999999804 | 94,131,736 | 9413.1736 | 0.15139999999994655 |
| 45% | 0.04163085684582124 | 450,000 | 0.44999999999999996 | 0.8370199999999804 | 97,674,752 | 9767.4752 | 0.14900000000000121 |
| 50% | 0.04822424045667972 | 500,000 | 0.5 | 0.8439699999999791 | 101,202,639 | 10120.2639 | 0.13899999999997348 |
| 60% | 0.06364421744309867 | 600,000 | 0.6000000000000001 | 0.8569699999999769 | 108,359,125 | 10835.9125 | 0.1299999999999778 |
| 70% | 0.08409502140700487 | 700,000 | 0.7 | 0.8675399999999754 | 115,622,774 | 11562.2774 | 0.10569999999998539 |
| 80% | 0.11411601785932612 | 800,000 | 0.8 | 0.8766199999999736 | 123,063,759 | 12306.3759 | 0.09079999999998192 |
| 90% | 0.1691230166774178 | 900,000 | 0.8999999999999999 | 0.8836099999999723 | 130,941,965 | 13094.1965 | 0.06989999999998618 |

Every selected/replicated fraction equals its target; no count-based tie
overshoot. Tiny actual-bloat residuals reflect floating subtraction. Recall,
scan averages, and thresholds increase throughout this measured sweep. All
snapshot and membership invariants passed. No measured negative recall gains.

### Automatic knee and sensitivity results

| Range | Selected index (sorted distinct, zero-based) | Budget | Delta | Actual bloat | Recall@10 | Scans/query | Normalized distance | Marginal before | Marginal after |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0–90% | 9 | 45% | 0.04163085684582124 | 0.44999999999999996 | 0.8370199999999804 | 9767.4752 | 0.09086217828580598 | 0.14900000000000121 | 0.13899999999997348 |
| 0–50% | 4 | 20% | 0.015807504067695266 | 0.19999999999999996 | 0.7975299999999858 | 8001.156 | 0.04140247470118012 | 0.18419999999998754 | 0.1703999999999882 |

Full-range selected knee: recall gain **7.882000 percentage points** over
R5 baseline; scan increase **3198.600200 occurrences/query (48.693272%)**.
Neighbor intervals 40→45% and 45→50% yield 1.490000 and
1.390000 recall percentage points per 10 percentage points
bloat. Marginals continue falling to approximately 0.0699 recall fraction per
bloat fraction over 80→90%. This supports gradual diminishing returns.

Sensitivity changes the candidate from 45% to 20% (25 percentage points),
with delta changing from 0.04163085684582124 to 0.015807504067695266. The
normalized geometric rule yields valid range-conditional candidates, but this
is a smoothly concave curve without a demonstrated sharp, range-independent
transition. Do not claim a universal uniquely optimal delta. Normalization removes
unit dependence, not dependence on sweep endpoints or sampling density.

### Fixed-delta references and interpretation

Fixed .10: 75.8716% bloat, recall 0.8728099999999744, scans/query 11995.7845.
Fixed .20: 92.998% bloat, recall 0.8849799999999716, scans/query 13344.9061.
The full-range knee uses 30.8716 and 47.998 percentage points less bloat,
respectively, but has lower recall by 3.579 and 4.796 percentage points. It also
scans fewer members. Fixed references are historical Part-7 measurements of the
identical snapshot/configuration, not new Part-8 treatments, and are excluded
from primary and sensitivity knee computation.

The contribution is automatic dataset-dependent threshold selection under
a budget or an explicitly specified measured sweep range. It does not show
that dynamic thresholds always outperform fixed delta. Marginal gains validate
the interpretation; they never override the geometric selection. Small normalized
deviation is not a statistical significance test; measurement uncertainty, query
resampling, other seeds, datasets, sweep densities, and additional ranges remain
unmeasured. All 15 treatments ran against the same full query workload.

### Artifacts and final validation

Machine-readable: `results/r2_part8/sweep_results.json`, `sweep_results.csv`,
and `knee_analysis.json`. Analysis retains all collapsed points, interval gains,
normalized coordinates/distances, candidate index/metrics, both ranges, algorithm
version and tolerance. Sweep JSON contains shared build/source provenance and
full row metrics. Artifacts remain in the existing git-ignored results directory.

Plots: `results/r2_part8/recall_vs_bloat_knee.png`,
`normalized_chord_diagnostic.png`, and `marginal_gain.png`. The first two were
visually inspected; all three exist with nonzero file sizes. JSON/CSV row counts
and numeric round trips, single snapshot, 10,000 queries, exact scan division,
replica counts, and disabled RNG were independently checked after execution.

Part 8 is scientifically complete for the specified SIFT1M experiment, with
range sensitivity explicitly retained as a research outcome. RNG is disabled;
no new RNG implementation, Part 4, per-cluster delta, update recomputation, R5
change, routing/scoring change, or Recall@10 change was introduced.

### Updated conclusions for Parts 6–8

Budget-derived thresholds achieved all requested SIFT1M counts without tie
overshoot; tied synthetic populations correctly expose overshoot. Overlap
improved measured recall with additional storage and scanning. Fixed .10/.20
provided higher recall at much larger bloat. A deterministic knee can select a
range-conditional threshold automatically, but 45% versus 20% sensitivity shows
that SIFT1M does not establish a unique range-independent optimum. These are
measured SIFT1M conclusions, not cross-dataset validation. Part 4 and RNG ablation
remain pending. No commit or push was performed.

## SPANN RNG Representative-Replication Ablation

### Motivation and inspected legacy implementation

This is a separate ablation after Parts 6–8, not a revision of those experiments.
Hold delta and boundary candidates fixed within each pair, then measure whether
representative filtering removes redundant storage/scans while preserving recall.
Operating points are 10% (low storage), 20% (restricted-range knee), and 45%
(full-range knee). No dense sweep or post-filter budget recalibration.

Inspected `vecstudy/index/postings.py:assign`: it recomputes geometric owners from
IP scores, selects second-IP-nearest replicas with relative 1-IP distances, and
rejects when `1-dot(c1,c2) < d2`. That strict inequality agrees algebraically with
a candidate-distance > representative-distance skip rule, and equality passes.
However, its 1-IP geometry and recomputed owners do not satisfy this ablation's
Euclidean-L2 and authoritative-R5 contract. It is left unchanged and not called.

### Exact one-replica adaptation

The tested reference rule supplied for this study skips a candidate c_ij when
Dist(c_ij,x) > Dist(c_i(j-1),c_ij). In this one-candidate overlay, the authoritative
owner centroid is the already accepted representative; the proposed c2 (owner=c1)
or c1 (otherwise) is the candidate. Reject iff:

```
||x - candidate_centroid||_2 > ||owner_centroid - candidate_centroid||_2
```

Equality is accepted; no tolerance or random tie-breaking is introduced.
`rng_keep_candidates` uses chunked explicit float64 Euclidean norms of stored
vectors/centroids, validates finite geometry and representative IDs, and rejects
nonfinite computed distances. Distinct centroid IDs may have identical locations:
a positive vector-to-candidate distance then fails, and zero distance passes.
Only selected vectors can pass. This helper makes no ownership decisions.

The existing fixed/budget overlay defaults to rng=False, preserving Parts 6–8.
When enabled, it filters the existing selected mask before constructing members;
selected_count still measures candidates, whereas replicated_count measures
accepted replicas. Owner lists, centroids, hierarchy and search remain unchanged.
No legacy assignment, per-cluster threshold, or update recomputation is added.

### One-snapshot experiment and metrics

`scripts/ablate_dynamic_rng.py` calls the shared SIFT runner with six treatments.
R5 builds once; Part-2 L2 boundary statistics compute once. Each pair independently
rederives delta using choose_delta, producing identical candidate sets; every
overlay starts from the same owned-only snapshot. The same 10,000 queries and
supplied raw-L2 GT run through unchanged IP search/Recall@10 at k=10, npost=8.
All R5 fields are the exact validated Part-7/8 configuration: seed 0, target 800,
alpha 1.5, lambda 0.25, max_iter 15, tiny fraction .10, depth 1, branching 16,
build delta=None and build rng_rule=False. Base/query normalization is unchanged.

Candidate counts/fractions and acceptance/rejection counts/fractions are distinct.
Acceptance fractions divide by candidate count (zero candidates -> 0). Realized
bloat is members/owners-1, verified equal accepted/owners within floating precision.
Paired effects are without-minus-with for bloat and scans, with-minus-without for
recall; relative reductions divide by the corresponding no-RNG value. No-RNG
results must exactly reproduce Part 8's stored values and snapshot/data/query/GT/
configuration provenance before effects are interpreted. Baseline is the verified
Part-8 reference, not an additional query treatment.

Validation rechecks ownership/centroids/hierarchy, original snapshot fingerprint,
unique memberships, correct destination, one-extra-copy multiplicity, RNG subset
of boundary candidates, accepted+rejected=candidate count, exact scan division,
and independent treatments. Machine-readable outputs preserve configurations,
source hashes, snapshot identity, six rows, paired effects and baseline reference.

### Tests and scope

Added `tests/test_rng_overlap.py`: nine collected cases covering pass/fail,
equality, zero distances, unselected vectors, all three owner cases, subset-only
pruning, unchanged ownership/centroids/hierarchy, duplicate prevention, metrics,
repeat application, malformed/nonfinite inputs, six-treatment shared harness,
paired effect checks, and heterogeneous JSON/CSV serialization. Existing tests
continue to verify no-RNG behavior and budget semantics unchanged.
Full suite before full-data run: **198 passed, 9 existing datetime.utcnow
warnings, 2.21s**.

This adaptation is not SPANN's full multi-representative closure expansion: it
has one proposed replica and one authoritative, possibly non-nearest accepted
owner. Therefore results concern this documented adaptation, not all SPANN RNG
variants. Geometry remains squared-L2 R5 ownership / Euclidean-L2 boundary and
RNG / IP search / supplied raw-L2 GT. Part 4 remains pending teammate integration.
Part 6's SIFT calibration and Part 8's range-sensitive conclusion are preserved.

### Measured SIFT1M RNG ablation results

Fresh R5 build: 129.36981666705105s; one build and one L2
boundary computation. Shared snapshot SHA256:
`f851bbe8d8f438a2b57d0cf40c21ef347e1ffbb86fb25c32695463986d5cba57`. Six treatments, all 10,000 queries
each. Exact no-RNG reproduction of delta, counts, bloat, recall, total scans and
scan average against Part-8 JSON passed for all three operating points.

| Treatment | Delta | Candidates | Accepted replicas | Rejected | Acceptance % | Rejection % | Actual bloat (stored fraction) | Recall@10 (stored) | Total scanned | Exact scans/query |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Dynamic 10% | 0.007516240289318743 | 100,000 | 100,000 | — | — | — | 0.10000000000000009 | 0.7786199999999865 | 72,875,436 | 7287.5436 |
| Dynamic 10% + RNG | 0.007516240289318743 | 100,000 | 4,860 | 95,140 | 4.86 | 95.14 | 0.0048600000000000865 | 0.7604399999999872 | 66,149,540 | 6614.954 |
| Dynamic 20% | 0.015807504067695266 | 200,000 | 200,000 | — | — | — | 0.19999999999999996 | 0.7975299999999858 | 80,011,560 | 8001.156 |
| Dynamic 20% + RNG | 0.015807504067695266 | 200,000 | 9,618 | 190,382 | 4.809 | 95.191 | 0.009617999999999904 | 0.7624699999999867 | 66,607,010 | 6660.701 |
| Dynamic 45% | 0.04163085684582124 | 450,000 | 450,000 | — | — | — | 0.44999999999999996 | 0.8370199999999804 | 97,674,752 | 9767.4752 |
| Dynamic 45% + RNG | 0.04163085684582124 | 450,000 | 20,115 | 429,885 | 4.47 | 95.53 | 0.020115000000000105 | 0.7662499999999852 | 67,670,898 | 6767.0898 |

Selected fractions remain 0.10/0.20/0.45 in both halves of each pair. After
RNG, replication fractions are exactly 0.00486/0.009618/0.020115; actual-bloat
subtraction differs only by machine precision. Candidate sets and deltas match
within each pair, accepted+rejected=candidates, and no accumulation/invariant
failure occurred. No budget was recalibrated.

### Paired effects

| Budget | Bloat reduction (fraction) | Relative bloat reduction | Recall difference (fraction) | Scan reduction/query | Relative scan reduction |
| --- | --- | --- | --- | --- | --- |
| 10% | 0.09514 | 0.9513999999999991 | -0.018179999999999308 | 672.5896000000002 | 0.09229304645257976 |
| 20% | 0.19038200000000005 | 0.9519100000000005 | -0.03505999999999909 | 1340.455 | 0.16753266652968646 |
| 45% | 0.42988499999999985 | 0.9552999999999998 | -0.07076999999999523 | 3000.385400000001 | 0.30718126624984937 |

### Scientific interpretation and negative result

- 10% pair: RNG recall gain over baseline is 0.224000 percentage points, retaining 10.969638% of the no-RNG recall gain.
- 20% pair: RNG recall gain over baseline is 0.427000 percentage points, retaining 10.856852% of the no-RNG recall gain.
- 45% pair: RNG recall gain over baseline is 0.805000 percentage points, retaining 10.213144% of the no-RNG recall gain.

The filter rejects approximately 95% of candidates. It reduces storage and
scan cost substantially but loses approximately 89–90% of the original recall
improvement. This is an aggressive-pruning negative result for the stated goal
of preserving most of dynamic overlap's recall benefit. Accepted replicas retain
some utility, but the measured recall losses show the rejected set cannot be
described as purely redundant. No condition was tuned after observing results.

A future matched-realized-bloat comparison could distinguish efficiency at
similar storage from the present same-delta pruning effect; it was not implemented
or inferred here. The authoritative R5 owner may lie outside geometric top two,
and a single accepted representative is a restricted adaptation of SPANN's
multi-representative closure rule. This result does not establish behavior for
full SPANN, other datasets, seeds, query budgets, or geometries. Existing metric
mismatch limitations remain. Part 6 calibration is robust on the tested SIFT
configuration; Part 8 remains range-sensitive; Part 4 integration is pending.

### Artifacts and completion

`results/r2_rng/rng_ablation.json` contains full configuration/provenance, six
rows, paired effects, and the verified baseline reference.
`results/r2_rng/rng_ablation.csv` contains six rows including all candidate/RNG/
replication/search metrics. `results/r2_rng/rng_comparison.png` shows baseline
and connected pairs; visually inspected. These artifacts remain git-ignored.
JSON/CSV numeric round trips, counts, snapshot identity, exact scan averages,
pair equality, bloat inequality, and plot existence were independently checked.

The separate RNG ablation is complete. Parts 6–8 numeric artifacts/results are
preserved; default overlap remains without RNG. R5 build, parameters, ownership,
centroids, routing/search geometry, query scoring, and recall calculation are
unchanged. No commit or push.

Final post-experiment full suite: **198 passed, 9 existing warnings, 2.39s**.
`git diff --check` passed; git status inspected on `r2-6-8-dynamic-overlap`.
All work remains unstaged/untracked as reported, including earlier Parts 6–8.

## Parameterized RNG / Relaxed RNG Ablation

Follow-up experiment, separate from Parts 6–8 and the original RNG ablation.
The original negative result above is unchanged. The shared RNG helper now keeps
selected candidates iff `d_x_candidate <= rng_alpha * d_owner_candidate`, with
explicit float64 Euclidean norms, equality accepted, and finite rng_alpha>=1.
Default 1.0 preserves the original condition exactly. This relaxation is a
parameterized extension of the one-replica adaptation, not original SPANN RNG.
R5's capacity parameter is unchanged and recorded as r5_balance_alpha=1.5 in
these new artifacts, distinctly from rng_alpha.

Controlled setup: only dynamic 20%, one fresh validated R5 build, one existing
Part-2 L2 boundary computation, and one choose_delta calibration. Identical
candidate mask and destinations are reused for every independent owned-only
overlay. All 10,000 SIFT queries, k=10, npost=8, and the exact previous build/query
configuration and normalization remain unchanged. Same-snapshot/data/query/GT/
configuration provenance is required; no-RNG and alpha=1 measurements must exactly
reproduce the original RNG artifacts before interpretation. Smaller-alpha accepted
sets must be subsets of larger-alpha sets. Recall monotonicity is observed only,
not enforced. No dynamic budget recalibration, 10%/45% budget, or large delta sweep.

Predefined rng_alpha values: 1, 1.25, 1.5, 2, 3, 5, plus no RNG. Retention is
`(recall_alpha - baseline_recall)/(recall_no_rng - baseline_recall)`, without
clipping. The primary successful best measured finite alpha requires retention
>=0.95 and minimizes actual bloat, then scans/query, then rng_alpha. No-RNG cannot
qualify as a finite alpha. If no predefined alpha qualifies, highest retention is
reported separately as a fallback. Secondary Pareto analysis maximizes recall
and minimizes bloat with at least one strict inequality for domination.

Allowed refinement is fixed in advance: first adjacent finite pair crossing
95%, or one extension to 10 if alpha=5 fails; if 10 fails, stop. Inside a valid
bracket, exactly up to four deterministic midpoint measurements update the local
bracket. Final selection uses only actually measured qualifying values; this is
a local practical search, not a global optimum or a monotonic-recall assumption.
The equation/objective are not tuned after observing measurements.

Added `tests/test_relaxed_rng.py`: eight cases covering alpha=1 identity, invalid
alphas, equality, nested accepted memberships, monotone counts/bloat, unchanged
owners/destinations/centroids/hierarchy, deterministic repeat overlays, fixed
calibration reuse, rng_alpha serialization, best-alpha tie ordering, Pareto
selection and unclipped negative retention. Existing tests remain intact.
Full suite before full-data run: **206 passed, 9 existing warnings, 2.19s**.
An initial new toy-metric fixture lacked total_scanned; the fixture was completed
and the full suite rerun. No experiment was run with failing tests.

### Measured results and bounded refinement

Fresh build time: 128.5170533329947s. All 11 treatments share
snapshot `f851bbe8d8f438a2b57d0cf40c21ef347e1ffbb86fb25c32695463986d5cba57`, delta
`0.015807504067695266`, and 200,000 identical candidates/destinations.
Endpoint reproduction and nested accepted sets/counts/bloat passed. Original
Parts 6–8 and RNG artifacts were checked byte-identical during this run.

| rng_alpha | Stage | Accepted | Rejected | Rejection % | Actual bloat (stored fraction) | Recall@10 (stored) | Exact scans/query | Gain retention % |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| no RNG | endpoint | 200,000 | 0 | 0.000000 | 0.19999999999999996 | 0.7975299999999858 | 8001.156 | 100.000000000 |
| 1.0 | predefined | 9,618 | 190,382 | 95.191000 | 0.009617999999999904 | 0.7624699999999867 | 6660.701 | 10.856852276 |
| 1.25 | predefined | 75,007 | 124,993 | 62.496500 | 0.07500700000000005 | 0.7803699999999871 | 7146.4131 | 56.369183829 |
| 1.5 | predefined | 155,884 | 44,116 | 22.058000 | 0.1558839999999999 | 0.7938299999999853 | 7694.0648 | 90.592423087 |
| 2.0 | predefined | 198,889 | 1,111 | 0.555500 | 0.1988890000000001 | 0.7974999999999858 | 7990.2538 | 99.923722349 |
| 3.0 | predefined | 199,999 | 1 | 0.000500 | 0.19999900000000004 | 0.7975299999999858 | 8001.132 | 100.000000000 |
| 5.0 | predefined | 200,000 | 0 | 0.000000 | 0.19999999999999996 | 0.7975299999999858 | 8001.156 | 100.000000000 |
| 1.75 | refinement | 191,319 | 8,681 | 4.340500 | 0.19131900000000002 | 0.7970499999999856 | 7933.3659 | 98.779557590 |
| 1.625 | refinement | 179,229 | 20,771 | 10.385500 | 0.17922900000000008 | 0.7960599999999858 | 7850.153 | 96.262395118 |
| 1.5625 | refinement | 169,131 | 30,869 | 15.434500 | 0.16913099999999992 | 0.7952199999999854 | 7782.079 | 94.126620900 |
| 1.59375 | refinement | 174,566 | 25,434 | 12.717000 | 0.174566 | 0.7956799999999858 | 7818.3466 | 95.296211543 |

Predefined qualifiers: rng_alpha=2, 3, 5. Best predefined alpha=2 by the
storage criterion. Valid initial bracket [1.5,2]; four midpoint measurements in
order: 1.75 (qualifies), 1.625 (qualifies), 1.5625 (fails), 1.59375 (qualifies).
Final bracket [1.5625,1.59375]. No extension to 10 and no further search.

Best measured alpha under the predefined 95%-retention criterion: **1.59375**.
Accepted 174,566; rejected 25,434;
acceptance 87.283%, rejection 12.717%.
Actual bloat 17.4566%; bloat reduction
2.543400 percentage points (12.717000% relative).
Recall 0.7956799999999858; loss vs no RNG 0.0018499999999999073
(0.185000 percentage points); gain over baseline
3.748000 percentage points; retained gain
95.296211543%. Scans/query 7818.3466;
saving 182.8094000000001/query (2.284787348%).

Measured nondominated finite alphas (sorted): 1.0, 1.25, 1.5, 1.5625, 1.59375, 1.625, 1.75, 2.0, 3.0.
No RNG and alpha=5 are dominated by alpha=3, which matches their recall with
one fewer replica. Equal measurements are not strict domination. Pareto analysis
does not change the primary retention-based selection.

### Interpretation and limits

Acceptance and bloat rise strongly between alpha=1 and 1.5, then approach full
replication by alpha=2. Recall recovers monotonically in these measured finite
points, though this was never enforced. Alpha=1 retains only 10.856852% of the
gain; the selected relaxed point retains 95.296212%. A finite useful point exists,
but satisfying this strict target still needs 87.283% of the original replicas.
Storage savings are moderate (12.717% of replica overhead), and scan savings are
small (2.284787%); this does not demonstrate a large operational improvement.
The relaxed rule helps recall retention substantially compared with alpha=1,
without changing the earlier negative conclusion about the original rule.

This is the best actually measured alpha under the predefined objective, not a
global optimum. One dataset/configuration/budget/query workload, four-step local
refinement, no uncertainty estimates, and the existing squared-L2 / Euclidean-L2 /
IP / raw-L2 GT distinction limit generalization. Duplicate-location centroids
with positive vector-to-candidate distance remain rejected for every finite
alpha, so approaching no RNG is not guaranteed for arbitrary geometry.
No unrelated experiments or R5/search changes; Part 4 remains pending.

### Artifacts and validation

Runner: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/ablate_relaxed_rng.py`.
Outputs: `results/r2_relaxed_rng/relaxed_rng_results.json`,
`relaxed_rng_results.csv`, `best_alpha_analysis.json`, `recall_vs_bloat.png`,
and `recall_retention_vs_bloat_saved.png`. Plots are regenerated from saved
JSON; label positions were adjusted for legibility without changing numerical
results or selection. Executed source hashes and a separate plot-source hash
are recorded. CSV has 11 measured rows; stages identify predefined/refinement/
endpoint treatments. Numeric round trips, provenance, candidate/destination
identity, calibration count, alpha naming and plot existence passed validation.

Final post-experiment suite: **206 passed, 9 existing warnings, 2.31s**.
`git diff --check` passed; git status inspected. All work remains unstaged/
untracked on `r2-6-8-dynamic-overlap`. No commit or push.


### Relaxed RNG Across Dynamic-Overlap Budgets

Research question: does the best measured relaxed-RNG alpha depend on the dynamic overlap budget? Studied 10%, 20%, 30%, and 45% on normalized SIFT1M with the existing R5 configuration and unchanged k=10, npost=8 query configuration. The completed 20% experiment was reused without any new queries. Baseline and no-RNG references were reused from Part 8 after configuration, dataset/query/ground-truth checksum, and snapshot fingerprint validation. Prior result artifacts remained byte-identical.

One fresh R5 build (129.361 seconds) and one boundary-statistics computation served all three new budgets. Stage A completed all 27 cheap mask evaluations before any ANN queries: alpha 1, 1.25, 1.5, 1.6, 1.75, 2, 2.5, 3, and 5 at each new budget. Candidate counts were exactly 100,000, 300,000, and 450,000; accepted masks were nested, counts/bloat monotonic, and ownership unchanged. Stage B measured 1.5 and 1.75, then three midpoints (1.625, 1.5625, 1.59375) in the observed failing/qualifying bracket. Total new expensive evaluations: **15 full 10,000-query evaluations**, five per new budget and zero at 20%. No new baseline or no-RNG query evaluations. Hard guards limit each new budget to six evaluations and the total to eighteen.

The unchanged criterion requires at least 95% of the original overlap recall gain above the R5 baseline (Recall@10 0.75820). Among measured qualifying points, minimize actual bloat, then scans/query, then alpha. The best measured alpha under this criterion is **1.59375 at all four budgets**. Cheap-only Stage-A alphas are excluded from query-result selection.

| Budget | Delta | Best Alpha | Bloat Before | Bloat After | Bloat Reduction % | Recall Before | Recall After | Gain Retained % | Scans Before | Scans After | Scan Reduction % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 0.007516240289318743 | 1.59375 | 10.0000% | 8.7634% | 12.366 | 0.77862 | 0.77766 | 95.2987 | 7287.5436 | 7197.8060 | 1.231 |
| 20% | 0.015807504067695266 | 1.59375 | 20.0000% | 17.4566% | 12.717 | 0.79753 | 0.79568 | 95.2962 | 8001.1560 | 7818.3466 | 2.285 |
| 30% | 0.025079955016746203 | 1.59375 | 30.0000% | 26.0813% | 13.062 | 0.81416 | 0.81151 | 95.2645 | 8708.0260 | 8428.0289 | 3.215 |
| 45% | 0.04163085684582124 | 1.59375 | 45.0000% | 38.8965% | 13.563 | 0.83702 | 0.83317 | 95.1155 | 9767.4752 | 9334.1469 | 4.436 |

Bloat and scan reductions in the table are relative to the corresponding no-RNG budget; they are percentages, not percentage-point differences. Additional selected-point quantities:

| Budget | Candidate replicas | Accepted | Rejected | Absolute bloat reduction (percentage points) | Absolute recall loss | Scans saved/query |
|---|---|---|---|---|---|---|
| 10% | 100,000 | 87,634 | 12,366 | 1.2366 | 0.00096 | 89.7376 |
| 20% | 200,000 | 174,566 | 25,434 | 2.5434 | 0.00185 | 182.8094 |
| 30% | 300,000 | 260,813 | 39,187 | 3.9187 | 0.00265 | 279.9971 |
| 45% | 450,000 | 388,965 | 61,035 | 6.1035 | 0.00385 | 433.3283 |

Zero-new-query common-alpha analysis: **1.59375** is the smallest already-measured alpha qualifying across all four budgets. Its per-budget retention and bloat reductions are exactly the selected-point values above. There is no observed systematic change in best measured alpha with budget at this resolution. Relative replica savings rise mildly (12.366% to 13.563%), while absolute and relative scan savings increase with overlap. Maintaining the requirement at larger budgets needs more absolute replicas, but does not require a larger measured alpha here. Relaxed RNG provides modest scan savings, smallest at low overlap and larger at high overlap; these counters do not establish a latency improvement.

Negative result: 1.5 and 1.5625 fail the 95% gain-retention criterion at every budget. The selected values retain only about 95.1–95.3% of overlap gain, so stronger pruning at those lower measured alphas is not acceptable under the predefined criterion. No recall monotonicity was assumed or enforced; selection uses measured qualifying points.

Limitations: SIFT1M only, one R5 seed/configuration, one query configuration and npost, three midpoint refinements per new budget, no global optimality claim, confidence interval, or latency measurement. Smaller unmeasured alpha values may qualify within the final brackets. Matching best measured values supports a common parameter for this setup; it does not establish transfer to other datasets or routing configurations.

Artifacts: `results/r2_relaxed_rng_multibudget/` contains `multibudget_results.json`, `multibudget_results.csv` (25 measured alpha points: 15 new and 10 reused), `stage_a_acceptance.csv` (27 cheap-only points), `best_alpha_by_budget.json`, and the three requested plots. R5 construction and search implementations are unchanged. No commit or push.
