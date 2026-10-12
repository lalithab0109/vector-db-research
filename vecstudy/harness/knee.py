"""Transparent, normalized beneficial chord-distance knee analysis."""
from __future__ import annotations
import copy
import numpy as np

ALGORITHM = 'normalized-beneficial-chord-v1'
TOLERANCE = 1e-10  # normalized perpendicular distance, also tie tolerance
BUDGETS = tuple(i / 100 for i in range(0, 51, 5)) + (.6, .7, .8, .9)


def analyze_knee(rows, *, tolerance=TOLERANCE):
    """Return measured candidate or explicit no-knee; do not mutate inputs.

    Sort by actual bloat. Equal x: highest recall, then lowest finite delta,
    then original input order. Marginals use distinct x, so never divide by zero.
    Signed distance is positive above the endpoint chord. Only positive interior
    deviations > tolerance count. Within tolerance of the maximum, prefer lower x.
    Nonmonotone measurements are retained and flagged, not smoothed or corrected.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    points = copy.deepcopy(list(rows))
    for p in points:
        if not np.isfinite(p['actual_bloat']) or not np.isfinite(p['recall_at_10']):
            raise ValueError('bloat and recall must be finite')
    points.sort(key=lambda p: (p['actual_bloat'], -p['recall_at_10'],
                              float('-inf') if p.get('delta') is None else p['delta']))
    unique = []
    for p in points:
        if not unique or p['actual_bloat'] != unique[-1]['actual_bloat']:
            unique.append(p)
    result = dict(algorithm=ALGORITHM, tolerance=tolerance,
                  duplicate_policy='highest recall, lowest delta, then input order',
                  input_count=len(points), unique_count=len(unique),
                  duplicates_collapsed=len(points)-len(unique), points=unique,
                  knee=None, selected_index=None, reason=None,
                  marginal_unit='recall fraction per bloat fraction')
    for i, p in enumerate(unique):
        dx = None if i == 0 else p['actual_bloat'] - unique[i-1]['actual_bloat']
        dy = None if i == 0 else p['recall_at_10'] - unique[i-1]['recall_at_10']
        p.update(delta_bloat=dx, delta_recall=dy,
                 marginal_gain=None if dx is None else dy / dx)
    if len(unique) < 3:
        result['reason'] = 'fewer than three distinct bloat points'
        return result
    x = np.array([p['actual_bloat'] for p in unique], dtype=float)
    y = np.array([p['recall_at_10'] for p in unique], dtype=float)
    result['recall_nondecreasing'] = bool(np.all(np.diff(y) >= 0))
    if np.ptp(y) == 0:
        result['reason'] = 'constant recall'
        return result
    xn, yn = (x-x.min())/np.ptp(x), (y-y.min())/np.ptp(y)
    vx, vy = xn[-1]-xn[0], yn[-1]-yn[0]
    distance = (vx*(yn-yn[0])-vy*(xn-xn[0])) / np.hypot(vx, vy)
    for p, a, b, d in zip(unique, xn, yn, distance):
        p.update(x_norm=float(a), y_norm=float(b), normalized_distance=float(d))
    if y[-1] <= y[0]:
        result['reason'] = 'no net endpoint recall improvement'
        return result
    maximum = float(np.max(distance[1:-1]))
    if maximum <= tolerance:
        result['reason'] = 'no beneficial deviation above numerical tolerance'
        return result
    eligible = np.flatnonzero((distance >= maximum-tolerance) & (distance > tolerance))
    eligible = eligible[(eligible > 0) & (eligible < len(unique)-1)]
    idx = int(eligible[0])
    result.update(selected_index=idx, reason='maximum beneficial normalized chord deviation',
                  maximum_normalized_distance=maximum,
                  knee=dict(unique[idx], marginal_before=unique[idx]['marginal_gain'],
                            marginal_after=unique[idx+1]['marginal_gain']))
    return result


def analyze_ranges(rows):
    # Fixed references never enter this API; caller passes only systematic sweep.
    return dict(full_range=analyze_knee(rows),
                operational_range=analyze_knee([r for r in rows if r['target_bloat'] <= .5]))


def plot_knee(rows, analysis, fixed_rows, out_dir):
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    full = analysis['full_range']
    points = full['points']
    knee = full['knee']
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([p['actual_bloat']*100 for p in points], [p['recall_at_10'] for p in points],
            'o-', label='Systematic dynamic sweep', markersize=4)
    for r in fixed_rows:
        ax.scatter(r['actual_bloat']*100, r['recall_at_10'], marker='^', s=60, color='tab:orange')
        ax.annotate(f"Fixed delta {r['delta']:.2f}", (r['actual_bloat']*100, r['recall_at_10']),
                    xytext=(-65, 10), textcoords='offset points', fontsize=8)
    if knee:
        ax.scatter(knee['actual_bloat']*100, knee['recall_at_10'], marker='*', s=180,
                   color='tab:red', label=f"Full-range knee ({knee['target_bloat']*100:g}%)", zorder=5)
    ax.set(xlabel='Actual bloat (%)', ylabel='Recall@10', title='SIFT1M: automatic chord-distance knee, RNG disabled')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out_dir/'recall_vs_bloat_knee.png', dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(axes, ('full_range','operational_range'), ('0–90% sweep','0–50% sensitivity')):
        a = analysis[key]; p = a['points']; k = a['knee']
        ax.plot([r['x_norm'] for r in p], [r['y_norm'] for r in p], 'o-', markersize=4, label='Normalized curve')
        ax.plot([p[0]['x_norm'],p[-1]['x_norm']], [p[0]['y_norm'],p[-1]['y_norm']], '--', color='gray', label='Endpoint chord')
        if k:
            x,y = k['x_norm'],k['y_norm']
            ax.scatter(x,y, marker='*',s=150,color='tab:red',label=f"Knee {k['target_bloat']*100:g}%")
            # These increasing experimental curves have normalized endpoints (0,0),(1,1).
            projection = (x+y)/2
            ax.plot([x,projection],[y,projection],':',color='tab:red',label='Perpendicular deviation')
        ax.set(xlabel='Normalized actual bloat', ylabel='Normalized recall',title=title)
        ax.set_aspect('equal'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir/'normalized_chord_diagnostic.png',dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot([p['actual_bloat']*100 for p in points[1:]], [p['marginal_gain'] for p in points[1:]], 'o-')
    ax.set(xlabel='Interval endpoint actual bloat (%)', ylabel='Recall fraction / bloat fraction',title='Measured consecutive marginal recall gains')
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out_dir/'marginal_gain.png',dpi=200); plt.close(fig)
