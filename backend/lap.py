"""Pure Python LAPJV-compatible shim for Ultralytics ByteTrack.

The upstream `lap` package ships a native DLL that may be blocked by Windows
application control. This small assignment adapter keeps the same thresholded
linear assignment behavior without loading that binary.
"""
from __future__ import annotations
import numpy as np

__version__ = "0.5.14-shim"

def lapjv(cost_matrix, extend_cost=False, cost_limit=float("inf"), return_cost=True):
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be two dimensional")
    rows, cols = cost.shape
    x = np.full(rows, -1, dtype=int)
    y = np.full(cols, -1, dtype=int)
    if rows == 0 or cols == 0:
        return (0.0, x, y) if return_cost else (x, y)

    # Each detection row can use a private unmatched column at the threshold.
    # This rectangular assignment leaves over-threshold matches unassigned.
    limit = float(cost_limit)
    if not np.isfinite(limit):
        limit = float(np.max(cost[np.isfinite(cost)], initial=0.0) + 1e6)
    penalty = max(1.0, abs(limit) * 1e-9)
    augmented = np.full((rows, cols + rows), limit + penalty, dtype=float)
    augmented[:, :cols] = np.where(np.isfinite(cost), cost, limit + penalty)
    augmented[:, cols:] = limit

    # Hungarian potentials algorithm; rows <= columns by construction.
    n, m = augmented.shape
    u = np.zeros(n + 1); v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int); way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(m + 1, np.inf); used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta = np.inf; j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = augmented[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(m + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    for j in range(1, m + 1):
        i = p[j]
        if i and j <= cols and cost[i - 1, j - 1] <= limit:
            x[i - 1] = j - 1; y[j - 1] = i - 1
    total = float(sum(cost[i, j] for i, j in enumerate(x) if j >= 0))
    return (total, x, y) if return_cost else (x, y)
