"""Escape routing on the height grid.

Every cell is a node with 8 neighbours. Moving from one cell to the next costs
walking time from Tobler's hiking function (uphill/downhill aware) multiplied
by a hazard penalty. One Dijkstra run from a virtual "safety" node on the
reversed graph gives every cell its time-to-safety and its next step, i.e. an
evacuation direction field for the whole area.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .hydrology import OFFSETS


def tobler_speed(slope: np.ndarray) -> np.ndarray:
    """Walking speed in m/s for a signed slope dz/dx (Tobler 1993)."""
    return 6.0 * np.exp(-3.5 * np.abs(slope + 0.05)) / 3.6


@dataclass
class RouteField:
    time_s: np.ndarray          # (h, w) seconds to reach safety, inf if trapped
    next_idx: np.ndarray        # (h, w) flat index of the next cell, -1 at targets/unreachable
    target_id: np.ndarray       # (h, w) id of the safe target each cell is routed to (-1 none)
    flow: np.ndarray            # (h, w) people passing through each cell
    shape: tuple[int, int]


def evacuation_field(z: np.ndarray | None, cell_m: float, multiplier: np.ndarray, passable: np.ndarray,
                     targets: np.ndarray, target_offset_s: np.ndarray | None = None,
                     population: np.ndarray | None = None) -> RouteField:
    """z: elevation in metres (None = flat). targets: int array, >0 = target id."""
    h, w = multiplier.shape
    n = h * w
    sink = n
    src_list, dst_list, cost_list = [], [], []
    idx = np.arange(n).reshape(h, w)
    for dr, dc in OFFSETS:
        r0, r1 = max(0, -dr), h - max(0, dr)
        c0, c1 = max(0, -dc), w - max(0, dc)
        a = idx[r0:r1, c0:c1]
        b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        ok = passable[r0:r1, c0:c1] & passable[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        dist = cell_m * np.hypot(dr, dc)
        if z is not None:
            slope = (z[r0 + dr:r1 + dr, c0 + dc:c1 + dc] - z[r0:r1, c0:c1]) / dist
            speed = tobler_speed(slope)
        else:
            speed = np.full(a.shape, tobler_speed(np.zeros(1))[0])
        mult = 0.5 * (multiplier[r0:r1, c0:c1] + multiplier[r0 + dr:r1 + dr, c0 + dc:c1 + dc])
        cost = dist / speed * mult
        src_list.append(a[ok]); dst_list.append(b[ok]); cost_list.append(cost[ok])

    tgt = np.flatnonzero((targets.ravel() > 0) & passable.ravel())
    offs = (target_offset_s.ravel()[tgt] if target_offset_s is not None else np.zeros(tgt.size)) + 1e-3
    src_list.append(tgt); dst_list.append(np.full(tgt.size, sink)); cost_list.append(offs)

    src = np.concatenate(src_list); dst = np.concatenate(dst_list); cost = np.concatenate(cost_list)
    # reversed graph: distances from the sink back to every cell
    graph = csr_matrix((cost, (dst, src)), shape=(n + 1, n + 1))
    dist, pred = dijkstra(graph, directed=True, indices=sink, return_predecessors=True)

    time_s = dist[:n].reshape(h, w)
    nxt = pred[:n].copy()
    nxt[(nxt == sink) | (nxt < 0)] = -1

    # assign each cell to its target, nearest-first
    tid = np.full(n, -1, np.int64)
    tflat = targets.ravel()
    order = np.argsort(dist[:n])
    nxt_l = nxt.tolist(); tid_l = tid.tolist(); tf_l = tflat.tolist(); pred_l = pred[:n].tolist()
    finite = np.isfinite(dist[:n]).tolist()
    for i in order.tolist():
        if not finite[i]:
            break
        if pred_l[i] == sink:
            tid_l[i] = tf_l[i]
        elif nxt_l[i] >= 0:
            tid_l[i] = tid_l[nxt_l[i]]
    tid = np.array(tid_l).reshape(h, w)

    flow = (population.ravel().astype(np.float64).copy() if population is not None else np.zeros(n))
    flow_l = flow.tolist()
    for i in order[::-1].tolist():
        if finite[i] and nxt_l[i] >= 0:
            flow_l[nxt_l[i]] += flow_l[i]
    flow = np.array(flow_l).reshape(h, w)
    return RouteField(time_s=time_s, next_idx=nxt.reshape(h, w), target_id=tid, flow=flow, shape=(h, w))


def trace(field: RouteField, row: int, col: int, max_steps: int = 100000) -> list[tuple[int, int]]:
    h, w = field.shape
    i = row * w + col
    path = [(row, col)]
    nxt = field.next_idx.ravel()
    for _ in range(max_steps):
        j = nxt[i]
        if j < 0:
            break
        i = int(j)
        path.append(divmod(i, w))
    return path


def arrows(field: RouteField, spacing: int, lookahead: int | None = None,
           skip: np.ndarray | None = None) -> list[list[float]]:
    """Sample direction arrows: [u0, v0, u1, v1, minutes] in normalised image coords."""
    h, w = field.shape
    look = lookahead or max(2, spacing // 2)
    out = []
    nxt = field.next_idx.ravel()
    for r in range(spacing // 2, h, spacing):
        for c in range(spacing // 2, w, spacing):
            t = field.time_s[r, c]
            if not np.isfinite(t) or nxt[r * w + c] < 0 or (skip is not None and skip[r, c]):
                continue
            i = r * w + c
            for _ in range(look):
                if nxt[i] < 0:
                    break
                i = int(nxt[i])
            r1, c1 = divmod(i, w)
            if (r1, c1) == (r, c):
                continue
            dv, du = r1 - r, c1 - c
            norm = np.hypot(du, dv)
            L = spacing * 0.42
            out.append([(c + 0.5) / w, (r + 0.5) / h,
                        (c + 0.5 + du / norm * L) / w, (r + 0.5 + dv / norm * L) / h,
                        round(float(t) / 60, 1)])
    return out
