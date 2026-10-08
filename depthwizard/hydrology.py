"""Terrain hydrology: depression filling, D8 flow, accumulation and HAND.

HAND (Height Above Nearest Drainage, Nobre et al. 2011) is the vertical
distance from each cell down to the stream cell it drains into. A flood that
raises water h metres above stream level reaches every cell with HAND < h,
which makes a fast, explainable flood scenario.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def priority_flood(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Fill depressions so every cell drains to the edge (Barnes et al. 2014, +epsilon)."""
    h, w = z.shape
    flat = z.astype(np.float64).ravel().copy()
    closed = np.zeros(h * w, bool)
    heap: list[tuple[float, int]] = []
    border = np.zeros((h, w), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    for i in np.flatnonzero(border):
        heap.append((flat[i], int(i)))
        closed[i] = True
    heapq.heapify(heap)
    nbr = [dr * w + dc for dr, dc in OFFSETS]
    while heap:
        e, i = heapq.heappop(heap)
        r, c = divmod(i, w)
        for k, (dr, dc) in enumerate(OFFSETS):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                j = i + nbr[k]
                if not closed[j]:
                    closed[j] = True
                    if flat[j] <= e:
                        flat[j] = e + eps
                    heapq.heappush(heap, (flat[j], j))
    return flat.reshape(h, w)


def d8_receivers(filled: np.ndarray, cell: float) -> np.ndarray:
    """Index of the steepest-descent neighbour of every cell (-1 = outlet)."""
    h, w = filled.shape
    pad = np.pad(filled, 1, constant_values=np.inf)
    best = np.zeros((h, w))
    rec = np.full((h, w), -1, np.int64)
    rows, cols = np.mgrid[0:h, 0:w]
    for dr, dc in OFFSETS:
        nb = pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        slope = (filled - nb) / (cell * np.hypot(dr, dc))
        better = slope > best
        best[better] = slope[better]
        rec[better] = ((rows + dr) * w + (cols + dc))[better]
    return rec.ravel()


@dataclass
class Hydrology:
    filled: np.ndarray
    receivers: np.ndarray       # flat indices
    accumulation: np.ndarray    # cells draining through each cell
    drainage: np.ndarray        # stream mask
    hand: np.ndarray            # height above nearest drainage
    order_desc: np.ndarray      # cells sorted high -> low


def compute_hydrology(z: np.ndarray, cell: float, stream_fraction: float = 0.01) -> Hydrology:
    h, w = z.shape
    span = float(np.ptp(z)) or 1.0
    filled = priority_flood(z, eps=span * 1e-6)
    rec = d8_receivers(filled, cell)
    order = np.argsort(filled.ravel(), kind="stable")[::-1]

    acc = np.ones(h * w, np.float64)
    rec_l = rec.tolist()
    acc_l = acc.tolist()
    for i in order.tolist():
        j = rec_l[i]
        if j >= 0:
            acc_l[j] += acc_l[i]
    acc = np.array(acc_l).reshape(h, w)

    threshold = max(30.0, stream_fraction * h * w)
    drainage = acc >= threshold
    fflat = filled.ravel()
    base = fflat.copy()
    drain_flat = drainage.ravel()
    base_l = base.tolist()
    for i in order[::-1].tolist():           # low -> high: receivers first
        j = rec_l[i]
        if not drain_flat[i] and j >= 0:
            base_l[i] = base_l[j]
    hand = np.maximum(fflat - np.array(base_l), 0).reshape(h, w)
    return Hydrology(filled=filled, receivers=rec, accumulation=acc, drainage=drainage,
                     hand=hand.astype(np.float32), order_desc=order)
