"""Neighbour stencils for the MTT solver and the cells each straight segment passes through."""
import numpy as np

_BASE = {
    8: [(0, 1), (1, 1)],
    16: [(0, 1), (1, 1), (1, 2)],
    32: [(0, 1), (1, 1), (1, 2), (1, 3), (2, 3)],
}


def offsets(n):
    base = _BASE[n]
    out = set()
    for a, b in base:
        for s1 in (1, -1):
            for s2 in (1, -1):
                out.add((s1 * a, s2 * b))
                out.add((s1 * b, s2 * a))
    out.discard((0, 0))
    return sorted(out, key=lambda o: np.arctan2(o[1], -o[0]))


def traverse(dy, dx):
    """Cells (dy, dx relative to the start cell) crossed by the segment from the start cell centre to the target cell
    centre, with the fraction of the segment length in each, and grid-corner crossings (pairs of side cells)."""
    x0, y0 = 0.5, 0.5
    x1, y1 = dx + 0.5, dy + 0.5
    ts = {0.0, 1.0}
    for X in range(min(0, dx), max(0, dx) + 2):
        if dx != 0:
            t = (X - x0) / (x1 - x0)
            if 0 < t < 1:
                ts.add(round(t, 12))
    for Y in range(min(0, dy), max(0, dy) + 2):
        if dy != 0:
            t = (Y - y0) / (y1 - y0)
            if 0 < t < 1:
                ts.add(round(t, 12))
    ts = sorted(ts)
    cells = []
    for ta, tb in zip(ts[:-1], ts[1:]):
        tm = 0.5 * (ta + tb)
        cx = int(np.floor(x0 + tm * (x1 - x0)))
        cy = int(np.floor(y0 + tm * (y1 - y0)))
        if cells and cells[-1][0] == cy and cells[-1][1] == cx:
            cells[-1][2] += tb - ta
        else:
            cells.append([cy, cx, tb - ta])
    corners = []
    for t in ts[1:-1]:
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if abs(x - round(x)) < 1e-9 and abs(y - round(y)) < 1e-9:
            X, Y = int(round(x)), int(round(y))
            sx = 1 if dx > 0 else -1
            sy = 1 if dy > 0 else -1
            # cells touching the corner off the path
            a = (Y - (1 if sy > 0 else 0), X - (1 if sx > 0 else 0) + (1 if sx > 0 else -1))
            b = (Y - (1 if sy > 0 else 0) + (1 if sy > 0 else -1), X - (1 if sx > 0 else 0))
            corners.append((a, b))
    return cells, corners


def build(n=16):
    offs = offsets(n)
    K = len(offs)
    off_dy = np.array([o[0] for o in offs], np.int64)
    off_dx = np.array([o[1] for o in offs], np.int64)
    off_len = np.hypot(off_dy, off_dx).astype(float)
    seg_start = np.zeros(K, np.int64)
    seg_n = np.zeros(K, np.int64)
    sdy, sdx, sfr = [], [], []
    corner_a = np.full((K, 2), -999, np.int64)
    corner_b = np.full((K, 2), -999, np.int64)
    for k, (dy, dx) in enumerate(offs):
        cells, corners = traverse(dy, dx)
        seg_start[k] = len(sdy)
        seg_n[k] = len(cells)
        for cy, cx, fr in cells:
            sdy.append(cy)
            sdx.append(cx)
            sfr.append(fr)
        if corners:
            (ay, ax), (by, bx) = corners[0]
            corner_a[k] = (ay, ax)
            corner_b[k] = (by, bx)
    idx = {o: i for i, o in enumerate(offs)}
    opp = np.array([idx[(-o[0], -o[1])] for o in offs], np.int64)
    nb_prev = np.array([(i - 1) % K for i in range(K)], np.int64)
    nb_next = np.array([(i + 1) % K for i in range(K)], np.int64)
    return dict(opp=opp, nb_prev=nb_prev, nb_next=nb_next, off_dy=off_dy, off_dx=off_dx, off_len=off_len, seg_start=seg_start, seg_n=seg_n,
                seg_dy=np.array(sdy, np.int64), seg_dx=np.array(sdx, np.int64), seg_frac=np.array(sfr, float),
                corner_a=corner_a, corner_b=corner_b)
