"""r7_context input: today's 35 feature rows plus a reference built from the same unit's recent days (r7 plan,
NOTEBOOK 2026-11-17). Pure numpy; faultfinder.py carries the same code.

Rows 0-34   today's features (features.featurize)
Rows 35-69  per-bin median of the reference days' 35 rows
Rows 70-80  today minus that median for rows 0-10 (6 temperatures, 5 physics differences)
Row  81     number of reference days / 3
No reference: rows 35-81 are 0 (single-day mode). A reference day's label is never an input.
"""
import numpy as np

NF1 = 35
NDIFF = 11
NF7 = NF1 + NF1 + NDIFF + 1        # 82
MAX_REF = 3


def context(f_today, f_refs=None):
    """f_today [35, 96]; f_refs [n, 35, 96] (n = 0..3) or None -> [82, 96] float32."""
    f_today = np.asarray(f_today, np.float32)
    out = np.zeros((NF7,) + f_today.shape[1:], np.float32)
    out[:NF1] = f_today
    if f_refs is not None and len(f_refs):
        m = np.median(np.asarray(f_refs, np.float32), axis=0)
        out[NF1:2 * NF1] = m
        out[2 * NF1:2 * NF1 + NDIFF] = f_today[:NDIFF] - m[:NDIFF]
        out[-1] = len(f_refs) / 3.0
    return out


def context_batch(ft, fr, nref):
    """ft [B, 35, T]; fr [B, 3, 35, T] (slots beyond nref ignored); nref [B] in 0..3 -> [B, 82, T] float32.
    Same numbers as context() row by row."""
    ft = np.asarray(ft, np.float32)
    B = len(ft)
    out = np.zeros((B, NF7, ft.shape[-1]), np.float32)
    out[:, :NF1] = ft
    for n in (1, 2, 3):
        i = np.where(nref == n)[0]
        if len(i) == 0:
            continue
        m = np.median(np.asarray(fr[i, :n], np.float32), axis=1)
        out[i, NF1:2 * NF1] = m
        out[i, 2 * NF1:2 * NF1 + NDIFF] = ft[i, :NDIFF] - m[:, :NDIFF]
        out[i, -1] = n / 3.0
    return out


def draw_refs(day, run, fault_day, avail, rng, p_none=0.3):
    """Training / frozen-set rule: with p_none no reference; else 1-3 days drawn from days max(1, k - 7)..k - 1 of
    the same run (faulted run from the fault start, healthy run before). avail(k, run) -> row index or -1.
    Returns a sorted list of row indices (possibly empty)."""
    if rng.random() < p_none:
        return []
    cand = []
    for j in range(max(1, day - 7), day):
        r = 1 if (run == 1 and j >= fault_day) else 0
        i = avail(j, r)
        if i >= 0:
            cand.append(i)
    if not cand:
        return []
    n = min(int(rng.integers(1, MAX_REF + 1)), len(cand))
    return sorted(int(c) for c in rng.choice(cand, n, replace=False))
