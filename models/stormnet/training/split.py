"""Frozen year roles, disjoint dev events, causal wind histories and future-only targets."""
import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from noise import HERE, PROTOCOL, RESULTS, fill_wind, read_year, write_json

H = PROTOCOL["history_hours"]
K = PROTOCOL["horizon_hours"]


def blocks(dst, time):
    crossings = np.flatnonzero(dst <= -50)
    storms = np.split(crossings, np.flatnonzero(np.diff(crossings) > 72) + 1)
    intervals = []
    cursor = 0

    def quiet(stop):
        nonlocal cursor
        while cursor < stop:
            end = min(cursor + 72, stop)
            intervals.append((cursor, end, "quiet"))
            cursor = end

    for storm in storms:
        if len(storm):
            lo = max(0, storm[0] - 24)
            hi = min(len(dst), storm[-1] + 25)
            quiet(lo)
            intervals.append((lo, hi, "storm"))
            cursor = hi
    quiet(len(dst))
    groups = np.empty(len(dst), dtype=int)
    meta = []
    for group, (lo, hi, kind) in enumerate(intervals):
        groups[lo:hi] = group
        vals = dst[lo:hi]
        meta.append(dict(id=f"{kind}-{time[lo]}", kind=kind, start=str(time[lo]),
                         end_exclusive=str(time[hi - 1] + np.timedelta64(1, "h")), hours=int(hi - lo),
                         minimum=float(np.nanmin(vals)) if np.isfinite(vals).any() else None))
    return groups, meta


def period(years, reserve=False):
    if any(b != a + 1 for a, b in zip(years, years[1:])):
        raise ValueError("Build each contiguous period separately")
    raw = [read_year(y, index=True, reserve=reserve) for y in years]
    time = np.concatenate([r[0] for r in raw])
    wind = fill_wind(np.concatenate([r[1] for r in raw]))
    dst = np.concatenate([r[2] for r in raw])
    group, meta = blocks(dst, time)
    i = np.arange(H - 1, len(dst) - K)
    windows = sliding_window_view(wind, H, axis=0).swapaxes(1, 2)
    future = sliding_window_view(dst[1:], K)[i]
    ok_wind = np.isfinite(windows[i - H + 1]).all(axis=(1, 2))
    ok_index = np.isfinite(dst[i]) & np.isfinite(future).all(axis=1)
    ok = ok_wind & ok_index
    idx = i[ok]
    cases = dict(wind=windows[idx - H + 1], dst=dst[idx], truth=future[ok].min(axis=1),
                 past_dst=sliding_window_view(dst, H)[idx - H + 1],
                 future=future[ok], future_wind=wind[idx[:, None] + np.arange(1, K + 1)],
                 time=time[idx] + np.timedelta64(1, "h"), group=group[idx], row=idx)
    counts = dict(raw_hours=len(time), possible=len(i), missing_wind=int((~ok_wind).sum()),
                  missing_index=int((~ok_index).sum()), valid=len(idx))
    return cases, meta, counts, {str(y): r[3] for y, r in zip(years, raw)}


def subset(cases, mask):
    return {key: value[mask] for key, value in cases.items()}


def choose_dev():
    rng = np.random.default_rng(PROTOCOL["split_seed"])
    result = {}
    for year in PROTOCOL["dev_years"]:
        cases, meta, _, _ = period([year])
        available = {i for i, count in zip(*np.unique(cases["group"], return_counts=True)) if count >= 24}
        storms = sorted((i for i in available if meta[i]["kind"] == "storm"),
                        key=lambda i: meta[i]["minimum"])
        quiet = sorted(i for i in available if meta[i]["kind"] == "quiet" and meta[i]["hours"] == 72)
        chosen = [int(rng.choice(part)) for part in np.array_split(storms, min(6, len(storms))) if len(part)]
        chosen.extend(rng.choice(quiet, size=min(6, len(quiet)), replace=False).tolist())
        result[str(year)] = [meta[i] for i in sorted(chosen)]
    return result


def dev_cases():
    frozen = json.loads((HERE / "split.json").read_text())
    sets = {}
    for year in PROTOCOL["dev_years"]:
        cases, meta, _, _ = period([year])
        expected = {m["id"]: m for m in frozen["dev_blocks"][str(year)]}
        kept = [i for i, m in enumerate(meta) if m["id"] in expected]
        if len(kept) != len(expected) or any(meta[i] != expected[meta[i]["id"]] for i in kept):
            raise ValueError("Frozen dev block changed")
        sets[str(year)] = subset(cases, np.isin(cases["group"], kept))
    return sets


if __name__ == "__main__":
    doc = dict(protocol=PROTOCOL, dev_blocks=choose_dev())
    path = HERE / "split.json"
    if path.exists() and doc != json.loads(path.read_text()):
        raise ValueError("Frozen split differs")
    write_json(RESULTS / "split.json", doc)
