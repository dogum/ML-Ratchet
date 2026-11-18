"""Whole-storm replays for the instrument and lab (17 Nov 2026), chosen by rule after both exams were scored:
the deepest storm block of each reserve year (2022-2025), the deepest storm block of the primary exam (2016-2020),
and one 72-hour quiet block from the reserve drawn with the protocol's example seed. The rule looks only at the
observed Dst minimum, never at any forecast error.

Each storm keeps 36 hours of lead-in before the block and 3 hours after it, so every hour of the block can be
issued and verified. Writes ../examples/storms.npz and ../examples/storms.json (OMNI wind, Kyoto Dst; see
DATA_LICENSES.md).
"""
import json
from pathlib import Path

import numpy as np

from _paths import MODEL
from noise import PROTOCOL, fill_wind, read_year
from split import blocks

PAD_BEFORE, PAD_AFTER = 36, 3


def series(years, reserve):
    raw = [read_year(y, index=True, reserve=reserve) for y in years]
    return (np.concatenate([r[0] for r in raw]), fill_wind(np.concatenate([r[1] for r in raw])),
            np.concatenate([r[2] for r in raw]))


def cut(time, wind, dst, m, label, source):
    lo = int(np.searchsorted(time, np.datetime64(m["start"], "h"))) - PAD_BEFORE
    hi = int(np.searchsorted(time, np.datetime64(m["end_exclusive"], "h"))) + PAD_AFTER
    lo, hi = max(lo, 0), min(hi, len(time))
    return dict(label=label, source=source, block=m["id"], minimum=m["minimum"], hours=m["hours"],
                start_utc=str(time[lo]) + ":00Z", issue_from=PAD_BEFORE - 1 if lo > 0 else 35), \
        dict(wind=wind[lo:hi].astype("float32"), dst=dst[lo:hi].astype("float32"))


def main():
    picks, arrays = [], {}
    t, w, d = series(PROTOCOL["sealed_reserve_years"], True)
    _, meta = blocks(d, t)
    for year in PROTOCOL["sealed_reserve_years"]:
        storms = [m for m in meta if m["kind"] == "storm" and m["start"].startswith(str(year)) and m["minimum"] is not None]
        m = min(storms, key=lambda m: m["minimum"])
        picks.append(cut(t, w, d, m, f"{year} deepest storm", "exam 2 (reserve)"))
    quiet = [m for m in meta if m["kind"] == "quiet" and m["hours"] == 72]
    q = quiet[int(np.random.default_rng(PROTOCOL["example_seed"]).integers(len(quiet)))]
    t2, w2, d2 = series(PROTOCOL["primary_years"], False)
    _, meta2 = blocks(d2, t2)
    m2 = min((m for m in meta2 if m["kind"] == "storm" and m["minimum"] is not None), key=lambda m: m["minimum"])
    picks.append(cut(t2, w2, d2, m2, "Deepest primary-exam storm", "exam 1 (primary)"))
    picks.append(cut(t, w, d, q, "Quiet days", "exam 2 (reserve)"))
    picks.sort(key=lambda p: p[0]["minimum"])
    info = []
    for i, (m, a) in enumerate(picks):
        arrays[f"wind{i}"], arrays[f"dst{i}"] = a["wind"], a["dst"]
        info.append(dict(m, length=len(a["dst"])))
    out = Path(MODEL) / "examples"
    np.savez_compressed(out / "storms.npz", **arrays)
    (out / "storms.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    for m in info:
        print(m["label"], m["block"], m["minimum"], m["length"])


if __name__ == "__main__":
    main()
