"""Freeze WakeNet's second exam (17 Nov 2026): years no version has seen.

  * Kelmarsh 2022, 2023, 2024 (same farm, later years)
  * Penmanshiel 2021, 2022, 2023 (the unseen farm, later years; WT01-10 is published for Feb-Dec 2023 only,
    Penmanshiel 2024 has only WT11-15 and is left out because WakeNet reads whole farms)

Same quality control as realdata.build_snapshots. Penmanshiel 2023 WT01-10 comes in four files per turbine, so this
script reads every file for a turbine and joins them (realdata.py keeps only the last file it finds, which is right for
the v1.0 years, each one file per turbine). Writes work/proc/*_exam2_*.npz and training/split_exam2.json with
timestamp hashes. Run once, before anything is scored on these years; never edit split_exam2.json afterwards.

    python fetch_data.py --exam2
    python exam2_split.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402
import realdata as R  # noqa: E402
import split as S  # noqa: E402


def _read_year_frames_joined(farm: str, year: int, titles: list[str]) -> list[pd.DataFrame]:
    raw = Path(_P.work("raw"))
    prefix = "Kelmarsh" if farm == "kelmarsh" else "Penmanshiel"
    zips = sorted(raw.glob(f"{prefix}_SCADA_{year}_*.zip"))
    if not zips:
        raise FileNotFoundError(f"Missing raw SCADA zips for {farm} {year} in {raw}")
    parts: dict[str, list[pd.DataFrame]] = {t: [] for t in titles}
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if "Turbine_Data_" not in name or not name.endswith(".csv"):
                    continue
                for title in titles:
                    tag = "Turbine_Data_" + title.replace(" ", "_") + "_"
                    if tag in name:
                        with zf.open(name) as fh:
                            d = pd.read_csv(fh, skiprows=9, usecols=R.SCADA_COLS)
                        parts[title].append(d.rename(columns={"# Date and time": "ts"}))
                        break
    missing = [t for t in titles if not parts[t]]
    if missing:
        raise RuntimeError(f"{farm} {year}: missing turbine CSVs for {missing}")
    out = []
    for t in titles:
        d = pd.concat(parts[t], ignore_index=True)
        out.append(d.drop_duplicates(subset=["ts"]).set_index("ts").sort_index())
    return out


ROLES = {
    "exam2_kelmarsh": ("kelmarsh", [2022, 2023, 2024], "kelmarsh_exam2_2022_2024.npz"),
    "exam2_penmanshiel": ("penmanshiel", [2021, 2022, 2023], "penmanshiel_exam2_2021_2023.npz"),
}


def main():
    out_path = os.path.join(HERE, "split_exam2.json")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists and is frozen; not rebuilding it.")
    R._read_year_frames = _read_year_frames_joined
    roles = {}
    for role, (farm, years, fname) in ROLES.items():
        R.build_snapshots(farm, years, fname)
        st = S._stats(fname)
        st["years"] = years
        roles[role] = st
    # No overlap with any v1.0 role (backdrop, dev, exam 1)
    v1 = json.load(open(os.path.join(HERE, "split.json"), encoding="utf-8"))
    old_ts = set()
    for r in v1["roles"].values():
        old_ts |= set(np.load(_P.work("proc", r["file"]))["ts"].tolist())
    for role, (farm, _, fname) in ROLES.items():
        z = np.load(_P.work("proc", fname))
        if farm == "kelmarsh":
            assert not (set(z["ts"].tolist()) & old_ts), f"{role} overlaps a v1.0 role"
    doc = {
        "frozen": "2026-11-17",
        "purpose": "WakeNet exam 2: years no version has seen; scored once after the v1.1 decision",
        "seed": 20261117,
        "quality_control": "identical to realdata.build_snapshots (5-12 m/s front-row inflow, uncurtailed, online flags)",
        "roles": roles,
    }
    doc["split_sha256"] = hashlib.sha256(json.dumps(doc["roles"], sort_keys=True).encode("utf-8")).hexdigest()
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("n_snapshots", "first_ts", "last_ts")}
                      for k, v in roles.items()}, indent=2))
    print(f"Wrote {out_path} (sha256 {doc['split_sha256'][:16]}...)")


if __name__ == "__main__":
    main()
