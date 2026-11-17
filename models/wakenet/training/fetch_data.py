"""Download Kelmarsh and Penmanshiel open wind-farm SCADA archives from Zenodo into work/raw/.

Sources (both CC BY 4.0, Charlie Plumley / Cubico Sustainable Investments):
  * Kelmarsh Wind Farm (6 x Senvion MM92, D = 92 m, H = 68.5 / 78.5 m):
    https://doi.org/10.5281/zenodo.5841834  (latest record 16807551)
  * Penmanshiel Wind Farm (14 x Senvion MM82, D = 82 m, H = 59.0 m):
    https://doi.org/10.5281/zenodo.5946808  (latest record 16807304)

Roles in WakeNet (frozen before any training or evaluation):
  * Kelmarsh 2016-2018 : freestream climate prior & real unwaked turbulence/inflow backdrop (zero wake labels)
  * Kelmarsh 2019      : real dev set (simulator calibration, twin test, per-season dev gate, widen-only sigma)
  * Kelmarsh 2020-2021 : held-out real exam (same farm, unseen years)
  * Penmanshiel 2019-2020 : zero-shot unseen-farm real exam (14 turbines, different rotor/hub/terrain, never in dev or backdrop)
"""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import _paths as _P

KELMARSH_REC = "16807551"
PENMANSHIEL_REC = "16807304"

FILES = [
    # Static metadata
    (KELMARSH_REC, "Kelmarsh_WT_static.csv"),
    (KELMARSH_REC, "Kelmarsh_WT_dataSignalMapping.csv"),
    (PENMANSHIEL_REC, "Penmanshiel_WT_static.csv"),
    # Kelmarsh 2016-2021 (backdrop + dev + exam)
    (KELMARSH_REC, "Kelmarsh_SCADA_2016_3082.zip"),
    (KELMARSH_REC, "Kelmarsh_SCADA_2017_3083.zip"),
    (KELMARSH_REC, "Kelmarsh_SCADA_2018_3084.zip"),
    (KELMARSH_REC, "Kelmarsh_SCADA_2019_3085.zip"),
    (KELMARSH_REC, "Kelmarsh_SCADA_2020_3086.zip"),
    (KELMARSH_REC, "Kelmarsh_SCADA_2021_4456.zip"),
    # Penmanshiel 2019-2020 (all 14 turbines = WT01-10 + WT11-15; zero-shot exam)
    (PENMANSHIEL_REC, "Penmanshiel_SCADA_2019_WT01-10_3112.zip"),
    (PENMANSHIEL_REC, "Penmanshiel_SCADA_2019_WT11-15_3117.zip"),
    (PENMANSHIEL_REC, "Penmanshiel_SCADA_2020_WT01-10_3109.zip"),
    (PENMANSHIEL_REC, "Penmanshiel_SCADA_2020_WT11-15_3118.zip"),
]


def _valid_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                return len(zf.namelist()) > 0
        except zipfile.BadZipFile:
            return False
    return True


def fetch_one(rec: str, fname: str) -> str:
    raw_dir = Path(_P.work("raw", fname, mkdir=True)).parent
    dest = raw_dir / fname
    if _valid_file(dest):
        return f"cached  {fname} ({dest.stat().st_size / 1e6:.1f} MB)"
    tmp = raw_dir / f"{fname}.part"
    url = f"https://zenodo.org/api/records/{rec}/files/{fname}/content"
    cmd = [
        "curl", "-fSL", "--retry", "4", "--retry-delay", "5",
        "-C", "-", "-o", str(tmp), url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {fname}: {r.stderr.strip()}")
    tmp.replace(dest)
    if not _valid_file(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file failed integrity check: {fname}")
    return f"fetched {fname} ({dest.stat().st_size / 1e6:.1f} MB)"


def main(jobs: int = 3) -> int:
    print(f"Downloading {len(FILES)} Zenodo files to {_P.work('raw')} (jobs={jobs})...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fetch_one, rec, fn): fn for rec, fn in FILES}
        for fut in concurrent.futures.as_completed(futs):
            fn = futs[fut]
            try:
                msg = fut.result()
                print(f"  [ok] {msg}", flush=True)
            except Exception as exc:
                print(f"  [ERR] {fn}: {exc}", file=sys.stderr, flush=True)
                return 1
    print("All Zenodo files ready.", flush=True)
    return 0


if __name__ == "__main__":
    j = int(os.environ.get("FETCH_JOBS", "3"))
    sys.exit(main(j))
