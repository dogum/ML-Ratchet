"""Fetch public bridge monitoring datasets from Zenodo into work/raw/.

All endpoints are documented public Zenodo records:
  - KW51 railway bridge (Maes & Lombaert 2020): zenodo.org/records/3745914 (CC BY-NC-SA 4.0)
  - KW51 finite element model (Maes & Lombaert 2025): zenodo.org/records/17366772 (CC BY-NC-SA 4.0)
  - Concrete box-girder bridge (Koehncke et al. 2026): zenodo.org/records/20522191 (CC BY 4.0)
  - Hell Bridge Test Arena (NTNU): zenodo.org/records/14632942 (CC BY 4.0)

Nothing downloaded here is committed to git.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import zipfile

from _paths import work


def _zenodo_file_url(record_id: int, filename: str) -> tuple[str, int]:
    meta_url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(meta_url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        doc = json.loads(r.read().decode("utf-8"))
    for f in doc.get("files", []):
        if f["key"] == filename:
            return f["links"]["self"], int(f["size"])
    raise FileNotFoundError(f"{filename} not found in Zenodo record {record_id}")


def _download(url: str, dst: str, expected_size: int | None = None) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and (expected_size is None or os.path.getsize(dst) == expected_size):
        print(f"  cached: {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.2f} MB)")
        return dst
    tmp = dst + ".part"
    start = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={start}-"} if start > 0 else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        mode = "ab" if (start > 0 and r.status == 206) else "wb"
        if mode == "wb":
            start = 0
        total = expected_size or int(r.headers.get("Content-Length", 0)) + start
        done = start
        last_mb = done // (50 * 1024 * 1024)
        with open(tmp, mode) as out:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                mb50 = done // (50 * 1024 * 1024)
                if mb50 > last_mb:
                    last_mb = mb50
                    pct = (100.0 * done / total) if total else 0.0
                    print(f"  ... {os.path.basename(dst)}: {done/1e6:.1f} / {total/1e6:.1f} MB ({pct:.1f}%)", flush=True)
    os.replace(tmp, dst)
    print(f"  fetched: {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.2f} MB)")
    return dst


def fetch_kw51() -> None:
    print("[KW51] Zenodo 3745914 + 17366772")
    url, sz = _zenodo_file_url(3745914, "trackedmodes.zip")
    zpath = _download(url, work("raw", "kw51", "trackedmodes.zip", mkdir=True), sz)
    mat_dst = work("raw", "kw51", "trackedmodes.mat")
    if not os.path.exists(mat_dst):
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                if member.endswith("trackedmodes.mat"):
                    with zf.open(member) as src, open(mat_dst, "wb") as dst:
                        dst.write(src.read())
                    print(f"  extracted: trackedmodes.mat ({os.path.getsize(mat_dst)/1e6:.2f} MB)")
                    break
    url_fem, sz_fem = _zenodo_file_url(17366772, "model_kw51.m")
    _download(url_fem, work("raw", "kw51", "model_kw51.m", mkdir=True), sz_fem)


def fetch_boxgirder() -> None:
    print("[Concrete Box-Girder] Zenodo 20522191 + NASA POWER T2M (Hamburg)")
    url, sz = _zenodo_file_url(20522191, "box_girder_bridge_eigenfrequencies.h5")
    _download(url, work("raw", "boxgirder", "box_girder_bridge_eigenfrequencies.h5", mkdir=True), sz)
    wx_dst = work("raw", "boxgirder", "nasa_power_hamburg_2024.json", mkdir=True)
    if not os.path.exists(wx_dst):
        wx_url = (
            "https://power.larc.nasa.gov/api/temporal/hourly/point?"
            "parameters=T2M&community=RE&longitude=10.0&latitude=53.55&start=20231225&end=20240423&format=JSON"
        )
        _download(wx_url, wx_dst)
    else:
        print(f"  cached: {os.path.basename(wx_dst)} ({os.path.getsize(wx_dst)/1e6:.2f} MB)")


def fetch_hbta(include_h5: bool = True) -> None:
    print("[Hell Bridge Test Arena] Zenodo 14632942")
    for fname in ("functions.py", "main.py"):
        url, sz = _zenodo_file_url(14632942, fname)
        _download(url, work("raw", "hbta", fname, mkdir=True), sz)
    if include_h5:
        url, sz = _zenodo_file_url(14632942, "data_100Hz.h5")
        _download(url, work("raw", "hbta", "data_100Hz.h5", mkdir=True), sz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "fast", "kw51", "boxgirder", "hbta"], default="all")
    args = ap.parse_args()
    if args.only in ("all", "fast", "kw51"):
        fetch_kw51()
    if args.only in ("all", "fast", "boxgirder"):
        fetch_boxgirder()
    if args.only in ("all", "fast"):
        fetch_hbta(include_h5=(args.only == "all"))
    elif args.only == "hbta":
        fetch_hbta(include_h5=True)


if __name__ == "__main__":
    main()
