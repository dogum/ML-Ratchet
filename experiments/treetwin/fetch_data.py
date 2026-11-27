"""Download public raw datasets for the TreeTwin + ConformalRatchet three-platform experiment.

Platforms:
  1. ``airfoilnet`` : UIUC Low-Speed Airfoil Tests Vols 1-5 + UIUC coordinate database (~10 MB)
  2. ``building``   : EIA RECS 2020 v7 microdata (56 MB) + 10 NASA POWER 2020 US climate-zone hourly weather stations
  3. ``vehicle``    : Vehicle Energy Dataset (VED) Part 1 & 2, 4 USGS 3DEP 1 m lidar tiles, 3 IEM ASOS weather CSVs

All downloads land in git-ignored folders (``experiments/treetwin/work/`` or ``shared/*/data/raw/``)
and are resumable. Never touches frozen split files.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

import tt_paths as P

UA = "mlratchet-treetwin/1.0 (+https://github.com/dogum/ML-Ratchet; research)"


def download(url: str, dst: Path, min_bytes: int = 100) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size >= min_bytes:
        print(f"  [skip] {dst.name} ({dst.stat().st_size / 1e6:.2f} MB)", flush=True)
        return dst
    part = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    part.replace(dst)
    print(f"  [ok]   {dst.name} ({dst.stat().st_size / 1e6:.2f} MB)", flush=True)
    return dst


def fetch_airfoilnet() -> None:
    print("=== Fetching AirfoilNet (UIUC LSAT Vols 1-5 + coord_seligFmt) ===", flush=True)
    raw = P.work("airfoilnet", "raw")
    raw.mkdir(parents=True, exist_ok=True)
    base = "https://m-selig.ae.illinois.edu/"
    for v in range(1, 6):
        zp = download(f"{base}pd/pub/lsat/volume0{v}.zip", raw / f"volume0{v}.zip")
        out_dir = raw / f"volume0{v}"
        if not out_dir.exists():
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(out_dir)
    cz = download(f"{base}ads/archives/coord_seligFmt.zip", raw / "coord_seligFmt.zip")
    cdir = raw / "coord_seligFmt"
    if not cdir.exists():
        with zipfile.ZipFile(cz) as zf:
            zf.extractall(cdir)
    print("  AirfoilNet raw archives ready.", flush=True)


# 10 representative US cities spanning IECC climate zones 1A..7 for BuildingSim RECS simulation
CLIMATE_STATIONS = {
    "1A": ("Miami", 25.76, -80.19),
    "2A": ("Houston", 29.76, -95.37),
    "2B": ("Phoenix", 33.45, -112.07),
    "3A": ("Atlanta", 33.75, -84.39),
    "3B": ("LosAngeles", 34.05, -118.24),
    "3C": ("SanFrancisco", 37.77, -122.42),
    "4A": ("Baltimore", 39.29, -76.61),
    "4C": ("Seattle", 47.61, -122.33),
    "5A": ("Chicago", 41.88, -87.63),
    "6A": ("Minneapolis", 44.98, -93.27),
}


def fetch_building() -> None:
    print("=== Fetching BuildingDetective (EIA RECS 2020 v7 + NASA POWER US 2020 hourly weather) ===", flush=True)
    broot = P.REPO / "shared" / "building" / "data"
    recs_csv = broot / "raw" / "recs" / "recs2020_public_v7.csv"
    download(
        "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv",
        recs_csv,
        min_bytes=10_000_000,
    )
    recs_pq = broot / "proc" / "eia" / "recs2020.parquet"
    if not recs_pq.exists():
        P.add_sys_path(broot / "scripts")
        import build_recs_cbecs
        build_recs_cbecs.recs()
    else:
        print(f"  [skip] {recs_pq.name}", flush=True)

    wx_dir = P.work("building", "wx")
    wx_dir.mkdir(parents=True, exist_ok=True)

    def _get_station(item):
        cz, (city, lat, lon) = item
        dst = wx_dir / f"power_{cz}_{city}_2020.json"
        url = (
            "https://power.larc.nasa.gov/api/temporal/hourly/point"
            f"?parameters=T2M,ALLSKY_SFC_SW_DWN,WS10M,RH2M&community=RE"
            f"&longitude={lon}&latitude={lat}&start=20200101&end=20201231&format=JSON"
        )
        download(url, dst, min_bytes=100_000)

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_get_station, CLIMATE_STATIONS.items()))
    print("  Building RECS 2020 + NASA POWER climate weather ready.", flush=True)


def fetch_vehicle() -> None:
    print("=== Fetching Vehicle (VED + USGS 3DEP Washtenaw 1m DEM + IEM ASOS weather) ===", flush=True)
    vraw = P.REPO / "shared" / "vehicle" / "data" / "raw"
    ved_base = "https://github.com/gsoh/VED/raw/master/Data/"
    tasks = [
        (f"{ved_base}VED_Static_Data_ICE%26HEV.xlsx", vraw / "ved" / "VED_Static_Data_ICE&HEV.xlsx"),
        (f"{ved_base}VED_Static_Data_PHEV%26EV.xlsx", vraw / "ved" / "VED_Static_Data_PHEV&EV.xlsx"),
        (f"{ved_base}VED_DynamicData_Part1.7z", vraw / "ved" / "VED_DynamicData_Part1.7z"),
        (f"{ved_base}VED_DynamicData_Part2.7z", vraw / "ved" / "VED_DynamicData_Part2.7z"),
    ]
    dem_tpl = (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/"
        "MI_31Co_Washtenaw_2016/TIFF/USGS_one_meter_{t}_MI_31Co_Washtenaw_2016.tif"
    )
    for t in ("x26y468", "x26y469", "x27y468", "x27y469"):
        tasks.append((dem_tpl.format(t=t), vraw / "elev" / f"{t}.tif"))
    asos_tpl = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={s}"
        "&data=tmpc&data=dwpc&data=alti&data=sknt&data=drct&data=wxcodes&data=p01i"
        "&year1=2017&month1=10&day1=31&year2=2018&month2=11&day2=13&tz=Etc%2FUTC"
        "&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4"
    )
    for s in ("ARB", "YIP", "DTW"):
        tasks.append((asos_tpl.format(s=s), vraw / "weather" / f"asos_{s}.csv"))

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(download, u, d) for u, d in tasks]
        for f in futs:
            f.result()

    # Run official Vehicle VED rebuild stages (stage1 -> survey -> splits_ved -> dem -> stage2 -> index)
    vidx = P.REPO / "shared" / "vehicle" / "data" / "proc" / "ved" / "index.parquet"
    if not vidx.exists():
        print("  Running official Vehicle VED rebuild stages (stage1,survey,splits_ved,dem,stage2,index)...", flush=True)
        subprocess.run(
            [sys.executable, "-I", str(P.REPO / "shared" / "vehicle" / "data" / "build.py"),
             "--only", "stage1,survey,splits_ved,dem,stage2,index", "--jobs", "4"],
            check=True,
        )
    else:
        print(f"  [skip] VED index already built: {vidx}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "airfoilnet", "building", "vehicle"], default="all")
    args = ap.parse_args()
    if args.only in ("all", "airfoilnet"):
        fetch_airfoilnet()
    if args.only in ("all", "building"):
        fetch_building()
    if args.only in ("all", "vehicle"):
        fetch_vehicle()


if __name__ == "__main__":
    main()
