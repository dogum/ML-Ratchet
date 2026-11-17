"""Fetch dated OMNI2 ASCII files and extract only the columns permitted for each role."""
from __future__ import annotations

import calendar
import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
from _paths import SRC, work

HERE = Path(SRC)
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
# Published OMNI2 fixed-width words 1–41. Dst (41) is read only by the labelled loader.
WIDTHS = [4, 4, 3, 5, 3, 3, 4, 4] + [6] * 14 + [9, 6, 6, 6, 6, 6, 6, 9, 6, 6, 6, 6, 6, 7, 7, 6, 3, 4, 6]
EDGES = np.cumsum([0] + WIDTHS)
WIND_WORDS = (17, 25, 29)


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _raw(year):
    if year in PROTOCOL["sealed_reserve_years"]:
        raise ValueError("The v1 reserve is sealed.")
    path = Path(work("raw", f"omni2_{year}.dat", mkdir=True))
    if not path.exists():
        with urlopen(URL.format(year=year), timeout=45) as response:
            data = response.read()
        if len(data.splitlines()) != (365 + calendar.isleap(year)) * 24:
            raise ValueError(f"Incomplete annual OMNI table for {year}")
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temp.write(data)
        os.replace(temp.name, path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    for fname in ("sources_dev.json", "sources_primary.json"):
        manifest = HERE / fname
        if manifest.exists():
            frozen = json.loads(manifest.read_text()).get(str(year))
            if frozen and digest != frozen:
                raise ValueError(f"Source revision changed for {year}: {digest} != {frozen}")
    return data, digest


def read_year(year, *, index=False):
    labelled_years = PROTOCOL["dev_years"] + PROTOCOL["primary_years"]
    permitted = labelled_years if index else PROTOCOL["training_wind_years"] + PROTOCOL["frozen_wind_years"]
    if year not in permitted:
        raise ValueError(f"Year {year} outside the requested data role")
    data, digest = _raw(year)
    words = (1, 2, 3, *WIND_WORDS, *((41,) if index else ()))
    slices = [slice(EDGES[w - 1], EDGES[w]) for w in words]
    records = np.array([[float(line[key]) for key in slices] for line in io.BytesIO(data)])
    hours = np.arange((365 + calendar.isleap(year)) * 24)
    if len(records) != len(hours) or not np.all(records[:, :3] == np.column_stack(
            [np.full(len(hours), year), hours // 24 + 1, hours % 24])):
        raise ValueError(f"Missing, duplicated or unordered hours in {year}")
    wind = records[:, 3:6].copy()
    wind[wind == np.array([999.9, 9999., 99.99])] = np.nan
    wind[:, 1:][wind[:, 1:] <= 0] = np.nan
    dst = records[:, 6].copy() if index else None
    if index:
        dst[dst == 99999] = np.nan
    start = np.datetime64(f"{year}-01-01T00", "h")
    return start + hours, wind, dst, digest


def fill_wind(wind):
    """Forward-fill each field by at most two hours. Separate input prevents chained fills."""
    filled = wind.copy()
    for lag in (1, 2):
        target = filled[lag:]
        np.copyto(target, wind[:-lag], where=~np.isfinite(target) & np.isfinite(wind[:-lag]))
    return filled


def wind_period(years):
    raw = [read_year(y) for y in years]
    return fill_wind(np.concatenate([r[1] for r in raw])), {str(y): r[3] for y, r in zip(years, raw)}


def wind_law(wind):
    """One-hour regression to the training-era mean for coupling and sqrt pressure."""
    e = np.maximum(-wind[:, 0], 0) * wind[:, 1] / 1000
    out = []
    for series in (e, np.sqrt(wind[:, 2])):
        x, y = series[:-1], series[1:]
        ok = np.isfinite(x) & np.isfinite(y)
        mu = np.mean(x[ok])
        rho = np.dot(x[ok] - mu, y[ok] - mu) / np.sum((x[ok] - mu) ** 2)
        out.extend((float(mu), float(np.clip(rho, 0, 1))))
    return out
