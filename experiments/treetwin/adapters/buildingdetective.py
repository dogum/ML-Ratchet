"""BuildingDetective (RECS 2020, annual) adapter for TreeTwin + ConformalRatchet.

- Simulator : BuildingSim (shared/building/buildingsim, 5R1C + HVAC hourly simulation) driven by
              10 NASA POWER 2020 US climate-zone hourly weather stations (IECC 1A..6A).
- Features  : Exact 12-feature BuildingDetective `feat_annual` vector (HDD, CDD, log10 HDD/CDD,
              and presence + log10(kWh)-5 for elec, fuel, dheat, dcool).
- Target    : log10(floor_area_m2); headline metric is floor-area MdAPE (%).
- Monotone  : +1 on log10(kWh) for all four meter channels (more annual energy -> larger floor area).
- Real splits: Frozen RECS 2020 splits from shared/building/data/proc/splits.json
              (dev = 1,848 homes; exam = 7,401 homes). Zero real labels enter training.
- Twins     : 300 seeded real RECS 2020 dev homes simulated forward through BuildingSim using each
              home's true floor area, vintage, stories, occupants, climate zone and HVAC system.
- Strata    : Climate-band x heating-fuel (cold_fuel, cold_elec, mixed_fuel, mixed_elec,
              warm_fuel, warm_elec) for Mondrian conformal calibration.
- Incumbent : Released BuildingDetective v1.0 annual MLP (models/buildingdetective/weights/annual.npz).
"""
from __future__ import annotations

import concurrent.futures as cf
import functools
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import tt_paths as P

P.add_sys_path(P.REPO / "shared" / "building" / "buildingsim")
P.add_sys_path(P.REPO / "models" / "buildingdetective")

NAME = "buildingdetective"
TITLE = "BuildingDetective (RECS 2020)"
DOMAIN = "Buildings"
TARGET_NAME = "log10_floor_area"
UNIT = "log10 m2"
SHAP_SCALE = 100.0 * np.log(10.0)   # delta log10(A) -> approx % floor-area shift
SHAP_UNIT = "% floor-area shift"

FEATURE_NAMES = [
    "hdd / 3000",
    "cdd / 1500",
    "log10(hdd + 50)",
    "log10(cdd + 50)",
    "has_elec",
    "log10(elec_kWh) - 5",
    "has_fuel",
    "log10(fuel_kWh) - 5",
    "has_dheat",
    "log10(dheat_kWh) - 5",
    "has_dcool",
    "log10(dcool_kWh) - 5",
]

FEATURE_GROUPS = {
    "Electricity (kWh/yr)": [4, 5],
    "Fuel gas/oil (kWh/yr)": [6, 7],
    "Heating degree days": [0, 2],
    "Cooling degree days": [1, 3],
    "District meters": [8, 9, 10, 11],
}

# Physically justified monotonicity on log10(floor_area_m2): higher energy on any meter -> larger area
MONOTONE_CST = [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]

CLIMATE_META = {
    "1A": ("Miami", 25.76, -80.19, "America/New_York", 5.0),
    "2A": ("Houston", 29.76, -95.37, "America/Chicago", 15.0),
    "2B": ("Phoenix", 33.45, -112.07, "America/Phoenix", 330.0),
    "3A": ("Atlanta", 33.75, -84.39, "America/New_York", 320.0),
    "3B": ("LosAngeles", 34.05, -118.24, "America/Los_Angeles", 90.0),
    "3C": ("SanFrancisco", 37.77, -122.42, "America/Los_Angeles", 16.0),
    "4A": ("Baltimore", 39.29, -76.61, "America/New_York", 45.0),
    "4C": ("Seattle", 47.61, -122.33, "America/Los_Angeles", 120.0),
    "5A": ("Chicago", 41.88, -87.63, "America/Chicago", 180.0),
    "6A": ("Minneapolis", 44.98, -93.27, "America/Chicago", 260.0),
}
CZ_KEYS = list(CLIMATE_META.keys())
CZ_MAP = {
    "1A": "1A", "2A": "2A", "2B": "2B", "3A": "3A", "3B": "3B", "3C": "3C",
    "4A": "4A", "4B": "3B", "4C": "4C", "5A": "5A", "5B": "5A", "5C": "4C",
    "6A": "6A", "6B": "6A", "7": "6A", "7A": "6A", "7B": "6A", "8": "6A",
}

_WX_CACHE: dict[str, tuple[pd.DataFrame, dict, float, float]] = {}


def _load_wx(cz: str) -> tuple[pd.DataFrame, dict, float, float]:
    cz_k = CZ_MAP.get(str(cz), "4A")
    if cz_k in _WX_CACHE:
        return _WX_CACHE[cz_k]
    city, lat, lon, tz, elev = CLIMATE_META[cz_k]
    fp = P.work("building", "wx", f"power_{cz_k}_{city}_2020.json")
    d = json.loads(fp.read_text())
    p = d["properties"]["parameter"]
    keys = sorted(p["T2M"].keys())
    idx = pd.to_datetime(keys, format="%Y%m%d%H", utc=True)
    T = np.array([p["T2M"][k] for k in keys], dtype=float)
    GHI = np.maximum(0.0, np.array([p["ALLSKY_SFC_SW_DWN"][k] for k in keys], dtype=float))
    WS = np.maximum(0.1, np.array([p["WS10M"][k] for k in keys], dtype=float))
    RH = np.clip(np.array([p["RH2M"][k] for k in keys], dtype=float), 1.0, 100.0)
    W = pd.DataFrame({"T_out": T, "RH": RH, "GHI": GHI, "wind": WS}, index=idx)
    meta = {"lat": lat, "lon": lon, "tz": tz, "elev_m": elev}
    # Annual degree days (base 18.33 C = 65 F, matching RECS and BuildingDetective)
    Td = W["T_out"].resample("D").mean().to_numpy()
    hdd = float(np.maximum(18.33 - Td, 0.0).sum())
    cdd = float(np.maximum(Td - 18.33, 0.0).sum())
    _WX_CACHE[cz_k] = (W, meta, hdd, cdd)
    return _WX_CACHE[cz_k]


def _feat_annual(elec: float, fuel: float, hdd: float, cdd: float, dheat: float = 0.0, dcool: float = 0.0) -> np.ndarray:
    f = [hdd / 3000.0, cdd / 1500.0, np.log10(hdd + 50.0), np.log10(cdd + 50.0)]
    for e in (elec, fuel, dheat, dcool):
        if e is None or not np.isfinite(e) or e <= 0:
            f.extend([0.0, 0.0])
        else:
            f.extend([1.0, float(np.log10(e) - 5.0)])
    return np.asarray(f, dtype=np.float32)


def _stratum(hdd: float, has_fuel: bool) -> str:
    band = "cold" if hdd >= 2800.0 else ("mixed" if hdd >= 1400.0 else "warm")
    return f"{band}_{'fuel' if has_fuel else 'elec'}"


def _sim_one_worker(task: tuple) -> tuple | None:
    os.environ["OMP_NUM_THREADS"] = "1"
    from buildingsim import simulate
    spec, cz_k, seed, quirk = task
    try:
        W, meta, hdd, cdd = _load_wx(cz_k)
        rng = np.random.default_rng(seed)
        out = simulate(spec, (W, meta), "2020-01-01", "2020-12-31", dt="1h", rng=rng)
        km = out["summary"]["kWh_by_meter"]
        elec = float(km.get("electricity", 0.0))
        fuel = float(km.get("gas", 0.0))
        if quirk:
            # BuildingDetective quirk_annual billing noise
            if elec > 0:
                elec *= float(np.exp(rng.normal(0.0, rng.uniform(0.03, 0.15))))
            if fuel > 0:
                if rng.random() < 0.05:
                    fuel = 0.0
                else:
                    fuel *= float(np.exp(rng.normal(0.0, rng.uniform(0.03, 0.15))))
        x = _feat_annual(elec, fuel, hdd, cdd)
        y = float(np.log10(spec["floor_area_m2"]))
        return (x, y, _stratum(hdd, fuel > 0), elec, fuel, hdd, cdd)
    except Exception:
        return None


def _random_res_spec(rng: np.random.Generator, idx: int) -> tuple[dict, str]:
    cz = CZ_KEYS[int(rng.integers(0, len(CZ_KEYS)))]
    use = "residential_single" if rng.random() < 0.78 else "residential_multi"
    if use == "residential_single":
        area = float(10 ** rng.uniform(np.log10(55.0), np.log10(520.0)))
        n_floors = int(rng.choice([1, 2, 3], p=[0.48, 0.44, 0.08]))
    else:
        area = float(10 ** rng.uniform(np.log10(38.0), np.log10(220.0)))
        n_floors = 1
    yb = int(rng.integers(1920, 2021))
    sys_choices = ["split_ac_furnace", "heat_pump_air", "baseboard_resistance", "radiant"]
    sys_probs = [0.54, 0.22, 0.14, 0.10]
    system = str(rng.choice(sys_choices, p=sys_probs))
    # Vary realistic occupancy / plug / setpoint spread across US residential stock
    spec = dict(
        id=f"sim_{idx:05d}",
        primary_use=use,
        floor_area_m2=area,
        n_floors=n_floors,
        year_built=yb,
        climate_zone=cz,
        system=system,
        setpoints=dict(
            heat=float(rng.uniform(18.5, 22.5)),
            cool=float(rng.uniform(22.0, 26.5)),
            setback_heat=float(rng.uniform(16.0, 20.0)),
            schedule="night_setback" if rng.random() < 0.6 else "constant",
        ),
    )
    return spec, cz


def build_sim_banks(n_train: int = 1200, n_val: int = 300, workers: int = 2) -> Path:
    cache = P.work("building", "sim_banks.npz", mkdir=True)
    if cache.exists():
        return cache
    print(f"[buildingdetective] Simulating {n_train} train + {n_val} val homes via BuildingSim ({workers} workers, resumable)...", flush=True)
    rng_tr = np.random.default_rng(20261124)
    rng_va = np.random.default_rng(20261125)
    tasks_tr = [(*_random_res_spec(rng_tr, i), 10000 + i, True) for i in range(n_train)]
    tasks_va = [(*_random_res_spec(rng_va, i), 50000 + i, True) for i in range(n_val)]

    def _run_chunked(tag: str, tasks: list, chunk_sz: int = 150):
        cdir = P.work("building", f"chunks_{tag}")
        cdir.mkdir(parents=True, exist_ok=True)
        xs, ys, sts = [], [], []
        n_chunks = (len(tasks) + chunk_sz - 1) // chunk_sz
        for ci in range(n_chunks):
            cfp = cdir / f"c{ci:03d}.npz"
            if not cfp.exists():
                sub = tasks[ci * chunk_sz : (ci + 1) * chunk_sz]
                with cf.ProcessPoolExecutor(max_workers=workers) as ex:
                    res = [r for r in ex.map(_sim_one_worker, sub, chunksize=6) if r is not None]
                np.savez_compressed(
                    cfp,
                    X=np.stack([r[0] for r in res]).astype(np.float32),
                    y=np.array([r[1] for r in res], dtype=np.float64),
                    st=np.array([r[2] for r in res], dtype=str),
                )
                print(f"[buildingdetective] {tag} chunk {ci + 1}/{n_chunks}: {len(res)} homes saved", flush=True)
            zc = np.load(cfp)
            xs.append(zc["X"]); ys.append(zc["y"]); sts.append(zc["st"])
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(sts)

    X_tr, y_tr, st_tr = _run_chunked("train", tasks_tr, chunk_sz=150)
    X_va, y_va, st_va = _run_chunked("val", tasks_va, chunk_sz=150)

    np.savez_compressed(cache, X_tr=X_tr, y_tr=y_tr, st_tr=st_tr, X_va=X_va, y_va=y_va, st_va=st_va)
    print(f"[buildingdetective] Saved sim banks: train={len(X_tr)}, val={len(X_va)} -> {cache}", flush=True)
    return cache


def _load_recs_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    broot = P.REPO / "shared" / "building" / "data" / "proc"
    splits = json.loads((broot / "splits.json").read_text())["recs2020"]
    want = set(splits[split])
    df = pd.read_parquet(broot / "eia" / "recs2020.parquet")
    df = df[df["id"].isin(want)].copy().reset_index(drop=True)

    X_list, y_list, st_list, uid_list, keep_idx = [], [], [], [], []
    for idx, row in df.iterrows():
        area = float(row["floor_area_m2"])
        elec = float(row["elec_kWh"]) if np.isfinite(row["elec_kWh"]) else 0.0
        fuel = sum(
            float(row[c]) for c in ("gas_kWh", "propane_kWh", "fueloil_kWh", "wood_kWh")
            if np.isfinite(row[c]) and row[c] > 0
        )
        hdd = float(row["hdd18C_year"])
        cdd = float(row["cdd18C_year"])
        if not (np.isfinite(area) and area > 10.0 and np.isfinite(hdd) and np.isfinite(cdd) and (elec > 0 or fuel > 0)):
            continue
        x = _feat_annual(elec, fuel, hdd, cdd)
        if not np.all(np.isfinite(x)):
            continue
        X_list.append(x)
        y_list.append(np.log10(area))
        st_list.append(_stratum(hdd, fuel > 0))
        uid_list.append(str(row["id"]))
        keep_idx.append(idx)

    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list, dtype=np.float64),
        np.array(st_list, dtype=str),
        np.array(uid_list, dtype=str),
        df.loc[keep_idx].reset_index(drop=True),
    )


def _recs_row_to_spec(row: pd.Series) -> tuple[dict, str] | None:
    area = float(row["floor_area_m2"])
    if not (np.isfinite(area) and 20.0 <= area <= 1500.0):
        return None
    cz = CZ_MAP.get(str(row["climate_zone"]), "4A")
    use = str(row["primary_use"]) if pd.notna(row["primary_use"]) else "residential_single"
    if use not in ("residential_single", "residential_multi"):
        use = "residential_single"
    nfl = int(row["n_floors"]) if pd.notna(row["n_floors"]) and row["n_floors"] >= 1 else 1
    nfl = min(nfl, 3) if use == "residential_single" else 1
    ylo = float(row["year_built_lo"]) if pd.notna(row["year_built_lo"]) else 1975.0
    yhi = float(row["year_built_hi"]) if pd.notna(row["year_built_hi"]) else 1985.0
    yb = int(round(0.5 * (ylo + yhi)))
    sys_name = str(row["system"]) if pd.notna(row["system"]) else "split_ac_furnace"
    if sys_name not in ("split_ac_furnace", "heat_pump_air", "heat_pump_ground", "baseboard_resistance", "radiant", "none"):
        sys_name = "split_ac_furnace" if row["heating_fuel"] in ("natural_gas", "propane", "fuel_oil") else "baseboard_resistance"
    th = float(row["T_heat_home_C"]) if pd.notna(row["T_heat_home_C"]) else 20.5
    tc = float(row["T_cool_home_C"]) if pd.notna(row["T_cool_home_C"]) else 24.0
    th = float(np.clip(th, 15.0, 26.0))
    tc = float(np.clip(tc, max(th + 1.5, 20.0), 30.0))
    spec = dict(
        id=str(row["id"]),
        primary_use=use,
        floor_area_m2=area,
        n_floors=nfl,
        year_built=yb,
        climate_zone=cz,
        system=sys_name,
        setpoints=dict(heat=th, cool=tc, setback_heat=max(15.0, th - 2.0), schedule="night_setback"),
    )
    return spec, cz


def build_dev_twins(n_twins: int = 300, workers: int = 2) -> Path:
    cache = P.work("building", "dev_twins.npz", mkdir=True)
    if cache.exists():
        return cache
    X_r, y_r, st_r, uid_r, df_r = _load_recs_split("dev")
    rng = np.random.default_rng(20261124)
    order = rng.permutation(len(df_r))
    tasks, chosen = [], []
    for i in order:
        sp = _recs_row_to_spec(df_r.iloc[i])
        if sp is None:
            continue
        spec, cz = sp
        tasks.append((spec, cz, 90000 + int(i), False))
        chosen.append(int(i))
        if len(tasks) >= n_twins:
            break
    print(f"[buildingdetective] Simulating {len(tasks)} BuildingSim twins for real RECS dev homes ({workers} workers)...", flush=True)
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(_sim_one_worker, tasks, chunksize=8))

    Xr_out, Xt_out, yr_out, yt_out, st_out, uid_out = [], [], [], [], [], []
    for idx_i, res in zip(chosen, outs):
        if res is None:
            continue
        xt, yt, _, _, _, _, _ = res
        Xr_out.append(X_r[idx_i])
        Xt_out.append(xt)
        yr_out.append(y_r[idx_i])
        yt_out.append(yt)
        st_out.append(st_r[idx_i])
        uid_out.append(uid_r[idx_i])

    np.savez_compressed(
        cache,
        X_real=np.stack(Xr_out).astype(np.float32),
        X_twin=np.stack(Xt_out).astype(np.float32),
        y_real=np.array(yr_out, dtype=np.float64),
        y_twin=np.array(yt_out, dtype=np.float64),
        strata=np.array(st_out, dtype=str),
        unit_ids=np.array(uid_out, dtype=str),
    )
    print(f"[buildingdetective] Saved {len(Xr_out)} paired RECS dev twins -> {cache}", flush=True)
    return cache


def sim_train_val() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(build_sim_banks())
    return z["X_tr"], z["y_tr"], z["st_tr"], z["X_va"], z["y_va"], z["st_va"]


def real_dev() -> dict:
    X, y, st, uid, _ = _load_recs_split("dev")
    tz = np.load(build_dev_twins())
    return {
        "X": X, "y": y, "strata": st, "unit_ids": uid,
        "twin_X_real": tz["X_real"], "twin_X_sim": tz["X_twin"],
        "twin_y_real": tz["y_real"], "twin_y_sim": tz["y_twin"],
        "twin_strata": tz["strata"], "twin_unit_ids": tz["unit_ids"],
    }


def real_exam() -> dict:
    X, y, st, uid, _ = _load_recs_split("exam")
    return {"X": X, "y": y, "strata": st, "unit_ids": uid}


@functools.lru_cache(maxsize=1)
def _incumbent_net():
    from buildingdetective import Net
    return Net(str(P.REPO / "models" / "buildingdetective" / "weights" / "annual.npz"))


def incumbent_predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_log10A, raw_unwidened_sigma_log10A) from released BuildingDetective v1.0."""
    net = _incumbent_net()
    mu, sg_widened, _ = net.raw(X)
    raw_sg = sg_widened[:, 0] / float(net.widen[0])
    return mu[:, 0].astype(np.float64), raw_sg.astype(np.float64)


def headline_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Floor-area MdAPE (%) on linear m2 (lower is better)."""
    yt = 10.0 ** np.asarray(y_true, dtype=np.float64)
    yp = 10.0 ** np.asarray(y_pred, dtype=np.float64)
    return float(100.0 * np.median(np.abs(yp - yt) / yt))


def metric_details(y_true: np.ndarray, y_pred: np.ndarray, strata: np.ndarray) -> dict:
    yt = 10.0 ** np.asarray(y_true, dtype=np.float64)
    yp = 10.0 ** np.asarray(y_pred, dtype=np.float64)
    ape = 100.0 * np.abs(yp - yt) / yt
    out = {
        "floor_area_mdape_pct": round(float(np.median(ape)), 2),
        "log10_mae": round(float(np.mean(np.abs(y_pred - y_true))), 4),
        "strata": {},
    }
    for g in sorted(set(strata.tolist())):
        m = strata == g
        out["strata"][g] = {
            "n": int(np.sum(m)),
            "mdape_pct": round(float(np.median(ape[m])), 2),
            "log10_mae": round(float(np.mean(np.abs(y_pred[m] - y_true[m]))), 4),
        }
    return out


if __name__ == "__main__":
    build_sim_banks()
    build_dev_twins()
