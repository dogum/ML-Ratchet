"""Parse raw Kelmarsh and Penmanshiel Zenodo SCADA archives into clean 10-minute farm snapshots.

Reads only from work/raw/ and writes processed .npz archives to work/proc/ via _paths.py.
Every filter uses operating status, local sensor validity, and geometric front-row inflow checks —
never downstream wake losses. Exam archives are written blindly and never inspected before exam.py.
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402
import physics as WP  # noqa: E402
import pipeline as PL  # noqa: E402

SCADA_COLS = [
    "# Date and time",
    "Wind speed (m/s)",
    "Wind speed, Standard deviation (m/s)",
    "Wind direction (°)",
    "Wind direction, Standard deviation (°)",
    "Nacelle position (°)",
    "Power (kW)",
    "Potential power default PC (kW)",
    "Potential power reference turbines (kW)",
    "Turbine Power setpoint (kW)",
    "Blade angle (pitch position) A (°)",
    "Rotor speed (RPM)",
    "Nacelle ambient temperature (°C)",
]


def load_static_farm(farm: str) -> dict:
    raw = Path(_P.work("raw"))
    if farm == "kelmarsh":
        df = pd.read_csv(raw / "Kelmarsh_WT_static.csv").dropna(subset=["Latitude", "Longitude"])
        turb = WP.TURBINES["MM92"]
    elif farm == "penmanshiel":
        df = pd.read_csv(raw / "Penmanshiel_WT_static.csv").dropna(subset=["Latitude", "Longitude"])
        turb = WP.TURBINES["MM82"]
    else:
        raise ValueError(farm)
    lats = df["Latitude"].to_numpy(dtype=np.float64)
    lons = df["Longitude"].to_numpy(dtype=np.float64)
    east, north = PL.latlon_to_east_north(lats, lons)
    return {
        "farm": farm,
        "titles": df["Title"].astype(str).tolist(),
        "codes": df["Alternative Title"].astype(str).tolist(),
        "east_m": east,
        "north_m": north,
        "elev_m": df["Elevation (m)"].to_numpy(dtype=np.float64),
        "hub_h_m": df["Hub Height (m)"].to_numpy(dtype=np.float64),
        "rotor_d_m": df["Rotor Diameter (m)"].to_numpy(dtype=np.float64),
        "rated_kw": float(df["Rated power (kW)"].iloc[0]),
        "lat_deg": lats,
        "lon_deg": lons,
        "turbine": turb,
    }


def _read_year_frames(farm: str, year: int, titles: list[str]) -> list[pd.DataFrame]:
    raw = Path(_P.work("raw"))
    if farm == "kelmarsh":
        zips = sorted(raw.glob(f"Kelmarsh_SCADA_{year}_*.zip"))
    else:
        zips = sorted(raw.glob(f"Penmanshiel_SCADA_{year}_*.zip"))
    if not zips:
        raise FileNotFoundError(f"Missing raw SCADA zips for {farm} {year} in {raw}")

    per_wt: dict[str, pd.DataFrame] = {}
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if "Turbine_Data_" not in name or not name.endswith(".csv"):
                    continue
                for title in titles:
                    # "Kelmarsh 1" -> "Turbine_Data_Kelmarsh_1_", "Penmanshiel 01" -> "Turbine_Data_Penmanshiel_01_"
                    tag = "Turbine_Data_" + title.replace(" ", "_") + "_"
                    if tag in name:
                        with zf.open(name) as fh:
                            d = pd.read_csv(fh, skiprows=9, usecols=SCADA_COLS)
                        d = d.rename(columns={"# Date and time": "ts"})
                        per_wt[title] = d.drop_duplicates(subset=["ts"]).set_index("ts").sort_index()
                        break
    if len(per_wt) != len(titles):
        missing = [t for t in titles if t not in per_wt]
        raise RuntimeError(f"{farm} {year}: missing turbine CSVs for {missing}")
    return [per_wt[t] for t in titles]


def build_snapshots(farm: str, years: list[int], out_name: str) -> dict:
    meta = load_static_farm(farm)
    titles = meta["titles"]
    n_wt = len(titles)
    p_rated = meta["rated_kw"]
    east, north = meta["east_m"], meta["north_m"]
    elev, hub_h = meta["elev_m"], meta["hub_h_m"]
    D = float(meta["turbine"]["rotor_diameter"])
    ws_tab = meta["turbine"]["wind_speed"]
    p_tab = meta["turbine"]["power_kw"]

    chunks = []
    for yr in years:
        frames = _read_year_frames(farm, yr, titles)
        common_ts = frames[0].index
        for fr in frames[1:]:
            common_ts = common_ts.intersection(fr.index)
        common_ts = common_ts.sort_values()
        T = len(common_ts)
        if T == 0:
            continue

        def _stack(col: str) -> np.ndarray:
            arr = np.empty((T, n_wt), dtype=np.float64)
            for i, fr in enumerate(frames):
                arr[:, i] = fr.loc[common_ts, col].to_numpy(dtype=np.float64)
            return arr

        u = _stack("Wind speed (m/s)")
        u_sd = _stack("Wind speed, Standard deviation (m/s)")
        wd = _stack("Wind direction (°)")
        wd_sd = _stack("Wind direction, Standard deviation (°)")
        nac = _stack("Nacelle position (°)")
        p_kw = _stack("Power (kW)")
        p_def = _stack("Potential power default PC (kW)")
        p_ref_op = _stack("Potential power reference turbines (kW)")
        sp = _stack("Turbine Power setpoint (kW)")
        pitch = _stack("Blade angle (pitch position) A (°)")
        rpm = _stack("Rotor speed (RPM)")
        t_amb = _stack("Nacelle ambient temperature (°C)")

        yaw_local = WP.wrap_deg(wd - nac)

        # Operating classification per turbine (using only local turbine status, never wake ratios)
        parked = (
            np.isfinite(p_kw)
            & (p_kw <= 5.0)
            & ((p_kw <= 1.0) | (pitch > 35.0) | (rpm < 0.5))
        )
        uncurtailed_sp = np.isnan(sp) | (sp >= 0.975 * p_rated)
        uncurtailed_pitch = (p_kw >= 0.88 * p_rated) | (pitch < 4.5)
        local_pc_ok = np.where(
            np.isfinite(p_def) & (p_def >= 80.0),
            (p_kw >= 0.50 * p_def) & (p_kw <= 1.45 * p_def),
            True,
        )
        normal = (
            np.isfinite(u)
            & np.isfinite(u_sd)
            & np.isfinite(wd)
            & np.isfinite(wd_sd)
            & np.isfinite(nac)
            & np.isfinite(p_kw)
            & np.isfinite(pitch)
            & (p_kw > 20.0)
            & (u >= 3.5)
            & uncurtailed_sp
            & uncurtailed_pitch
            & local_pc_ok
            & (np.abs(yaw_local) <= 40.0)
        )

        # Every turbine in the farm must be unambiguously NORMAL or PARKED (no curtailed/faulted unknown wake casters)
        clean_row = np.all(normal | parked, axis=1) & (np.sum(normal, axis=1) >= max(n_wt - 2, 4))
        idx_candidates = np.where(clean_row)[0]

        kept_ts = []
        kept_u_inf, kept_wd_inf, kept_ti_inf, kept_swd_inf, kept_rho, kept_p_front = [], [], [], [], [], []
        kept_online, kept_free, kept_yaw, kept_p, kept_u, kept_pref_op = [], [], [], [], [], []

        h_ref = float(np.max(hub_h))  # reference hub height (78.5 m on Kelmarsh, 59.0 m on Penmanshiel)
        shear_norm = (h_ref / hub_h) ** 0.14

        for r in idx_candidates:
            on_r = normal[r]
            # Initial circular mean wind direction over online units, then refine once on geometric front row
            ang0 = np.deg2rad(wd[r, on_r])
            wd0 = float((np.degrees(np.arctan2(np.mean(np.sin(ang0)), np.mean(np.cos(ang0)))) + 360.0) % 360.0)
            free_r = WP.freestream_mask(east, north, D, wd0, online=on_r)
            if int(np.sum(free_r)) < 2:
                continue
            ang_f = np.deg2rad(wd[r, free_r])
            wd_inf = float((np.degrees(np.arctan2(np.mean(np.sin(ang_f)), np.mean(np.cos(ang_f)))) + 360.0) % 360.0)
            free_r = WP.freestream_mask(east, north, D, wd_inf, online=on_r)
            n_free = int(np.sum(free_r))
            n_waked = int(np.sum(on_r & (~free_r)))
            if n_free < 2 or n_waked < 1:
                continue

            u_front_vec = u[r, free_r] * shear_norm[free_r]
            u_inf = float(np.mean(u_front_vec))
            if not (5.0 <= u_inf <= 12.0):
                continue

            ti_vec = u_sd[r, free_r] / np.maximum(u[r, free_r], 1.0)
            ti_inf = float(np.clip(np.median(ti_vec), 0.02, 0.35))
            swd_inf = float(np.clip(np.median(wd_sd[r, free_r]), 0.5, 25.0))

            temp_vals = t_amb[r, on_r]
            temp_c = float(np.nanmedian(temp_vals)) if np.any(np.isfinite(temp_vals)) else 10.0
            rho = float(np.clip(1.225 * (288.15 / (273.15 + temp_c)), 1.12, 1.32))

            p_front_vec = p_kw[r, free_r]
            p_front_mean = float(np.mean(p_front_vec))
            if p_front_mean < 120.0:
                continue
            # Reject storm-front gusts where unwaked front-row turbines disagree wildly
            if float(np.std(p_front_vec)) / p_front_mean > 0.28:
                continue
            p_pc_expected = float(np.interp(u_inf * (rho / 1.225) ** (1.0 / 3.0), ws_tab, p_tab))
            if p_pc_expected < 100.0 or not (0.72 <= p_front_mean / p_pc_expected <= 1.28):
                continue

            # Causal mechanical yaw offset relative to upstream freestream wind direction (uses nac[i], never downstream vane!)
            yaw_r = np.where(on_r, WP.wrap_deg(wd_inf - nac[r]), 0.0)
            if np.any(np.abs(yaw_r[on_r]) > 35.0):
                continue

            kept_ts.append(str(common_ts[r]))
            kept_u_inf.append(u_inf)
            kept_wd_inf.append(wd_inf)
            kept_ti_inf.append(ti_inf)
            kept_swd_inf.append(swd_inf)
            kept_rho.append(rho)
            kept_p_front.append(p_front_mean)
            kept_online.append(on_r)
            kept_free.append(free_r)
            kept_yaw.append(yaw_r.astype(np.float32))
            kept_p.append(np.where(on_r, p_kw[r], 0.0).astype(np.float32))
            kept_u.append(np.where(on_r, u[r], 0.0).astype(np.float32))
            kept_pref_op.append(np.where(on_r & np.isfinite(p_ref_op[r]), p_ref_op[r], np.nan).astype(np.float32))

        print(f"  {farm} {yr}: {len(kept_ts):5d} clean farm snapshots (from {T} raw 10-min stamps)", flush=True)
        if not kept_ts:
            continue

        chunks.append({
            "ts": np.array(kept_ts, dtype="U19"),
            "year": np.full(len(kept_ts), yr, dtype=np.int16),
            "u_inf": np.array(kept_u_inf, dtype=np.float32),
            "wd_inf": np.array(kept_wd_inf, dtype=np.float32),
            "ti_inf": np.array(kept_ti_inf, dtype=np.float32),
            "sigma_wd": np.array(kept_swd_inf, dtype=np.float32),
            "air_density": np.array(kept_rho, dtype=np.float32),
            "p_front_kw": np.array(kept_p_front, dtype=np.float32),
            "online": np.array(kept_online, dtype=bool),
            "is_free": np.array(kept_free, dtype=bool),
            "yaw_deg": np.array(kept_yaw, dtype=np.float32),
            "power_kw": np.array(kept_p, dtype=np.float32),
            "u_scada_ms": np.array(kept_u, dtype=np.float32),
            "p_ref_op_kw": np.array(kept_pref_op, dtype=np.float32),
        })

    merged = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    merged.update({
        "farm": np.array(farm),
        "codes": np.array(meta["codes"], dtype="U8"),
        "east_m": meta["east_m"].astype(np.float32),
        "north_m": meta["north_m"].astype(np.float32),
        "elev_m": meta["elev_m"].astype(np.float32),
        "hub_h_m": meta["hub_h_m"].astype(np.float32),
        "rotor_d_m": meta["rotor_d_m"].astype(np.float32),
        "rated_kw": np.float32(meta["rated_kw"]),
        "lat_deg": meta["lat_deg"].astype(np.float64),
        "lon_deg": meta["lon_deg"].astype(np.float64),
    })
    out_path = _P.work("proc", out_name, mkdir=True)
    np.savez_compressed(out_path, **merged)
    return {"name": out_name, "n_snapshots": int(len(merged["ts"])), "n_turbines": n_wt, "years": years}


def main():
    print("Building WakeNet quality-controlled SCADA snapshot tables in work/proc/...", flush=True)
    s_back = build_snapshots("kelmarsh", [2016, 2017, 2018], "kelmarsh_backdrop_2016_2018.npz")
    s_dev = build_snapshots("kelmarsh", [2019], "kelmarsh_dev_2019.npz")
    s_kex = build_snapshots("kelmarsh", [2020, 2021], "kelmarsh_exam_2020_2021.npz")
    s_pex = build_snapshots("penmanshiel", [2019, 2020], "penmanshiel_exam_2019_2020.npz")
    for s in (s_back, s_dev, s_kex, s_pex):
        print(" ->", s, flush=True)


if __name__ == "__main__":
    main()
