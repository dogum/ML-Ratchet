"""Load real full-scale bridge monitoring datasets into unified BridgeSense episode format.

Three real bridges:
  1. KW51 railway bridge (Leuven, Belgium; steel bowstring arch, 14 modes, 15 months)
     Zenodo 3745914 (Maes & Lombaert 2020, CC BY-NC-SA 4.0)
  2. Concrete box-girder bridge (Hamburg, Germany; 2 modes, 4 months + static load tests)
     Zenodo 20522191 (Koehncke et al. 2026, CC BY 4.0) + NASA POWER hourly T2M
  3. Hell Bridge Test Arena (HBTA, Norway; 35 m steel truss, undamaged vs DS1..DS8 imposed damage)
     Zenodo 14632942 (Svendsen et al. 2025, CC BY 4.0)

Zero real labels ever enter training.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import numpy as np
import scipy.io
import h5py

from _paths import work

N_REF = 32
N_Q = 32
MAX_MODES = 14


def _matlab_datenum_to_dt(sdn: float) -> datetime:
    return datetime.fromordinal(int(sdn)) + timedelta(days=float(sdn) % 1.0) - timedelta(days=366)


def _rolling_mean_nan(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean over window w ignoring NaNs."""
    out = np.empty_like(x, dtype=np.float64)
    n = len(x)
    valid = np.isfinite(x).astype(np.float64)
    vals = np.where(np.isfinite(x), x, 0.0)
    cs_v = np.concatenate([[0.0], np.cumsum(vals)])
    cs_c = np.concatenate([[0.0], np.cumsum(valid)])
    idx = np.arange(n)
    lo = np.maximum(0, idx + 1 - w)
    cnt = cs_c[idx + 1] - cs_c[lo]
    s = cs_v[idx + 1] - cs_v[lo]
    out = np.where(cnt > 0, s / np.maximum(cnt, 1.0), np.nan)
    return out


def _interp_nans(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    ok = np.isfinite(x)
    if not np.any(ok):
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def load_kw51_hourly() -> dict:
    """Load KW51 trackedmodes.mat as aligned hourly arrays."""
    mat = scipy.io.loadmat(work("raw", "kw51", "trackedmodes.mat"), squeeze_me=True, struct_as_record=False)
    modes = mat["modes"]
    sdn = np.asarray(modes.sdn, dtype=np.float64)
    dts = [_matlab_datenum_to_dt(d) for d in sdn]
    f = np.asarray(modes.f, dtype=np.float64)       # [N, 14]
    env = np.asarray(modes.env, dtype=np.float64)   # [N, 11]
    # Col 2 is air temp at Vliet building (tVL, 99.5% complete); fallback to steel surface temp (col 0)
    T_air = env[:, 2].copy()
    miss_t = ~np.isfinite(T_air)
    T_air[miss_t] = env[miss_t, 0]
    T_air = _interp_nans(T_air)
    T24 = _rolling_mean_nan(T_air, 24)
    T72 = _rolling_mean_nan(T_air, 72)

    # Published KW51 phase dates (Maes & Lombaert 2020):
    # Pre-retrofit: < 2019-05-15; During retrofit: 2019-05-15 .. 2019-09-27; Post-retrofit: >= 2019-09-27
    t_ret_start = datetime(2019, 5, 15, 0, 0, 0)
    t_ret_end = datetime(2019, 9, 27, 0, 0, 0)
    phase = []
    for dt in dts:
        if dt < t_ret_start:
            phase.append("pre_healthy")
        elif dt < t_ret_end:
            phase.append("during_retrofit")
        else:
            phase.append("post_retrofit")

    return {
        "dts": dts,
        "f": f,
        "T_air": T_air,
        "T24": T24,
        "T72": T72,
        "T_steel": env[:, 0],
        "phase": phase,
        "mode_names": [f"M{i+1}" for i in range(14)],
    }


def build_window_episode(
    T_ref_3: np.ndarray,
    f_ref: np.ndarray,
    T_q_3: np.ndarray,
    f_q: np.ndarray,
    n_ref: int = N_REF,
    n_q: int = N_Q,
    max_modes: int = MAX_MODES,
    seed: int = 0,
    meta: dict | None = None,
) -> dict:
    """Pack raw reference & query temperature/frequency tables into fixed-shape BridgeSense inputs.

    Subsamples evenly (deterministic) to n_ref and n_q rows, computes fractional shift relative to
    the reference window median per mode, and pads mode axis to max_modes.
    """
    M = f_ref.shape[1]
    assert M <= max_modes

    def _subsample(n_tot: int, n_out: int) -> np.ndarray:
        if n_tot >= n_out:
            return np.linspace(0, n_tot - 1, n_out).round().astype(int)
        # Repeat evenly if shorter (e.g. short HBTA damage-state test runs)
        return np.linspace(0, n_tot - 1, n_out).round().astype(int)

    ir = _subsample(len(T_ref_3), n_ref)
    iq = _subsample(len(T_q_3), n_q)

    Tr = T_ref_3[ir].astype(np.float32)
    Tq = T_q_3[iq].astype(np.float32)
    fr = f_ref[ir]
    fq = f_q[iq]

    df_ref = np.zeros((n_ref, max_modes), dtype=np.float32)
    df_q = np.zeros((n_q, max_modes), dtype=np.float32)
    mask_ref = np.zeros((n_ref, max_modes), dtype=np.float32)
    mask_q = np.zeros((n_q, max_modes), dtype=np.float32)
    f0 = np.full(max_modes, np.nan, dtype=np.float64)

    for m in range(M):
        ok_r = np.isfinite(fr[:, m]) & (fr[:, m] > 0)
        ok_q = np.isfinite(fq[:, m]) & (fq[:, m] > 0)
        if np.sum(ok_r) < 3 or np.sum(ok_q) < 2:
            continue
        med_m = float(np.median(fr[ok_r, m]))
        f0[m] = med_m
        df_ref[ok_r, m] = ((fr[ok_r, m] - med_m) / med_m).astype(np.float32)
        df_q[ok_q, m] = ((fq[ok_q, m] - med_m) / med_m).astype(np.float32)
        mask_ref[ok_r, m] = 1.0
        mask_q[ok_q, m] = 1.0

    ep = {
        "T_ref": Tr,
        "T_q": Tq,
        "df_ref": df_ref,
        "df_q": df_q,
        "mask_ref": mask_ref,
        "mask_q": mask_q,
        "n_modes": int(np.sum(np.isfinite(f0))),
        "f0": f0,
    }
    if meta:
        ep.update(meta)
    return ep


def build_kw51_blocks(kw: dict | None = None, block_days: int = 7) -> tuple[dict, list[dict]]:
    """Build KW51 warm reference baseline (Oct 1 - Nov 11, 2018) and non-overlapping 7-day blocks."""
    if kw is None:
        kw = load_kw51_hourly()
    dts = kw["dts"]
    T3 = np.stack([kw["T_air"], kw["T24"], kw["T72"]], axis=-1)
    f = kw["f"]

    # Common warm-only healthy baseline: 2018-10-01 .. 2018-11-11 (6 weeks, pre-winter, pre-retrofit)
    t_base_end = datetime(2018, 11, 12, 0, 0, 0)
    base_idx = [i for i, dt in enumerate(dts) if dt < t_base_end]
    ref_pack = {
        "idx": np.array(base_idx, dtype=int),
        "T3": T3[base_idx],
        "f": f[base_idx],
        "start": dts[base_idx[0]].isoformat(),
        "end": dts[base_idx[-1]].isoformat(),
    }

    # Build 7-day blocks starting 2018-11-12
    blocks = []
    cur = t_base_end
    t_end = dts[-1]
    bid = 0
    t_ret_start = datetime(2019, 5, 15, 0, 0, 0)
    t_ret_end = datetime(2019, 9, 27, 0, 0, 0)

    # Also compute true matched-temperature post-retrofit shift per mode ONLY as an independent
    # physics/catalogue ground truth reference (for exam grading of dK/K on post-retrofit!)
    while cur + timedelta(days=block_days) <= t_end + timedelta(hours=12):
        nxt = cur + timedelta(days=block_days)
        idx = [i for i, dt in enumerate(dts) if cur <= dt < nxt]
        if len(idx) >= 48:
            # Skip boundary-straddling blocks that mix phases
            if cur < t_ret_start < nxt or cur < t_ret_end < nxt:
                cur = nxt
                continue
            ph = kw["phase"][idx[0]]
            t_min = float(np.min(kw["T_air"][idx]))
            t_mean = float(np.mean(kw["T_air"][idx]))
            # Valid observations check
            n_valid_modes = int(np.sum(np.sum(np.isfinite(f[idx]), axis=0) >= 6))
            if n_valid_modes >= 4:
                is_freezing = bool(t_min < 0.0)
                if ph == "pre_healthy":
                    stratum = "healthy_freezing" if is_freezing else "healthy_mild"
                    is_changed = 0
                elif ph == "during_retrofit":
                    stratum = "during_retrofit"
                    is_changed = 1
                else:
                    stratum = "post_retrofit"
                    is_changed = 1
                ep = build_window_episode(
                    ref_pack["T3"], ref_pack["f"], T3[idx], f[idx],
                    meta={
                        "block_id": f"kw51_b{bid:02d}",
                        "bridge": "kw51",
                        "start": cur.strftime("%Y-%m-%d"),
                        "end": nxt.strftime("%Y-%m-%d"),
                        "phase": ph,
                        "stratum": stratum,
                        "is_changed": is_changed,
                        "T_min": round(t_min, 2),
                        "T_mean": round(t_mean, 2),
                        "is_freezing": is_freezing,
                    },
                )
                blocks.append(ep)
                bid += 1
        cur = nxt
    return ref_pack, blocks


def load_boxgirder_episodes(block_days: int = 5) -> dict:
    """Load Concrete Box-Girder Bridge (Zenodo 20522191, CC BY 4.0) + NASA POWER Hamburg T2M.

    Table 3 (Koehncke et al. 2026, arXiv:2412.15713):
      - Healthy unloaded: 2024-01-01 to 2024-02-22 09:58 UTC (includes Jan 2024 freezing)
      - Long-term static load test (sand bags 680 / 1,420 / 2,160 kg): 2024-02-22 09:58 to 2024-04-21
    """
    h5_path = work("raw", "boxgirder", "box_girder_bridge_eigenfrequencies.h5")
    wx_path = work("raw", "boxgirder", "nasa_power_hamburg_2024.json")
    with h5py.File(h5_path, "r") as hf:
        ts_raw = [t.decode() if isinstance(t, bytes) else str(t) for t in hf["bridge/F1/timestamp"][:]]
        f1 = np.asarray(hf["bridge/F1/freq"][:], dtype=np.float64)
        m1 = np.asarray(hf["bridge/F1/freq_missing"][:], dtype=np.uint8)
        f2 = np.asarray(hf["bridge/F2/freq"][:], dtype=np.float64)
        m2 = np.asarray(hf["bridge/F2/freq_missing"][:], dtype=np.uint8)
    f1[m1 != 0] = np.nan
    f2[m2 != 0] = np.nan
    # Remove extreme OMA outliers (> 25% away from median)
    for arr in (f1, f2):
        med = np.nanmedian(arr)
        arr[np.abs(arr - med) / med > 0.25] = np.nan
    dts = [datetime.fromisoformat(s).astimezone(timezone.utc).replace(tzinfo=None) for s in ts_raw]

    wx = json.load(open(wx_path))["properties"]["parameter"]["T2M"]
    # Build hourly timeline from 2023-12-25 00:00 UTC
    wx_keys = sorted(wx.keys())
    wx_t0 = datetime.strptime(wx_keys[0], "%Y%m%d%H")
    wx_vals = np.array([wx[k] if wx[k] > -90 else np.nan for k in wx_keys], dtype=np.float64)
    wx_vals = _interp_nans(wx_vals)
    wx_t24 = _rolling_mean_nan(wx_vals, 24)
    wx_t72 = _rolling_mean_nan(wx_vals, 72)

    # Interpolate hourly weather onto bridge timestamps
    hours_from_t0 = np.array([(dt - wx_t0).total_seconds() / 3600.0 for dt in dts], dtype=np.float64)
    wx_h = np.arange(len(wx_vals), dtype=np.float64)
    T_air = np.interp(hours_from_t0, wx_h, wx_vals)
    T24 = np.interp(hours_from_t0, wx_h, wx_t24)
    T72 = np.interp(hours_from_t0, wx_h, wx_t72)
    T3 = np.stack([T_air, T24, T72], axis=-1)
    f = np.stack([f1, f2], axis=-1)

    # Reference healthy baseline: 2024-01-25 to 2024-02-21 (4 weeks of mild/cool unloaded winter-spring)
    # The baseline is the mild, unloaded period (Jan 25 - Feb 21), so the Jan 1 - Jan 24 freezing weeks stay outside it
    # and serve as unseen cold-snap false-alarm tests on a concrete bridge.
    t_ref_start = datetime(2024, 1, 25, 0, 0, 0)
    t_load_start = datetime(2024, 2, 22, 9, 58, 0)
    ref_idx = [i for i, dt in enumerate(dts) if t_ref_start <= dt < t_load_start]

    # Define Table 3 load test phases
    load_phases = [
        ("healthy_jan_cold", datetime(2024, 1, 1), datetime(2024, 1, 25), 0, 0),
        ("load_mid_light_680kg", datetime(2024, 2, 22, 10), datetime(2024, 3, 3, 7), 1, 680),
        ("load_mid_med_1420kg", datetime(2024, 3, 3, 8), datetime(2024, 3, 13, 9), 1, 1420),
        ("load_mid_heavy_2160kg", datetime(2024, 3, 13, 10), datetime(2024, 3, 23, 9), 1, 2160),
        ("load_qtr_light_680kg", datetime(2024, 3, 23, 10), datetime(2024, 4, 2, 7), 1, 680),
        ("load_qtr_med_1420kg", datetime(2024, 4, 2, 8), datetime(2024, 4, 12, 7), 1, 1420),
        ("load_qtr_heavy_2160kg", datetime(2024, 4, 12, 8), datetime(2024, 4, 21, 23), 1, 2160),
    ]

    episodes = []
    bid = 0
    for pname, ps, pe, is_chg, mass_kg in load_phases:
        cur = ps
        while cur + timedelta(days=block_days) <= pe + timedelta(hours=12):
            nxt = min(pe, cur + timedelta(days=block_days))
            idx = [i for i, dt in enumerate(dts) if cur <= dt < nxt]
            if len(idx) >= 48:
                t_min = float(np.min(T_air[idx]))
                t_mean = float(np.mean(T_air[idx]))
                ep = build_window_episode(
                    T3[ref_idx], f[ref_idx], T3[idx], f[idx],
                    meta={
                        "block_id": f"boxgirder_b{bid:02d}",
                        "bridge": "boxgirder",
                        "start": cur.strftime("%Y-%m-%d"),
                        "end": nxt.strftime("%Y-%m-%d"),
                        "phase": pname,
                        "stratum": "healthy_freezing" if (is_chg == 0 and t_min < 0.0) else ("healthy_mild" if is_chg == 0 else "static_load"),
                        "is_changed": is_chg,
                        "added_mass_kg": mass_kg,
                        "T_min": round(t_min, 2),
                        "T_mean": round(t_mean, 2),
                        "is_freezing": bool(t_min < 0.0),
                    },
                )
                episodes.append(ep)
                bid += 1
            cur = cur + timedelta(days=block_days)

    return {"ref_T3": T3[ref_idx], "ref_f": f[ref_idx], "episodes": episodes, "T_air": T_air, "f": f}
