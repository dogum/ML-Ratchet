"""Held-out real exams for WakeNet.

Exam 1 (opened once on 16 Nov 2026 for v1.0; re-scored 17 Nov 2026 with more baselines, nothing chosen from it):
  1. Kelmarsh 2020-2021 (same farm, unseen years)
  2. Penmanshiel 2019-2020 (zero-shot unseen 14-turbine MM82 farm)

Exam 2 (frozen in split_exam2.json on 17 Nov 2026, never seen by any version): Kelmarsh 2022-2024 and
Penmanshiel 2021-2023. Run with `python exam.py exam2 <weights.npz>...` after the v1.1 decision.

Baselines (fixed in NOTEBOOK.md, 17 Nov 2026, before any data were re-read):
  * Classical: FLORIS 3-D Gaussian tuned on 2019 dev (calibrated_v3 in dev_calibration.json: meandering 0.25 x vane
    sigma, hill speedup 0.18, shear 0.14, terrain-follow 0.80) -- the strongest classical baseline
  * Classical: textbook NREL FLORIS 3-D Gaussian (Bastankhah-Porte-Agel + Gaussian yaw + Crespo-Hernandez + SOSFS)
  * Classical: Jensen Park + Jimenez with the onshore wake expansion 0.075 (the 0.05 offshore value is kept as a note)
  * Dumb: the Kelmarsh 2019 dev mean wake loss, and "no wake" (eta = 1)

Every row reports WakeNet minus the best baseline with a 95 % interval from a bootstrap over calendar days
(2,000 resamples, seed 20261117). Verdict: win if the interval is below zero, loss if above, tie if it spans zero.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import _paths as _P  # noqa: E402
from mlratchet.calibrate import coverage  # noqa: E402
import physics as WP  # noqa: E402
import pipeline as PL  # noqa: E402
from wakenet import WakeNet  # noqa: E402

BOOT_N = 2000
BOOT_SEED = 20261117

BASELINES = {
    # key: (label, kind)
    "floris_tuned": ("FLORIS 3-D Gaussian tuned on 2019 dev", "classical"),
    "floris_gauss": ("NREL FLORIS 3-D Gaussian", "classical"),
    "jensen075": ("Jensen Park + Jiménez (onshore k = 0.075)", "classical"),
    "dev_mean": ("Dev mean wake loss", "dumb"),
    "unwaked": ("Unwaked guess", "dumb"),
}
NOTE_ONLY = {"jensen050": "Jensen Park + Jiménez (offshore k = 0.05, the v1.0 exam's setting)"}

ROWS = [
    "all_waked", "direct_wake", "multi_wake_2plus", "large_yaw_8deg", "partial_parked",
    "region2_below_10ms", "near_rated_10p5_12ms", "hilltop_turbines",
]


def _tuned_params() -> dict:
    cal = json.load(open(os.path.join(HERE, "dev_calibration.json"), encoding="utf-8"))["calibrated_v3"]["params"]
    return cal


def _eta_from(r: dict, on: np.ndarray, free: np.ndarray) -> np.ndarray:
    den = float(np.mean(r["power_kw"][free]))
    return np.where(on, r["power_kw"] / max(den, 1.0), 0.0)


def _baseline_chunk(args) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Textbook FLORIS (exactly v1.0's anchor), tuned FLORIS and both Jensen settings for snapshots [t0, t1)."""
    proc_npz, t0, t1, cal = args
    z = np.load(_P.work("proc", proc_npz))
    farm = str(z["farm"])
    turb = WP.TURBINES["MM92" if farm == "kelmarsh" else "MM82"]
    east, north = z["east_m"].astype(np.float64), z["north_m"].astype(np.float64)
    elev, hub_h = z["elev_m"].astype(np.float64), z["hub_h_m"].astype(np.float64)
    n = east.shape[0]
    out_t = np.zeros((t1 - t0, n), np.float32)
    out_j75 = np.zeros((t1 - t0, n), np.float32)
    out_j50 = np.zeros((t1 - t0, n), np.float32)
    out_f = np.zeros((t1 - t0, n), np.float32)
    D = float(turb["rotor_diameter"])
    for k, t in enumerate(range(t0, t1)):
        on, free = z["online"][t], z["is_free"][t]
        u, wd, ti = float(z["u_inf"][t]), float(z["wd_inf"][t]), float(z["ti_inf"][t])
        swd, rho, yaw = float(z["sigma_wd"][t]), float(z["air_density"][t]), z["yaw_deg"][t]
        r = WP.solve_wake(
            east, north, hub_h, turb, u, wd, ti, yaw_deg=yaw, online=on, elev_m=elev,
            params={
                "velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez",
                "wind_shear": cal["wind_shear"], "sigma_wd_deg": cal["c_meander"] * swd,
                "hill_speedup": cal["hill_speedup"], "terrain_follow": cal["terrain_follow"], "air_density": rho,
            },
        )
        out_t[k] = _eta_from(r, on, free)
        # Textbook FLORIS, normalised exactly as pipeline.build_graph normalises v1.0's anchor
        cfg = dict(PL.ANCHOR_PARAMS, air_density=rho, ref_height=float(np.mean(hub_h[on])) if np.any(on) else float(np.mean(hub_h)))
        rf = WP.solve_wake(east, north, hub_h, turb, u, wd, ti, yaw_deg=yaw, online=on, elev_m=elev, params=cfg)
        gfree = WP.freestream_mask(east, north, D, wd, online=on)
        fidx = gfree if np.any(gfree) else on
        den = float(np.mean(rf["power_kw"][fidx])) if np.any(fidx) else 1.0
        out_f[k] = np.where(on, 1.0, 0.0) if den < 1.0 else np.where(on, rf["power_kw"] / den, 0.0)
        for we, dest in ((0.075, out_j75), (0.05, out_j50)):
            rj = WP.solve_wake(
                east, north, np.mean(hub_h), turb, u_inf=u, wind_dir_deg=wd, ti_inf=ti, yaw_deg=yaw, online=on,
                params={"velocity_model": "jensen", "deflection_model": "jimenez", "turbulence_model": "none",
                        "wind_shear": 0.0, "air_density": rho, "we": we},
            )
            dest[k] = _eta_from(rj, on, free)
    return t0, out_t, out_j75, out_j50, out_f


def _baselines(proc_npz: str, T: int, n: int, jobs: int) -> dict[str, np.ndarray]:
    cal = _tuned_params()
    step = 250
    tasks = [(proc_npz, a, min(a + step, T), cal) for a in range(0, T, step)]
    res = {k: np.zeros((T, n), np.float32) for k in ("floris_tuned", "jensen075", "jensen050", "floris_gauss")}
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        for t0, et, j75, j50, fg in ex.map(_baseline_chunk, tasks):
            t1 = t0 + et.shape[0]
            res["floris_tuned"][t0:t1], res["jensen075"][t0:t1], res["jensen050"][t0:t1] = et, j75, j50
            res["floris_gauss"][t0:t1] = fg
    return res


def _day_index(ts: np.ndarray) -> np.ndarray:
    days = np.array([str(s)[:10] for s in ts])
    _, idx = np.unique(days, return_inverse=True)
    return idx


def _boot_diff(err_m: np.ndarray, err_b: np.ndarray, day: np.ndarray, rng_seed: int) -> list[float]:
    """95 % interval of mean(err_m) - mean(err_b), resampling calendar days."""
    nd = int(day.max()) + 1
    sm = np.bincount(day, weights=err_m, minlength=nd)
    sb = np.bincount(day, weights=err_b, minlength=nd)
    cnt = np.bincount(day, minlength=nd).astype(np.float64)
    keep = cnt > 0
    sm, sb, cnt = sm[keep], sb[keep], cnt[keep]
    rng = np.random.default_rng(rng_seed)
    pick = rng.integers(0, len(cnt), size=(BOOT_N, len(cnt)))
    w = np.zeros((BOOT_N, len(cnt)))
    np.add.at(w, (np.repeat(np.arange(BOOT_N), len(cnt)), pick.ravel()), 1.0)
    d = (w @ sm) / (w @ cnt) - (w @ sb) / (w @ cnt)
    lo, hi = np.quantile(d, [0.025, 0.975])
    return [round(100.0 * float(lo), 2), round(100.0 * float(hi), 2)]


def _row(preds: dict[str, np.ndarray], truth: np.ndarray, mask: np.ndarray, day2d: np.ndarray, seed: int) -> dict:
    """preds: name -> array shaped like truth (or broadcastable); errors in % of front-row power."""
    t_v = truth[mask]
    d_v = day2d[mask]
    err = {k: np.abs(np.broadcast_to(p, truth.shape)[mask] - t_v) for k, p in preds.items()}
    row = {"n": int(mask.sum()), "n_days": int(len(np.unique(d_v)))}
    for k, e in err.items():
        row[f"{k}_pct"] = round(100.0 * float(np.mean(e)), 2)
    row["wakenet_bias_pct"] = round(100.0 * float(np.mean(np.broadcast_to(preds["wakenet"], truth.shape)[mask] - t_v)), 2)
    best = min(BASELINES, key=lambda k: float(np.mean(err[k])))
    ci = _boot_diff(err["wakenet"], err[best], d_v, seed)
    row["best_baseline"] = best
    row["diff_vs_best_pct"] = round(100.0 * float(np.mean(err["wakenet"]) - np.mean(err[best])), 2)
    row["diff_vs_best_ci95"] = ci
    row["diff_vs_tuned_pct"] = round(100.0 * float(np.mean(err["wakenet"]) - np.mean(err["floris_tuned"])), 2)
    row["diff_vs_tuned_ci95"] = _boot_diff(err["wakenet"], err["floris_tuned"], d_v, seed + 1)
    row["verdict"] = "win" if ci[1] < 0 else ("loss" if ci[0] > 0 else "tie")
    return row


def evaluate_farm_exam(nets: dict[str, WakeNet], proc_npz: str, jobs: int = 12) -> dict:
    z = np.load(_P.work("proc", proc_npz))
    farm = str(z["farm"])
    turb = WP.TURBINES["MM92" if farm == "kelmarsh" else "MM82"]
    D = float(turb["rotor_diameter"])
    east, north = z["east_m"].astype(np.float64), z["north_m"].astype(np.float64)
    elev, hub_h = z["elev_m"].astype(np.float64), z["hub_h_m"].astype(np.float64)
    T, n = z["online"].shape

    t0 = time.time()
    base = _baselines(proc_npz, T, n, jobs)
    print(f"  {farm}: baselines in {time.time() - t0:.0f}s", flush=True)

    out_all = {}
    for tag, net in nets.items():
        dev_base = net.meta["dev_baselines"]
        dev_waked_eta = float(dev_base["mean_waked_eta"])
        dev_farm_wake = float(dev_base["mean_farm_wake_loss"])

        eta_wn = np.zeros((T, n), np.float32)
        sig_wn = np.zeros((T, n), np.float32)
        eta_floris = base["floris_gauss"]
        farm_wn = np.zeros(T, np.float32)
        farm_sig_wn = np.zeros(T, np.float32)
        direct_mask = np.zeros((T, n), bool)
        multi_wake_mask = np.zeros((T, n), bool)
        anchor_gap = 0.0
        t1 = time.time()
        for t in range(T):
            on = z["online"][t]
            r_wn = net.read(
                east, north, hub_h, turb, u_inf=float(z["u_inf"][t]), wind_dir_deg=float(z["wd_inf"][t]),
                ti_inf=float(z["ti_inf"][t]), sigma_wd_deg=float(z["sigma_wd"][t]), yaw_deg=z["yaw_deg"][t],
                online=on, elev_m=elev, air_density=float(z["air_density"][t]),
                p_front_kw=float(z["p_front_kw"][t]), calibrated=True,
            )
            eta_wn[t], sig_wn[t] = r_wn["eta"], r_wn["eta_sig"]
            farm_wn[t], farm_sig_wn[t] = r_wn["farm_wake_loss"], r_wn["farm_wake_sig"]
            if net.meta.get("anchor", "textbook") == "textbook":
                anchor_gap = max(anchor_gap, float(np.max(np.abs(r_wn["eta_anchor"] - eta_floris[t]))))

            # Pure geometric masks for stratum breakdowns (independent of targets)
            xm, ym = WP.rotate_to_wind_frame(east, north, float(z["wd_inf"][t]))
            dx_d = (xm[None, :] - xm[:, None]) / D
            dy_d = np.abs(ym[None, :] - ym[:, None]) / D
            hit_dir = on[:, None] & on[None, :] & (dx_d > 0.5) & (dx_d < 8.0) & (dy_d < 0.65)
            np.fill_diagonal(hit_dir, False)
            direct_mask[t] = np.any(hit_dir, axis=0)
            hit_any = on[:, None] & on[None, :] & (dx_d > 0.5) & (dx_d < 16.0) & (dy_d < 1.1 + 0.08 * np.maximum(dx_d, 0.0))
            np.fill_diagonal(hit_any, False)
            multi_wake_mask[t] = np.sum(hit_any, axis=0) >= 2
        print(f"  {farm} [{tag}]: WakeNet in {time.time() - t1:.0f}s; textbook FLORIS vs v1.0 anchor max gap {anchor_gap:.1e}", flush=True)

        on_all, free_all = z["online"], z["is_free"]
        waked_all = on_all & (~free_all)
        eta_true = np.where(on_all, z["power_kw"] / z["p_front_kw"][:, None], 0.0)
        farm_true = 1.0 - np.sum(eta_true * on_all, axis=1) / np.maximum(np.sum(on_all, axis=1), 1)

        def farm_of(eta):
            return (1.0 - np.sum(eta * on_all, axis=1) / np.maximum(np.sum(on_all, axis=1), 1)).astype(np.float32)

        preds = {
            "wakenet": eta_wn, "floris_tuned": base["floris_tuned"], "floris_gauss": eta_floris,
            "jensen075": base["jensen075"], "jensen050": base["jensen050"],
            "dev_mean": np.float32(dev_waked_eta), "unwaked": np.float32(1.0),
        }
        day = _day_index(z["ts"])
        day2d = np.broadcast_to(day[:, None], (T, n))

        large_yaw_snap = np.any(free_all & (np.abs(z["yaw_deg"]) >= 8.0), axis=1)
        parked_snap = np.any(~on_all, axis=1)
        masks = {
            "all_waked": waked_all,
            "direct_wake": waked_all & direct_mask,
            "multi_wake_2plus": waked_all & multi_wake_mask,
            "large_yaw_8deg": waked_all & large_yaw_snap[:, None],
            "partial_parked": waked_all & parked_snap[:, None],
            "region2_below_10ms": waked_all & (z["u_inf"] < 10.0)[:, None],
            "near_rated_10p5_12ms": waked_all & (z["u_inf"] >= 10.5)[:, None],
            # High-relief turbines: top quartile of elevation on the farm
            "hilltop_turbines": waked_all & (elev[None, :] >= np.quantile(elev, 0.75)),
        }
        res = {"farm": farm, "n_snapshots": int(T), "n_waked_readings": int(waked_all.sum()),
               "n_days": int(day.max() + 1), "first_ts": str(z["ts"][0]), "last_ts": str(z["ts"][-1])}
        for i, (k, m) in enumerate(masks.items()):
            res[k] = _row(preds, eta_true, m, day2d, BOOT_SEED + 10 * i)

        fpreds = {
            "wakenet": farm_wn, "floris_tuned": farm_of(base["floris_tuned"]), "floris_gauss": farm_of(eta_floris),
            "jensen075": farm_of(base["jensen075"]), "jensen050": farm_of(base["jensen050"]),
            "dev_mean": np.float32(dev_farm_wake), "unwaked": np.float32(0.0),
        }
        res["farm_wake_loss"] = _row(fpreds, farm_true, np.ones(T, bool), day, BOOT_SEED + 100)
        res["coverage_68_pct"] = {
            "waked_turbine_eta": round(100.0 * coverage(eta_wn[waked_all] - eta_true[waked_all], sig_wn[waked_all]), 1),
            "all_online_eta": round(100.0 * coverage(eta_wn[on_all] - eta_true[on_all], sig_wn[on_all]), 1),
            "farm_wake_loss": round(100.0 * coverage(farm_wn - farm_true, farm_sig_wn), 1),
        }
        out_all[tag] = res
    return out_all


def _summary(res: dict) -> None:
    for farm, by_tag in res.items():
        if not (isinstance(by_tag, dict) and by_tag and all(isinstance(v, dict) and "farm" in v for v in by_tag.values())):
            continue
        for tag, r in by_tag.items():
            for k in ROWS + ["farm_wake_loss"]:
                x = r[k]
                print(f"{farm:28s} {tag:6s} {k:22s} WN {x['wakenet_pct']:6.2f}  tuned {x['floris_tuned_pct']:6.2f}  "
                      f"FLORIS {x['floris_gauss_pct']:6.2f}  best={x['best_baseline']:12s} "
                      f"{x['diff_vs_best_pct']:+.2f} {x['diff_vs_best_ci95']}  {x['verdict']}")


def main():
    jobs = int(os.environ.get("EXAM_JOBS", "12"))
    if len(sys.argv) > 1 and sys.argv[1] == "exam2":
        weights = sys.argv[2:] or [os.path.join(MODEL, "wakenet_weights.npz")]
        nets = {json.loads(str(np.load(w)["meta"])).get("version", os.path.basename(w)): WakeNet(w) for w in weights}
        split2 = json.load(open(os.path.join(HERE, "split_exam2.json"), encoding="utf-8"))
        out = {"exam": "exam2", "split_sha256": split2["split_sha256"]}
        for role in ("exam2_kelmarsh", "exam2_penmanshiel"):
            out[role] = evaluate_farm_exam(nets, split2["roles"][role]["file"], jobs)
        out_path = os.path.join(HERE, "exam2_results.json")
    else:
        net = WakeNet()
        out = {
            "released_rung": net.meta["released_rung"],
            "sigma_factor": net.meta["sigma_factor"],
            "rescored": "2026-11-17: tuned-FLORIS and onshore-Jensen baselines, all rows, day-bootstrap intervals",
            "kelmarsh_2020_2021": evaluate_farm_exam({net.meta["version"]: net}, "kelmarsh_exam_2020_2021.npz", jobs),
            "penmanshiel_2019_2020": evaluate_farm_exam({net.meta["version"]: net}, "penmanshiel_exam_2019_2020.npz", jobs),
        }
        out_path = os.path.join(HERE, "exam_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    _summary(out)


if __name__ == "__main__":
    main()
