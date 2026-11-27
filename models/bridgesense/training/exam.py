"""One-shot real exam for BridgeSense across three full-scale bridges.

Run ONLY ONCE after the ratchet ladder has chosen and exported the released model
(models/bridgesense/bridgesense.py + bridgesense_weights.npz).

Exams:
  - Exam A: KW51 Railway Bridge (49 held-out weekly blocks, splits.json['kw51_exam'])
  - Exam B: Concrete Box-Girder Bridge (Zenodo 20522191, CC BY 4.0, 100% zero-shot)
  - Exam C: Hell Bridge Test Arena (Zenodo 14632942, CC BY 4.0, 100% zero-shot imposed damage)

Compares BridgeSense (pure-numpy released model) beside:
  1. Dumb baseline (dev prevalence / 0% shift / raw uncompensated 200*df)
  2. Classical 1: Per-mode Linear Thermal OLS + RMS z-score control chart
  3. Classical 2: Dev-Tuned Bilinear Freezing + RMS z-score (Peeters & De Roeck style,
     freezing hinge & thresholds fitted on the exact same KW51 dev split)
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

from mlratchet.calibrate import coverage
from _paths import MODEL, work
from baselines import DevTunedBilinearBaseline, LinearOLSBaseline, dumb_predict
from calibrate import estimate_kw51_post_retrofit_true_dk
from dev_eval import load_kw51_dev_blocks
from realdata import MAX_MODES, N_Q, N_REF, build_kw51_blocks, build_window_episode, load_boxgirder_episodes, load_kw51_hourly

sys.path.insert(0, MODEL)
from bridgesense import BridgeSense  # noqa: E402


def _boot_ci(vals: np.ndarray, fn=np.mean, n_boot: int = 1000, seed: int = 2026) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        v = float(fn(vals)) if len(vals) == 1 else float("nan")
        return [round(v, 4), round(v, 4)]
    bs = [float(fn(vals[rng.integers(0, len(vals), size=len(vals))])) for _ in range(n_boot)]
    return [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]


def _boot_auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 1000, seed: int = 2026) -> list[float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        bs.append(float(roc_auc_score(y[idx], s[idx])))
    if not bs:
        return [float("nan"), float("nan")]
    return [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]


def load_hbta_exam_episodes() -> list[dict]:
    """Build zero-shot HBTA exam episodes from work/hbta_modes.npz (CC BY 4.0).

    Reference baseline: the 5 first-replicate undamaged recordings (UDS *_01).
    Exam queries: the 5 second-replicate undamaged recordings (UDS *_02, healthy control)
                  and all 40 imposed physical damage recordings (DS1..DS8).
    """
    z = np.load(work("hbta_modes.npz"), allow_pickle=True)
    rec_names = list(z["rec_names"])
    states = list(z["states"])
    dates = list(z["dates"])
    temps = np.asarray(z["temps"], dtype=np.float64)
    freqs = np.asarray(z["freqs"], dtype=np.float64)  # [50, 8, 7]

    # Reference pool: UDS recordings ending in _01
    ref_ids = [i for i, (rn, st) in enumerate(zip(rec_names, states)) if st == "UDS" and rn.endswith("_01")]
    q_ids = [i for i in range(len(rec_names)) if i not in ref_ids]

    # Stack reference windows: [len(ref_ids)*8, 7]
    ref_f_list, ref_t_list = [], []
    for i in ref_ids:
        n_w = freqs.shape[1]
        ref_f_list.append(freqs[i])
        ref_t_list.append(np.full((n_w, 3), temps[i], dtype=np.float64))
    ref_f = np.concatenate(ref_f_list, axis=0)
    ref_T3 = np.concatenate(ref_t_list, axis=0)
    f0_uds = np.nanmedian(ref_f, axis=0)

    episodes = []
    for i in q_ids:
        st = states[i]
        is_chg = 0 if st == "UDS" else 1
        q_f = freqs[i]
        q_T3 = np.full((len(q_f), 3), temps[i], dtype=np.float64)
        # Model-free ground truth stiffness-equivalent shift from measured frequencies vs UDS baseline
        ok_m = np.isfinite(np.nanmedian(q_f, axis=0)) & np.isfinite(f0_uds)
        if np.any(ok_m):
            f_q_med = np.nanmedian(q_f[:, ok_m], axis=0)
            true_dk = float(100.0 * np.mean((f_q_med / f0_uds[ok_m]) ** 2 - 1.0)) if is_chg else 0.0
        else:
            true_dk = 0.0
        cat = "undamaged" if st == "UDS" else ("local_damage" if st in ("DS1", "DS2", "DS3", "DS4") else "global_damage")
        ep = build_window_episode(
            ref_T3, ref_f, q_T3, q_f,
            meta={
                "block_id": rec_names[i],
                "bridge": "hbta",
                "state": st,
                "stratum": cat,
                "date": dates[i],
                "T_min": round(float(temps[i]), 1),
                "T_mean": round(float(temps[i]), 1),
                "is_changed": is_chg,
                "true_dk_pct": round(true_dk, 3),
            },
        )
        episodes.append(ep)
    return episodes


LOG_DIR = os.path.join(MODEL, "training", "logs", "v1_1")
RUN_RECORD = os.path.join(LOG_DIR, "exam_run.json")


def _sha256(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write_json(path: str, obj, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=indent)
        f.write("\n")


def run_exam() -> dict:
    # Opened once: refuse to run again once a dated run record exists for this version
    if os.path.exists(RUN_RECORD):
        raise SystemExit(f"Exam already opened for v1.1 ({RUN_RECORD}); a re-run needs a new version")
    import datetime
    import subprocess
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    weights = os.path.join(MODEL, "bridgesense_weights.npz")
    model = BridgeSense()
    dev_blocks = load_kw51_dev_blocks()
    dev_prev = float(np.mean([b["is_changed"] for b in dev_blocks]))

    # Fit classical baselines strictly on KW51 Dev (zero exam access)
    ols = LinearOLSBaseline()
    ols_fit = ols.fit_dev(dev_blocks)
    bilin = DevTunedBilinearBaseline()
    bilin_fit = bilin.fit_dev(dev_blocks)
    print("Baseline dev fits:", {"ols": ols_fit, "bilinear": bilin_fit, "dev_prev": dev_prev})

    # Compute independent matched-temperature post-retrofit dK/K reference on KW51 Exam post blocks
    splits = json.load(open(os.path.join(MODEL, "splits.json")))
    exam_ids = {b["block_id"] for b in splits["kw51_exam"]}
    _, all_kw = build_kw51_blocks(load_kw51_hourly(), block_days=7)
    kw_exam = [b for b in all_kw if b["block_id"] in exam_ids]

    # Ground truth post-retrofit dK/K measured model-free at matched non-freezing temperatures
    kw_post_true_dk = estimate_kw51_post_retrofit_true_dk(kw_exam)
    print(f"\n=== EXAM A: KW51 HELD-OUT ({len(kw_exam)} blocks, post true dK={kw_post_true_dk:+.2f}%) ===")

    def eval_suite(episodes: list[dict], true_dk_fn) -> dict:
        rows = []
        for ep in episodes:
            r_bs = model.predict(ep["T_ref"], ep["T_q"], ep["df_ref"], ep["df_q"], ep["mask_ref"], ep["mask_q"])
            r_ols = ols.predict(ep)
            r_bil = bilin.predict(ep)
            r_dmb = dumb_predict(ep, dev_prevalence=dev_prev)
            tdk = true_dk_fn(ep)
            rows.append({
                "block_id": ep["block_id"],
                "stratum": ep["stratum"],
                "is_changed": ep["is_changed"],
                "T_min": ep.get("T_min", 0.0),
                "true_dk": tdk,
                "bs_p": r_bs["p_change"], "bs_dk": r_bs["delta_k_pct"], "bs_sig": r_bs["delta_k_sigma"],
                "ols_s": r_ols["score"], "ols_p": r_ols["p_change"], "ols_dk": r_ols["dk_pct"],
                "bil_s": r_bil["score"], "bil_p": r_bil["p_change"], "bil_dk": r_bil["dk_pct"],
                "dmb_p": r_dmb["p_change"], "dmb_dk_zero": r_dmb["dk_zero_pct"], "dmb_dk_raw": r_dmb["dk_raw_pct"],
                "dmb_s": r_dmb["score"],
            })
        y = np.array([r["is_changed"] for r in rows], dtype=float)
        res = {"n": len(rows), "n_pos": int(np.sum(y > 0.5)), "n_neg": int(np.sum(y < 0.5)), "rows": rows}
        if len(np.unique(y)) == 2:
            for k_m, k_s, k_p in [("bs", "bs_p", "bs_p"), ("ols", "ols_s", "ols_p"), ("bil", "bil_s", "bil_p"), ("dmb", "dmb_s", "dmb_p")]:
                sc = np.array([r[k_s] for r in rows])
                pr = np.array([r[k_p] for r in rows])
                res[f"{k_m}_auc"] = round(float(roc_auc_score(y, sc)), 4)
                res[f"{k_m}_auc_ci"] = _boot_auc_ci(y, sc)
                res[f"{k_m}_brier"] = round(float(np.mean((pr - y) ** 2)), 4)
                res[f"{k_m}_brier_ci"] = _boot_ci((pr - y) ** 2)

        # Freezing false-alarm rate (healthy blocks with stratum == healthy_freezing)
        frz = [r for r in rows if r["stratum"] == "healthy_freezing"]
        if frz:
            res["n_freezing"] = len(frz)
            for k_m, k_p in [("bs", "bs_p"), ("ols", "ols_p"), ("bil", "bil_p")]:
                ps = np.array([r[k_p] for r in frz])
                res[f"{k_m}_freeze_far"] = round(float(np.mean(ps >= 0.5)), 4)
                res[f"{k_m}_freeze_p_mean"] = round(float(np.mean(ps)), 4)

        # Healthy dK/K MAE (true = 0%)
        hlth = [r for r in rows if r["is_changed"] == 0]
        if hlth:
            for k_m, k_dk in [("bs", "bs_dk"), ("ols", "ols_dk"), ("bil", "bil_dk"), ("dmb_raw", "dmb_dk_raw")]:
                err = np.abs(np.array([r[k_dk] for r in hlth]))
                res[f"{k_m}_healthy_dk_mae"] = round(float(np.mean(err)), 3)
                res[f"{k_m}_healthy_dk_ci"] = _boot_ci(err)

        # Scored true_dk subset (where true_dk is not None)
        scored = [r for r in rows if r["true_dk"] is not None]
        if scored:
            td = np.array([r["true_dk"] for r in scored])
            for k_m, k_dk in [("bs", "bs_dk"), ("ols", "ols_dk"), ("bil", "bil_dk"), ("dmb_zero", "dmb_dk_zero"), ("dmb_raw", "dmb_dk_raw")]:
                err = np.abs(np.array([r[k_dk] for r in scored]) - td)
                res[f"{k_m}_dk_mae"] = round(float(np.mean(err)), 3)
                res[f"{k_m}_dk_ci"] = _boot_ci(err)
            bs_res = np.array([r["bs_dk"] for r in scored]) - td
            bs_sig = np.array([r["bs_sig"] for r in scored])
            res["bs_cov_68"] = round(float(coverage(bs_res, bs_sig, 1.0)), 4)
            res["bs_mean_sig"] = round(float(np.mean(bs_sig)), 3)
        return res

    # Exam A: KW51 held-out
    exam_a = eval_suite(
        kw_exam,
        lambda ep: 0.0 if ep["is_changed"] == 0 else (kw_post_true_dk if ep["stratum"] == "post_retrofit" else None),
    )
    # Also compute KW51 post-retrofit-only dK MAE
    kw_post_rows = [r for r in exam_a["rows"] if r["stratum"] == "post_retrofit"]
    exam_a["kw_post_true_dk_pct"] = round(kw_post_true_dk, 3)
    for k_m, k_dk in [("bs", "bs_dk"), ("ols", "ols_dk"), ("bil", "bil_dk"), ("dmb_zero", "dmb_dk_zero"), ("dmb_raw", "dmb_dk_raw")]:
        err = np.abs(np.array([r[k_dk] for r in kw_post_rows]) - kw_post_true_dk)
        exam_a[f"{k_m}_post_dk_mae"] = round(float(np.mean(err)), 3)

    print(f"  KW51 AUC:   BS={exam_a['bs_auc']} {exam_a['bs_auc_ci']} | OLS={exam_a['ols_auc']} | Bilin={exam_a['bil_auc']} | Dumb={exam_a['dmb_auc']}")
    print(f"  KW51 Brier: BS={exam_a['bs_brier']} | OLS={exam_a['ols_brier']} | Bilin={exam_a['bil_brier']} | Dumb={exam_a['dmb_brier']}")
    print(f"  KW51 Freeze FAR (n={exam_a['n_freezing']}): BS={exam_a['bs_freeze_far']} (meanP={exam_a['bs_freeze_p_mean']}) | "
          f"OLS={exam_a['ols_freeze_far']} | Bilin={exam_a['bil_freeze_far']}")
    print(f"  KW51 Healthy dK MAE: BS={exam_a['bs_healthy_dk_mae']}% | OLS={exam_a['ols_healthy_dk_mae']}% | Bilin={exam_a['bil_healthy_dk_mae']}% | Raw={exam_a['dmb_raw_healthy_dk_mae']}%")
    print(f"  KW51 Post dK MAE:    BS={exam_a['bs_post_dk_mae']}% | OLS={exam_a['ols_post_dk_mae']}% | Bilin={exam_a['bil_post_dk_mae']}% | Zero={exam_a['dmb_zero_post_dk_mae']}%")
    print(f"  KW51 68% Coverage (n=31): {exam_a['bs_cov_68']*100:.1f}% (mean sigma={exam_a['bs_mean_sig']}%)")

    # Exam B: Concrete Box-Girder Bridge (100% zero-shot)
    bg = load_boxgirder_episodes(block_days=5)
    bg_eps = bg["episodes"]
    print(f"\n=== EXAM B: CONCRETE BOX-GIRDER ZERO-SHOT ({len(bg_eps)} blocks, 2 modes) ===")
    exam_b = eval_suite(bg_eps, lambda ep: 0.0 if ep["is_changed"] == 0 else None)
    print(f"  Box-Girder AUC (cold healthy vs 680-2160kg load): BS={exam_b.get('bs_auc')} | OLS={exam_b.get('ols_auc')} | Bilin={exam_b.get('bil_auc')}")
    print(f"  Box-Girder Freeze FAR (n={exam_b.get('n_freezing')}): BS={exam_b.get('bs_freeze_far')} (meanP={exam_b.get('bs_freeze_p_mean')}) | "
          f"OLS={exam_b.get('ols_freeze_far')} | Bilin={exam_b.get('bil_freeze_far')}")
    print(f"  Box-Girder Healthy dK MAE: BS={exam_b.get('bs_healthy_dk_mae')}% | OLS={exam_b.get('ols_healthy_dk_mae')}% | Bilin={exam_b.get('bil_healthy_dk_mae')}% | Raw={exam_b.get('dmb_raw_healthy_dk_mae')}%")
    print(f"  Box-Girder Healthy 68% Coverage: {exam_b.get('bs_cov_68', 0)*100:.1f}%")

    # Exam C: Hell Bridge Test Arena (100% zero-shot imposed physical damage)
    hbta_eps = load_hbta_exam_episodes()
    print(f"\n=== EXAM C: HELL BRIDGE TEST ARENA ZERO-SHOT ({len(hbta_eps)} recordings, 7 modes, DS1-DS8) ===")
    exam_c = eval_suite(hbta_eps, lambda ep: ep["true_dk_pct"])
    # Also compute Global Damage (DS5-DS8 vs UDS) specifically
    hbta_glob = [ep for ep in hbta_eps if ep["stratum"] in ("undamaged", "global_damage")]
    exam_c_glob = eval_suite(hbta_glob, lambda ep: ep["true_dk_pct"])
    exam_c["global_damage_only"] = {k: v for k, v in exam_c_glob.items() if k != "rows"}
    print(f"  HBTA All Damage AUC:    BS={exam_c['bs_auc']} {exam_c['bs_auc_ci']} | OLS={exam_c['ols_auc']} | Bilin={exam_c['bil_auc']} | Dumb={exam_c['dmb_auc']}")
    print(f"  HBTA Global DS5-8 AUC:  BS={exam_c_glob['bs_auc']} {exam_c_glob['bs_auc_ci']} | OLS={exam_c_glob['ols_auc']} | Bilin={exam_c_glob['bil_auc']}")
    print(f"  HBTA dK/K MAE:          BS={exam_c['bs_dk_mae']}% {exam_c['bs_dk_ci']} | OLS={exam_c['ols_dk_mae']}% | Bilin={exam_c['bil_dk_mae']}% | Zero={exam_c['dmb_zero_dk_mae']}%")
    print(f"  HBTA 68% Coverage:      {exam_c['bs_cov_68']*100:.1f}% (mean sigma={exam_c['bs_mean_sig']}%)")

    out = {
        "baselines_dev_fit": {"ols": ols_fit, "bilinear": bilin_fit, "dev_prev": dev_prev},
        "exam_a_kw51": exam_a,
        "exam_b_boxgirder": exam_b,
        "exam_c_hbta": exam_c,
    }
    dst = work("exam_results.json", mkdir=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    _write_json(os.path.join(LOG_DIR, "exam.json"), out, indent=1)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=MODEL).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=MODEL).stdout.strip())
    except OSError:
        commit, dirty = "unknown", None
    _write_json(RUN_RECORD, {
        "model": "bridgesense",
        "version": model.meta.get("version"),
        "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "weights_sha256": _sha256(weights),
        "members": model.meta.get("members"),
        "splits_sha256": _sha256(os.path.join(MODEL, "splits.json")),
        "hbta_bands_sha256": _sha256(os.path.join(MODEL, "training", "hbta_bands.json")),
        "hbta_modes_sha256": _sha256(work("hbta_modes.npz")),
        "command": "python training/exam.py",
        "n_blocks": {"kw51": exam_a["n"], "boxgirder": exam_b["n"], "hbta": exam_c["n"]},
    })
    print(f"\nSaved {dst}, {LOG_DIR}/exam.json and the run record {RUN_RECORD}")
    return out


if __name__ == "__main__":
    run_exam()
