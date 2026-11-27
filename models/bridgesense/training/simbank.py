"""Generate float16 simulation banks for BridgeSense (resumable, 1-thread friendly).

Usage:
    python simbank.py train1 16000 101 --physics p1
    python simbank.py val1    2500 901 --physics p1
    python simbank.py train2 20000 201 --physics p2
    python simbank.py val2    2500 902 --physics p2
    python simbank.py train3 24000 301 --physics p3 --cal cal_p3.json
    python simbank.py val3    2500 903 --physics p3 --cal cal_p3.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from _paths import work
from features import extract_features
from realdata import MAX_MODES, N_Q, N_REF
from simulator import simulate_episode


def _worker_chunk(args):
    n_eps, seed, physics, noise_path, cal = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    rng = np.random.default_rng(seed)
    noise_pool = None
    if noise_path and os.path.exists(noise_path):
        np_z = np.load(noise_path)
        noise_pool = {"resid_windows": np_z["resid_windows"], "mask_windows": np_z["mask_windows"]}

    x_ref = np.empty((n_eps, N_REF, 10), dtype=np.float16)
    x_q = np.empty((n_eps, N_Q, 10), dtype=np.float16)
    x_mode = np.empty((n_eps, MAX_MODES, 14), dtype=np.float16)
    m_mask = np.empty((n_eps, MAX_MODES), dtype=np.float16)
    g_feat = np.empty((n_eps, 16), dtype=np.float16)
    anchor = np.empty((n_eps, 3), dtype=np.float32)
    mode_anchor = np.empty((n_eps, MAX_MODES), dtype=np.float16)

    y_changed = np.empty((n_eps,), dtype=np.float32)
    y_dk = np.empty((n_eps,), dtype=np.float32)
    y_env = np.empty((n_eps,), dtype=np.float32)
    y_mode_dk = np.empty((n_eps, MAX_MODES), dtype=np.float16)
    is_cold_extrap = np.empty((n_eps,), dtype=np.uint8)

    for i in range(n_eps):
        ep = simulate_episode(
            rng, physics_version=physics, n_ref=N_REF, n_q=N_Q, max_modes=MAX_MODES,
            noise_pool=noise_pool, cal=cal,
        )
        f = extract_features(
            ep["T_ref"], ep["T_q"], ep["df_ref"], ep["df_q"], ep["mask_ref"], ep["mask_q"]
        )
        x_ref[i] = f["x_ref"].astype(np.float16)
        x_q[i] = f["x_q"].astype(np.float16)
        x_mode[i] = f["x_mode"].astype(np.float16)
        m_mask[i] = f["m_mask"].astype(np.float16)
        g_feat[i] = f["g_feat"].astype(np.float16)
        anchor[i] = f["anchor"]
        mode_anchor[i] = f["mode_anchor"].astype(np.float16)

        y_changed[i] = float(ep["is_changed"])
        y_dk[i] = float(ep["true_dk_pct"])
        y_env[i] = float(ep["true_env_pct"])
        y_mode_dk[i] = ep["mode_dk_pct"].astype(np.float16)
        is_cold_extrap[i] = 1 if (float(f["g_feat"][10]) > 0.2 and float(f["g_feat"][13]) > 0.1) else 0

    return {
        "x_ref": x_ref, "x_q": x_q, "x_mode": x_mode, "m_mask": m_mask,
        "g_feat": g_feat, "anchor": anchor, "mode_anchor": mode_anchor,
        "y_changed": y_changed, "y_dk": y_dk, "y_env": y_env,
        "y_mode_dk": y_mode_dk, "is_cold_extrap": is_cold_extrap,
    }


def generate_bank(name: str, n: int, seed: int, physics: str, cal: dict | None = None,
                  workers: int = 2) -> str:
    dst = work("banks", f"{name}.npz", mkdir=True)
    if os.path.exists(dst):
        print(f"Bank already exists: {dst}")
        return dst
    noise_path = work("noise_pool.npz") if physics == "p3" else None
    t0 = time.time()
    chunk_size = 500
    tasks = []
    rem = n
    cid = 0
    while rem > 0:
        sz = min(chunk_size, rem)
        tasks.append((sz, seed * 10000 + cid, physics, noise_path, cal))
        rem -= sz
        cid += 1

    parts = []
    if workers <= 1 or len(tasks) == 1:
        for k, t in enumerate(tasks, 1):
            parts.append(_worker_chunk(t))
            if k % 4 == 0 or k == len(tasks):
                done = sum(len(p["y_changed"]) for p in parts)
                print(f"  [{name}] {done}/{n} ({time.time()-t0:.1f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, p in enumerate(ex.map(_worker_chunk, tasks), 1):
                parts.append(p)
                if k % 4 == 0 or k == len(tasks):
                    done = sum(len(x["y_changed"]) for x in parts)
                    print(f"  [{name}] {done}/{n} ({time.time()-t0:.1f}s)", flush=True)

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    np.savez(dst, **merged)
    dt = time.time() - t0
    chg_frac = float(np.mean(merged["y_changed"]))
    cold_frac = float(np.mean(merged["is_cold_extrap"]))
    print(f"Saved {dst}: {n} episodes ({physics}) in {dt:.1f}s | changed={chg_frac:.2f}, cold_extrap={cold_frac:.2f}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("--physics", default="p2", choices=["p1", "p2", "p3"])
    ap.add_argument("--cal", default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    cal = json.load(open(args.cal)) if args.cal else None
    generate_bank(args.name, args.n, args.seed, args.physics, cal=cal, workers=args.workers)


if __name__ == "__main__":
    main()
