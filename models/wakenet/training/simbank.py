"""Generate resumable float16 simulated wind-farm graph banks and cache dev-set graphs.

Usage:
    python simbank.py dev_cache
    python simbank.py val1 4000 101 v1 val
    python simbank.py train_v1 24000 1 v1 train
"""
from __future__ import annotations

import concurrent.futures
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402
import physics as WP  # noqa: E402
import pipeline as PL  # noqa: E402
import simulator as SIM  # noqa: E402


def _worker_chunk(args: tuple[int, int, str, str]) -> dict:
    n_target, seed, version, bsplit = args
    rng = np.random.default_rng(seed)
    backdrop = SIM.BackdropSampler(split=bsplit)
    N_MAX = SIM.N_MAX

    node_x = np.empty((n_target, N_MAX, PL.N_NODE_FEAT), dtype=np.float16)
    edge_x = np.empty((n_target, N_MAX, N_MAX, PL.N_EDGE_FEAT), dtype=np.float16)
    edge_mask = np.empty((n_target, N_MAX, N_MAX), dtype=bool)
    online = np.empty((n_target, N_MAX), dtype=bool)
    is_free = np.empty((n_target, N_MAX), dtype=bool)
    eta_anchor = np.empty((n_target, N_MAX), dtype=np.float16)
    eta_target = np.empty((n_target, N_MAX), dtype=np.float16)
    udef_target = np.empty((n_target, N_MAX), dtype=np.float16)
    farm_wake_target = np.empty(n_target, dtype=np.float32)

    k = 0
    while k < n_target:
        s = SIM.simulate_one(rng, version, backdrop)
        if s is None:
            continue
        node_x[k] = s["node_x"]
        edge_x[k] = s["edge_x"]
        edge_mask[k] = s["edge_mask"]
        online[k] = s["online"]
        is_free[k] = s["is_free"]
        eta_anchor[k] = s["eta_anchor"]
        eta_target[k] = s["eta_target"]
        udef_target[k] = s["udef_target"]
        farm_wake_target[k] = s["farm_wake_target"]
        k += 1

    return {
        "node_x": node_x,
        "edge_x": edge_x,
        "edge_mask": edge_mask,
        "online": online,
        "is_free": is_free,
        "eta_anchor": eta_anchor,
        "eta_target": eta_target,
        "udef_target": udef_target,
        "farm_wake_target": farm_wake_target,
    }


def build_sim_bank(name: str, count: int, seed: int, version: str, bsplit: str, workers: int = 4):
    out_path = _P.work("banks", f"{name}.npz", mkdir=True)
    if os.path.exists(out_path):
        print(f"Bank {name} already exists at {out_path}", flush=True)
        return
    t0 = time.time()
    n_chunks = workers * 4
    base = count // n_chunks
    sizes = [base + (1 if i < (count % n_chunks) else 0) for i in range(n_chunks)]
    tasks = [(sz, seed * 10_000 + i, version, bsplit) for i, sz in enumerate(sizes) if sz > 0]

    parts = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_worker_chunk, tasks):
            parts.append(res)

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    merged["physics_version"] = np.array(version)
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, **merged)
    os.replace(tmp, out_path)
    dt = time.time() - t0
    n_waked = int(np.sum(merged["online"] & (~merged["is_free"])))
    print(f"Built bank {name} ({version}, {bsplit}): {count} farms ({n_waked} waked rotors) in {dt:.1f}s -> {out_path}", flush=True)


def build_real_graph_cache(proc_npz: str, out_name: str):
    """Precompute PL.build_graph features for a real SCADA table (e.g. kelmarsh_dev_2019.npz)."""
    out_path = _P.work("banks", f"{out_name}.npz", mkdir=True)
    if os.path.exists(out_path):
        print(f"Real graph cache {out_name} already exists at {out_path}", flush=True)
        return
    t0 = time.time()
    z = np.load(_P.work("proc", proc_npz))
    farm = str(z["farm"])
    turb = WP.TURBINES["MM92"] if farm == "kelmarsh" else WP.TURBINES["MM82"]
    east, north, elev, hub_h = z["east_m"], z["north_m"], z["elev_m"], z["hub_h_m"]
    T, n = z["online"].shape

    node_x = np.zeros((T, n, PL.N_NODE_FEAT), dtype=np.float32)
    edge_x = np.zeros((T, n, n, PL.N_EDGE_FEAT), dtype=np.float32)
    edge_mask = np.zeros((T, n, n), dtype=bool)
    eta_anchor = np.zeros((T, n), dtype=np.float32)
    u_free_ms = np.zeros((T, n), dtype=np.float32)

    for t in range(T):
        g = PL.build_graph(
            east, north, hub_h, turb,
            u_inf=float(z["u_inf"][t]),
            wind_dir_deg=float(z["wd_inf"][t]),
            ti_inf=float(z["ti_inf"][t]),
            sigma_wd_deg=float(z["sigma_wd"][t]),
            yaw_deg=z["yaw_deg"][t],
            online=z["online"][t],
            elev_m=elev,
            air_density=float(z["air_density"][t]),
        )
        node_x[t] = g["node_x"]
        edge_x[t] = g["edge_x"]
        edge_mask[t] = g["edge_mask"]
        eta_anchor[t] = g["eta_anchor"]
        u_free_ms[t] = g["u_free_ms"]

    on = z["online"]
    free = z["is_free"]
    eta_true = np.where(on, z["power_kw"] / z["p_front_kw"][:, None], 0.0).astype(np.float32)
    farm_wake_true = (1.0 - np.sum(eta_true * on, axis=1) / np.maximum(np.sum(on, axis=1), 1)).astype(np.float32)

    np.savez(
        out_path,
        ts=z["ts"],
        node_x=node_x,
        edge_x=edge_x,
        edge_mask=edge_mask,
        online=on,
        is_free=free,
        eta_anchor=eta_anchor,
        u_free_ms=u_free_ms,
        eta_true=eta_true,
        farm_wake_true=farm_wake_true,
        power_kw=z["power_kw"],
        p_front_kw=z["p_front_kw"],
        u_inf=z["u_inf"],
        wd_inf=z["wd_inf"],
        ti_inf=z["ti_inf"],
        sigma_wd=z["sigma_wd"],
        yaw_deg=z["yaw_deg"],
        air_density=z["air_density"],
    )
    print(f"Cached {T} real graphs ({out_name}) in {time.time() - t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dev_cache":
        # WAKENET_ANCHOR=tuned builds the v1.1 anchor features into dev_2019_graphs_tuned
        sfx = {"tuned": "_tuned", "tuned_v4": "_tuned_v4"}.get(os.environ.get("WAKENET_ANCHOR", ""), "")
        build_real_graph_cache("kelmarsh_dev_2019.npz", f"dev_2019_graphs{sfx}")
    else:
        count = int(sys.argv[2])
        seed = int(sys.argv[3])
        ver = sys.argv[4]
        bsplit = sys.argv[5] if len(sys.argv) > 5 else ("val" if cmd.startswith("val") else "train")
        build_sim_bank(cmd, count, seed, ver, bsplit)
