"""AirfoilNet (UIUC Low-Speed Airfoil Tests, ln Cd drag) adapter for TreeTwin + ConformalRatchet.

- Simulator : mfoil_fast (Numba-accelerated XFOIL-class coupled viscous-inviscid panel/boundary-layer
              solver) run on non-tested UIUC coordinate shapes + synthetic CST blends across
              Re in [4e4, 6e5], ncrit in [5, 11] (zero tested UIUC airfoils in training!).
- Features  : Exact 53-feature AirfoilNet tabular vector: 24 thickness stations, 24 camber
              stations, normalized log10(Re), tmax/0.12, plus 3 alpha features (alpha/10,
              sin(2*alpha), cos(alpha)).
- Target    : ln(Cd) (wake-rake pre-stall drag); headline metric is |ln(Cd_pred / Cd_true)| MAE.
- Monotone  : -1 on log10(Re) (skin friction and laminar separation bubble drag decrease as Re
              rises) and +1 on tmax (form drag increases with maximum thickness).
- Real splits: Frozen UIUC family split from models/airfoilnet/training/splits.json
              (dev = 7 airfoils in 4 families; exam = 75 unseen airfoils). Zero real labels in train.
- Strata    : Reynolds-number x angle-of-attack regime (lowRe_lowA, lowRe_highA, highRe_lowA,
              highRe_highA, split at Re = 1.5e5 and |alpha| = 6 deg) for Mondrian conformal.
- Incumbent : Released AirfoilNet v1.0 DeepONet ensemble (models/airfoilnet/airfoilnet_weights.npz).
"""
from __future__ import annotations

import concurrent.futures as cf
import functools
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

import tt_paths as P

P.add_sys_path(P.REPO / "models" / "airfoilnet")
P.add_sys_path(P.REPO / "models" / "airfoilnet" / "training")
os.environ.setdefault("NUMBA_CACHE_DIR", str(P.work("airfoilnet", "numba_cache")))

NAME = "airfoilnet"
TITLE = "AirfoilNet (UIUC LSAT drag)"
DOMAIN = "Aerodynamics"
TARGET_NAME = "ln_cd"
UNIT = "ln Cd"
SHAP_SCALE = 100.0   # delta ln(Cd) -> approx % drag shift
SHAP_UNIT = "% drag shift"

FEATURE_NAMES = (
    [f"thick_{i:02d}" for i in range(24)]
    + [f"camber_{i:02d}" for i in range(24)]
    + ["log10_Re_norm", "tmax_norm", "alpha_norm", "sin_2alpha", "cos_alpha"]
)

FEATURE_GROUPS = {
    "Leading-edge thickness (0-25% c)": list(range(0, 8)),
    "Mid-chord thickness (25-75% c)": list(range(8, 16)),
    # Stations are cosine-spaced (airfoilnet.py XS): 0-7 lie at 0.4-23 % c, 8-15 at 29-71 %, 16-23 at 77-100 %.
    "Trailing-edge thickness (75-100% c)": list(range(16, 24)),
    # tmax sits near 25-35 % c on most sections; it was filed under trailing edge until 27 Nov 2026.
    "Maximum thickness (tmax)": [49],
    "Leading-edge camber": list(range(24, 32)),
    "Mid/aft camber": list(range(32, 48)),
    "Reynolds number (log10 Re)": [48],
    "Angle of attack (alpha)": [50, 51, 52],
}

# Physically justified monotonicity on ln(Cd): -1 on log10(Re), +1 on tmax
MONOTONE_CST = [0] * 48 + [-1, 1, 0, 0, 0]


def _stratum(Re: float, alpha: float) -> str:
    re_band = "lowRe" if Re < 1.5e5 else "highRe"
    a_band = "highA" if abs(alpha) >= 6.0 else "lowA"
    return f"{re_band}_{a_band}"


def _build_units_and_shapes() -> tuple[Path, Path]:
    units_path = P.work("airfoilnet", "units.json", mkdir=True)
    shapes_path = P.work("airfoilnet", "shapes.npz", mkdir=True)
    if units_path.exists() and shapes_path.exists():
        return units_path, shapes_path

    raw_dir = P.work("airfoilnet", "raw")
    import uiuc_lsat as ul
    import dataset as ds
    import shapes as sh
    import xsolver as xs

    # Point uiuc_lsat and dataset at our work/airfoilnet/raw directory
    ul.ROOT = str(raw_dir)
    ds.RAW = str(raw_dir)
    ds.DB = str(raw_dir / "coord_seligFmt" / "coord_seligFmt")

    recs = ul.parse_all()
    cf_coords = ds.load_coord_files()
    splits = json.loads((P.REPO / "models" / "airfoilnet" / "training" / "splits.json").read_text())
    known_ids = set(splits["dev"]) | set(splits["exam"])

    units: dict[str, dict] = {}
    tested_names: set[str] = set()
    for r in recs:
        if r["kind"] != "drag":
            continue
        comment = (r.get("comment") or "").lower()
        if any(w in comment for w in ds.EXCLUDE_WORDS):
            continue
        if not r.get("airfoil"):
            continue
        aid, model = ds.label_key(r["airfoil"])
        if aid not in known_ids:
            continue
        src, coords = ds.find_coords(aid, model, cf_coords)
        if coords is None or len(coords) < 15:
            continue
        Y = xs.normalize(coords)
        p, rms = sh.cst_fit(Y)
        if rms > 0.015:
            continue
        tested_names.add(aid)
        if aid not in units:
            units[aid] = {"airfoil": aid, "runs": []}
        units[aid]["runs"].append({
            "vol": r["vol"],
            "model": model or "",
            "Re": float(r["Re"]),
            "p": p.tolist(),
            "data": [[float(row[0]), float(row[1]), float(row[2])] for row in r["data"]
                     if len(row) >= 3 and np.isfinite(row[0]) and np.isfinite(row[2]) and float(row[2]) > 0],
        })

    units_path.write_text(json.dumps(units))

    # Build non-tested UIUC shape pool (strictly excluding every dev/exam airfoil!)
    db_files = sorted(glob.glob(str(raw_dir / "coord_seligFmt" / "coord_seligFmt" / "*.dat")))
    pool = []
    for f in db_files:
        stem = Path(f).stem.lower()
        if any(aid in stem for aid in known_ids):
            continue
        try:
            coords = xs.read_dat(f)
            if len(coords) < 20:
                continue
            Y = xs.normalize(coords)
            p, rms = sh.cst_fit(Y)
            if rms < 0.005 and sh.valid(p):
                pool.append(p)
        except Exception:
            continue

    rng = np.random.default_rng(20261124)
    pool_arr = np.array(pool, dtype=np.float64)
    order = rng.permutation(len(pool_arr))
    pool_arr = pool_arr[order]
    n_val_shapes = min(70, max(30, len(pool_arr) // 10))
    isval = np.zeros(len(pool_arr), dtype=bool)
    isval[:n_val_shapes] = True
    np.savez_compressed(shapes_path, pool=pool_arr, isval=isval)
    print(f"[airfoilnet] Parsed {len(units)} tested airfoils, {len(pool_arr)} non-tested pool shapes.", flush=True)
    return units_path, shapes_path


def _sweep_worker(task: tuple) -> list[tuple] | None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import airfoilnet as af
    import shapes as sh
    import xsolver as xs

    p, Re, ncrit, up, dn = task
    p_arr = np.asarray(p, dtype=np.float64)
    Y = sh.cst_coords(p_arr)
    try:
        res = xs.polar_sweep(
            Y, float(Re), alphas_up=np.array(up), alphas_dn=np.array(dn),
            ncrit=float(ncrit), npanel=100, max_fail=1, time_limit=4.0,
        )
    except Exception:
        return None
    if not res:
        return None
    bf = af.features(p_arr, float(Re))
    rows = []
    for r in res:
        cd = float(r["cd"])
        al = float(r["alpha"])
        if not (np.isfinite(cd) and 1e-4 < cd < 0.5 and -6.0 <= al <= 14.0):
            continue
        tf = af.trunk_features(np.array([al]))[0]
        x = np.concatenate([bf, tf]).astype(np.float32)
        rows.append((x, float(np.log(cd)), _stratum(float(Re), al)))
    return rows


def build_sim_banks(n_train_sweeps: int = 360, n_val_sweeps: int = 70, workers: int = 2) -> Path:
    cache = P.work("airfoilnet", "sim_banks.npz", mkdir=True)
    if cache.exists():
        return cache
    _, shapes_path = _build_units_and_shapes()
    z = np.load(shapes_path)
    pool, isval = z["pool"], z["isval"]
    P_tr = pool[~isval]
    P_va = pool[isval]

    def _make_tasks(P_src: np.ndarray, n_sweeps: int, seed_offset: int):
        r2 = np.random.default_rng(20261124 + seed_offset)
        sd = P_src.std(axis=0)
        import shapes as sh
        tasks = []
        for k in range(n_sweeps):
            if k < len(P_src):
                p = P_src[k]
            else:
                i, j = r2.choice(len(P_src), 2, replace=False)
                w = r2.uniform(0, 1)
                p = w * P_src[i] + (1 - w) * P_src[j] + r2.normal(0, 0.12, P_src.shape[1]) * sd
                p[-1] = abs(p[-1]) * 0.5
                if not sh.valid(p):
                    p = P_src[i]
            Re = float(10 ** r2.uniform(np.log10(4e4), np.log10(6e5)))
            ncrit = float(r2.uniform(5.0, 11.0))
            a0 = float(r2.uniform(-0.8, 0.8))
            up = list(a0 + np.arange(0.0, 11.0, 2.0))
            dn = list(a0 - np.arange(2.0, 5.0, 2.0))
            tasks.append((p.tolist(), Re, ncrit, up, dn))
        return tasks

    def _run_chunked(tag: str, tasks: list, chunk_sz: int = 20):
        cdir = P.work("airfoilnet", f"chunks_{tag}")
        cdir.mkdir(parents=True, exist_ok=True)
        xs, ys, sts = [], [], []
        n_chunks = (len(tasks) + chunk_sz - 1) // chunk_sz
        for ci in range(n_chunks):
            cfp = cdir / f"c{ci:03d}.npz"
            if not cfp.exists():
                sub = tasks[ci * chunk_sz : (ci + 1) * chunk_sz]
                with cf.ProcessPoolExecutor(max_workers=workers) as ex:
                    rows = [row for batch in ex.map(_sweep_worker, sub, chunksize=2) if batch for row in batch]
                np.savez_compressed(
                    cfp,
                    X=np.stack([r[0] for r in rows]).astype(np.float32),
                    y=np.array([r[1] for r in rows], dtype=np.float64),
                    st=np.array([r[2] for r in rows], dtype=str),
                )
                print(f"[airfoilnet] {tag} chunk {ci + 1}/{n_chunks}: {len(rows)} pts saved", flush=True)
            zc = np.load(cfp)
            xs.append(zc["X"]); ys.append(zc["y"]); sts.append(zc["st"])
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(sts)

    t_tr = _make_tasks(P_tr, n_train_sweeps, 1)
    t_va = _make_tasks(P_va, n_val_sweeps, 2)
    print(f"[airfoilnet] Running mfoil_fast for {len(t_tr)} train + {len(t_va)} val sweeps ({workers} workers, resumable)...", flush=True)
    X_tr, y_tr, st_tr = _run_chunked("train", t_tr, chunk_sz=20)
    X_va, y_va, st_va = _run_chunked("val", t_va, chunk_sz=20)

    np.savez_compressed(cache, X_tr=X_tr, y_tr=y_tr, st_tr=st_tr, X_va=X_va, y_va=y_va, st_va=st_va)
    print(f"[airfoilnet] Saved sim banks: train={len(X_tr)} pts, val={len(X_va)} pts -> {cache}", flush=True)
    return cache


def _load_real_split(split: str) -> dict:
    units_path, _ = _build_units_and_shapes()
    units = json.loads(units_path.read_text())
    splits = json.loads((P.REPO / "models" / "airfoilnet" / "training" / "splits.json").read_text())
    want = set(splits[split])

    import airfoilnet as af
    X_list, y_list, st_list, uid_list, p_list, re_list, a_list = [], [], [], [], [], [], []
    for aid in sorted(want):
        if aid not in units:
            continue
        for run in units[aid]["runs"]:
            Re = float(run["Re"])
            if not (3.5e4 <= Re <= 7e5):
                continue
            p = np.array(run["p"], dtype=np.float64)
            bf = af.features(p, Re)
            for row in run["data"]:
                al, cd = float(row[0]), float(row[2])
                if not (-6.0 <= al <= 14.0 and 1e-4 < cd < 0.5):
                    continue
                tf = af.trunk_features(np.array([al]))[0]
                X_list.append(np.concatenate([bf, tf]).astype(np.float32))
                y_list.append(float(np.log(cd)))
                st_list.append(_stratum(Re, al))
                uid_list.append(aid)
                p_list.append(p)
                re_list.append(Re)
                a_list.append(al)

    return {
        "X": np.stack(X_list).astype(np.float32),
        "y": np.array(y_list, dtype=np.float64),
        "strata": np.array(st_list, dtype=str),
        "unit_ids": np.array(uid_list, dtype=str),
        "Re": np.array(re_list, dtype=np.float64),
        "alpha": np.array(a_list, dtype=np.float64),
    }


def sim_train_val() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(build_sim_banks())
    return z["X_tr"], z["y_tr"], z["st_tr"], z["X_va"], z["y_va"], z["st_va"]


def real_dev() -> dict:
    return _load_real_split("dev")


def real_exam() -> dict:
    return _load_real_split("exam")


@functools.lru_cache(maxsize=1)
def _incumbent_net():
    import airfoilnet as af
    return af.AirfoilNet()


def incumbent_predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_lncd, raw_unwidened_sigma_lncd) from released AirfoilNet v1.0 DeepONet."""
    net = _incumbent_net()
    X2 = np.asarray(X, dtype=np.float64)
    F = X2[:, :50]
    A = X2[:, 50] * 10.0
    anc = np.zeros((len(X2), 3), dtype=np.float64)
    anc[:, 1] = np.log(0.01)
    mu, sd = net._forward(F, A, anc)
    return mu[:, 1].astype(np.float64), sd[:, 1].astype(np.float64)


def headline_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Drag |ln(Cd_pred / Cd_true)| MAE (lower is better; approx fractional drag error)."""
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def metric_details(y_true: np.ndarray, y_pred: np.ndarray, strata: np.ndarray) -> dict:
    ae = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))
    out = {
        "lncd_mae": round(float(np.mean(ae)), 4),
        "cd_mdape_pct": round(float(100.0 * np.median(np.abs(np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true))), 2),
        "strata": {},
    }
    for g in sorted(set(strata.tolist())):
        m = strata == g
        out["strata"][g] = {
            "n": int(np.sum(m)),
            "lncd_mae": round(float(np.mean(ae[m])), 4),
            "cd_mdape_pct": round(float(100.0 * np.median(np.abs(np.exp(y_pred[m]) - np.exp(y_true[m])) / np.exp(y_true[m]))), 2),
        }
    return out


if __name__ == "__main__":
    build_sim_banks()
    d = real_dev()
    print(f"[airfoilnet] dev pts={len(d['X'])}, airfoils={len(set(d['unit_ids']))}, strata={dict(zip(*np.unique(d['strata'], return_counts=True)))}")
