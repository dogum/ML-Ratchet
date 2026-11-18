"""Deterministic banks. Only the wind-role loader is imported for simulation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from _paths import work
from noise import HERE, PROTOCOL, RESULTS, read_result, wind_period, write_json
from simulator import simulate
from stormphysics import HISTORY, HORIZON, PUBLISHED, drivers

WARMUP = 48


def make_bank(version, frozen, center, with_past=False):
    """with_past (v1.1, 17 Nov 2026) also stores the simulated Dst of the 36-hour window as past_dst. It draws the
    same random numbers, so its wind, dst and truth equal the plain bank of the same version."""
    seed = (7821 if frozen else 4931) + version
    size = 4500 if frozen else 17000
    years = PROTOCOL["frozen_wind_years" if frozen else "training_wind_years"]
    wind, _ = wind_period(years)
    rng = np.random.default_rng(seed)
    width = WARMUP + HISTORY + HORIZON
    valid = np.lib.stride_tricks.sliding_window_view(np.isfinite(wind).all(axis=1), width).all(axis=1)
    starts = np.flatnonzero(valid)
    e, _ = drivers(wind)
    last = starts + WARMUP + HISTORY - 1
    recent = np.max(e[last[:, None] - np.arange(6)], axis=1)
    active = starts[recent >= np.quantile(recent, .8)]
    draws = np.concatenate((rng.choice(starts, size=size // 2), rng.choice(active, size=size - size // 2)))
    rng.shuffle(draws)
    parts = []
    for offset in range(0, size, 500):
        full = wind[draws[offset:offset + 500, None] + np.arange(width)]
        trace = simulate(full, center, rng)
        cutoff = WARMUP + HISTORY
        parts.append((full[:, WARMUP:cutoff].astype("float16"), trace[:, cutoff - 1].astype("float32"),
                      trace[:, cutoff:].min(axis=1).astype("float32"))
                     + ((trace[:, WARMUP:cutoff].astype("float32"),) if with_past else ()))
    keys = ("wind", "dst", "truth") + (("past_dst",) if with_past else ())
    bank = {key: np.concatenate(col) for key, col in zip(keys, zip(*parts))}
    digest = hashlib.sha256()
    for a in bank.values():
        digest.update(a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes())
    label = ("val" if frozen else "train") + str(version) + ("h" if with_past else "")
    file = Path(work("banks", label + ".npz", mkdir=True))
    if file.exists():
        with np.load(file) as z:
            if not all(np.array_equal(z[k], bank[k]) for k in bank):
                raise ValueError(f"Existing {label} differs. Use a new version or a separate work folder.")
    else:
        np.savez_compressed(file, **bank)
    return dict(rows=size, seed=seed, years=years, sha256=digest.hexdigest())


def load_bank(name):
    with np.load(work("banks", name + ".npz")) as z:
        return {k: z[k].astype(float) for k in z}


if __name__ == "__main__":
    cal = read_result("calibration.json")
    log = {}
    for version, center in ((1, PUBLISHED), (2, cal["double"])):
        for frozen in (False, True):
            log[("val" if frozen else "train") + str(version)] = make_bank(version, frozen, center)
    path = HERE / "banks.json"
    if path.exists() and log != json.loads(path.read_text()):
        raise ValueError("Bank fingerprints differ from release")
    write_json(RESULTS / "banks.json", log)
    print(log)
