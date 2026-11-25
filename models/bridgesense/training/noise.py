"""Extract real unlabelled healthy OMA frequency noise & missing-mode masks into work/noise_pool.npz.

Uses ONLY the KW51 early healthy autumn baseline window (2018-10-01 .. 2018-11-11, 6 weeks,
strictly pre-winter and 6 months pre-retrofit — zero structural change labels).
Detrends each mode linearly against air temperature within this healthy window, then slices
overlapping 32-step windows of (residual_jitter, valid_mask) for p3 simulator noise injection.
"""
from __future__ import annotations

import numpy as np

from _paths import work
from realdata import N_Q, build_kw51_blocks, load_kw51_hourly


def main() -> None:
    kw = load_kw51_hourly()
    ref_pack, _ = build_kw51_blocks(kw, block_days=7)
    T = ref_pack["T3"][:, 0]     # [H]
    f = ref_pack["f"]            # [H, 14]
    H, M = f.shape

    # Detrend linear thermal slope within healthy autumn reference to isolate pure OMA tracking jitter
    resid = np.zeros((H, M), dtype=np.float32)
    mask = np.zeros((H, M), dtype=np.float32)
    T0 = float(np.mean(T))
    for m in range(M):
        ok = np.isfinite(f[:, m]) & (f[:, m] > 0)
        if np.sum(ok) < 20:
            continue
        med = float(np.median(f[ok, m]))
        df = (f[ok, m] - med) / med
        x = T[ok] - T0
        a = float(np.dot(x, df) / (np.dot(x, x) + 1e-6))
        b = float(np.mean(df) - a * np.mean(x))
        r = df - (a * x + b)
        # Clip extreme >5% tracking mis-associations only for summary diagnostics; keep real glitches <=3%
        r = np.clip(r, -0.03, 0.03)
        resid[ok, m] = r.astype(np.float32)
        mask[ok, m] = 1.0

    # Slice 32-step subsampled windows (matching real block subsampling stride ~5 hours -> 32 steps ~ 7 days)
    win_r = []
    win_m = []
    for stride in (2, 4, 5, 6):
        span = stride * N_Q
        for start in range(0, H - span, 8):
            idx = start + np.arange(N_Q) * stride
            win_r.append(resid[idx])
            win_m.append(mask[idx])

    resid_windows = np.stack(win_r, axis=0).astype(np.float32)
    mask_windows = np.stack(win_m, axis=0).astype(np.float32)

    out_path = work("noise_pool.npz", mkdir=True)
    np.savez_compressed(out_path, resid_windows=resid_windows, mask_windows=mask_windows)
    obs_std = float(np.std(resid[mask > 0.5]))
    valid_rate = float(np.mean(mask))
    print(f"Saved {out_path}: {len(resid_windows)} real OMA noise windows (shape {resid_windows.shape})")
    print(f"  Healthy autumn OMA residual std = {obs_std*100:.3f}%, mean mode availability = {valid_rate*100:.1f}%")


if __name__ == "__main__":
    main()
