"""Classical Frequency Domain Decomposition (FDD / multi-channel Welch SVD) modal extractor
for the Hell Bridge Test Arena (HBTA, Zenodo 14632942, CC BY 4.0).

Uses the 18 global truss accelerometers (AG01..AG18, y and z channels = 36 channels) at 100 Hz.
Reference modal peak bands are identified strictly from the five first-replicate undamaged
recordings (UDS *_01), which Exam C uses as the reference window. The second-replicate UDS *_02
recordings are held-out healthy controls in Exam C and play no part in choosing the bands (v1.1 fix;
v1.0 used all ten UDS recordings). The bands are frozen in training/hbta_bands.json, then tracked across overlapping 120-s segments of every recording (UDS + DS1..DS8) using
parabolic sub-bin interpolation of the first singular value of the cross-spectral density matrix.

Saves work/hbta_modes.npz for Exam C (opened once by exam.py after the ratchet closes).
"""
from __future__ import annotations

import json
import os
import time
import h5py
import numpy as np
from scipy.signal import csd, find_peaks

from _paths import SRC, work

GLOBAL_SENSORS = [f"AG{i:02d}" for i in range(1, 19)]
BANDS_PATH = os.path.join(SRC, "hbta_bands.json")


def is_reference_recording(name: str, state: str) -> bool:
    """Exam C reference pool: undamaged first replicates only (same rule as exam.load_hbta_exam_episodes)."""
    return state == "UDS" and name.endswith("_01")


def _load_global_matrix(rec_group: h5py.Group) -> np.ndarray:
    """Load all available AG01..AG18 y/z acceleration channels as [C, T] float64."""
    acc = rec_group["acceleration"]
    cols = []
    for s in GLOBAL_SENSORS:
        if s not in acc:
            continue
        for ch in ("y", "z"):
            if ch in acc[s]:
                x = np.asarray(acc[s][ch][:], dtype=np.float64)
                x = x - np.mean(x)
                cols.append(x)
    min_len = min(len(c) for c in cols)
    mat = np.stack([c[:min_len] for c in cols], axis=0)
    return mat


def _fdd_sv1(mat: np.ndarray, fs: float = 100.0, nperseg: int = 4096,
             f_lo: float = 2.0, f_hi: float = 22.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute first singular value spectrum SV1(f) of the multi-channel Welch PSD/CSD matrix.

    Using fast segment FFT average: G(f) = (1/K) sum_k X_k(f) X_k(f)^H -> sv1(f) = max_eig(G(f)).
    """
    C, T = mat.shape
    step = nperseg // 2
    n_seg = max(1, (T - nperseg) // step + 1)
    win = np.hanning(nperseg)
    win_norm = float(np.sum(win ** 2)) * fs
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    f_band = freqs[band]
    n_f = len(f_band)

    # Accumulate G(f) [n_f, C, C] using vectorized segment FFTs
    G = np.zeros((n_f, C, C), dtype=np.complex128)
    for k in range(n_seg):
        s0 = k * step
        seg = mat[:, s0:s0 + nperseg] * win[None, :]
        X = np.fft.rfft(seg, axis=1)[:, band].T  # [n_f, C]
        G += X[:, :, None] * np.conj(X[:, None, :])
    G /= (n_seg * win_norm)
    # Top singular value = top eigenvalue of Hermitian PSD matrix
    evals = np.linalg.eigvalsh(G)
    sv1 = np.maximum(evals[:, -1], 1e-20)
    return f_band, sv1


def _parabolic_peak(freqs: np.ndarray, log_sv: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(freqs) - 1:
        return float(freqs[idx])
    y0, y1, y2 = log_sv[idx - 1], log_sv[idx], log_sv[idx + 1]
    den = (y0 - 2.0 * y1 + y2)
    if abs(den) < 1e-12:
        return float(freqs[idx])
    delta = np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)
    df = freqs[1] - freqs[0]
    return float(freqs[idx] + delta * df)


def extract_all() -> str:
    dst = work("hbta_modes.npz", mkdir=True)
    h5_path = work("raw", "hbta", "data_100Hz.h5")
    t0 = time.time()

    with h5py.File(h5_path, "r") as hf:
        keys = sorted(hf.keys())
        # Step 1: Discover global modal peak centres from the UDS *_01 reference recordings only
        uds_keys = [k for k in keys if is_reference_recording(k, str(hf[k].attrs.get("damage_state", "")))]
        held_out = [k for k in keys if hf[k].attrs.get("damage_state", "") == "UDS" and k not in uds_keys]
        print(f"[HBTA] band discovery on {len(uds_keys)} reference recordings {uds_keys}; "
              f"{len(held_out)} UDS controls held out {held_out}")
        sv_sum = None
        f_ref_grid = None
        for k in uds_keys:
            mat = _load_global_matrix(hf[k])
            fb, sv1 = _fdd_sv1(mat, fs=100.0, nperseg=4096, f_lo=2.0, f_hi=22.0)
            log_sv = np.log10(sv1)
            log_sv = (log_sv - np.median(log_sv)) / (np.std(log_sv) + 1e-9)
            sv_sum = log_sv if sv_sum is None else (sv_sum + log_sv)
            f_ref_grid = fb
        sv_uds_mean = sv_sum / len(uds_keys)
        # Pick up to 8 well-separated prominent global modes (minimum 0.6 Hz spacing)
        min_dist = int(round(0.6 / (f_ref_grid[1] - f_ref_grid[0])))
        pks, props = find_peaks(sv_uds_mean, distance=min_dist, prominence=0.6)
        order = np.argsort(props["prominences"])[::-1][:8]
        peak_freqs = np.sort(f_ref_grid[pks[order]])
        if os.path.exists(BANDS_PATH):
            frozen = json.load(open(BANDS_PATH, encoding="utf-8"))
            if frozen["reference_recordings"] != uds_keys or not np.allclose(frozen["peak_freqs_hz"], peak_freqs):
                raise SystemExit(f"{BANDS_PATH} disagrees with a fresh discovery; delete it only on purpose")
        else:
            with open(BANDS_PATH, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"source": "Welch-SVD FDD peaks on HBTA UDS *_01 recordings (Zenodo 14632942, CC BY 4.0)",
                           "reference_recordings": uds_keys, "held_out_uds": held_out,
                           "peak_freqs_hz": [round(float(x), 6) for x in peak_freqs]}, f, indent=2)
                f.write("\n")
        print(f"[HBTA UDS] Found {len(peak_freqs)} global reference modes (Hz): {[round(float(x), 3) for x in peak_freqs]}")

        # Step 2: Track those modes across overlapping 120-s windows of every recording
        rec_names, states, dates, temps, modes_list = [], [], [], [], []
        for k in keys:
            grp = hf[k]
            mat = _load_global_matrix(grp)
            fs = float(grp.attrs.get("sample_rate", 100.0))
            T_len = mat.shape[1]
            win_len = min(T_len, int(120 * fs))
            n_sub = max(1, min(8, (T_len - win_len) // int(60 * fs) + 1))
            starts = np.linspace(0, max(0, T_len - win_len), n_sub).round().astype(int)

            seg_freqs = np.full((n_sub, len(peak_freqs)), np.nan, dtype=np.float64)
            for si, s0 in enumerate(starts):
                fb, sv1 = _fdd_sv1(mat[:, s0:s0 + win_len], fs=fs, nperseg=4096, f_lo=2.0, f_hi=22.0)
                lsv = np.log10(sv1)
                for m, f0 in enumerate(peak_freqs):
                    # Search within ±6% around UDS mode frequency (captures severe damage drops up to -12% dK)
                    hw = max(0.35, 0.06 * f0)
                    sel = np.where((fb >= f0 - hw) & (fb <= f0 + hw))[0]
                    if len(sel) < 3:
                        continue
                    loc_i = sel[int(np.argmax(lsv[sel]))]
                    # Quality check: peak must exceed band median by >= 0.25 log10 units
                    if lsv[loc_i] - np.median(lsv[sel]) >= 0.25:
                        seg_freqs[si, m] = _parabolic_peak(fb, lsv, loc_i)

            rec_names.append(k)
            states.append(str(grp.attrs.get("damage_state", "")))
            dates.append(str(grp.attrs.get("date", "")))
            temps.append(float(grp.attrs.get("temperature", 10.0)))
            modes_list.append(seg_freqs)

    # Pad variable segment counts to max n_sub (usually 8)
    max_seg = max(a.shape[0] for a in modes_list)
    M = len(peak_freqs)
    f_all = np.full((len(rec_names), max_seg, M), np.nan, dtype=np.float64)
    for i, a in enumerate(modes_list):
        f_all[i, :a.shape[0], :] = a

    np.savez_compressed(
        dst,
        rec_names=np.array(rec_names),
        states=np.array(states),
        dates=np.array(dates),
        temps=np.array(temps, dtype=np.float64),
        peak_freqs_uds=peak_freqs,
        freqs=f_all,
    )
    print(f"Saved {dst}: {len(rec_names)} recordings x {max_seg} windows x {M} modes in {time.time()-t0:.1f}s")
    return dst


if __name__ == "__main__":
    extract_all()
