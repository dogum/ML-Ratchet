"""Dev calibration: compare REAL dev swell arrivals with simulated ones, aligned on the true dispersion ridge.
For each window: anomaly A = log10 E - (25th percentile of log10 E over the window, per frequency); each frequency row is
shifted so that lag 0 = predicted arrival time of that frequency from the true (D, t0). Averaging gives a ridge-aligned
stack (frequency x lag). Also: r1 along the ridge, the ridge's peak frequency, the lowest frequency with any energy, and
the ridge contrast. Uses dev events only (and the simulated validation bank)."""
import os, sys, json, numpy as np
import swellsim as SS, events as EV, evaluate as EVL
TC = (np.arange(SS.NT) * SS.DT_H - SS.HALF_H + SS.DT_H / 2); LAGS = np.arange(-72, 73, 6.0)

def stack_one(x, t0, D):
    E = x[0].astype(np.float64); m = np.isfinite(E); L = np.log10(np.maximum(np.where(m, E, 0.005), 0.005)); L[~m] = np.nan
    A = L - np.nanpercentile(L, 25, axis=0)[None]; out = np.full((34, len(LAGS)), np.nan)
    r1 = np.sqrt(x[1].astype(np.float64) ** 2 + x[2].astype(np.float64) ** 2); R = np.full((34, len(LAGS)), np.nan)
    for j, f in enumerate(SS.FGRID):
        ta = t0 + D * 1e3 / SS.cg(f) / 3600
        for k, lg in enumerate(LAGS):
            i = int(np.round((ta + lg - TC[0]) / SS.DT_H))
            if 0 <= i < SS.NT: out[j, k] = A[i, j]; R[j, k] = r1[i, j]
    lowest = np.array([SS.FGRID[np.argmax(np.nan_to_num(E[i]) >= 0.005)] if np.isfinite(E[i]).any() else np.nan for i in range(SS.NT)])
    return out, R, np.nanmedian(lowest)

def summary(S, R, low, con):
    S = np.array(S); R = np.array(R); ms = np.nanmean(S, 0); mr = np.nanmean(R, 0); k0 = np.argmin(np.abs(LAGS))
    band = (SS.FGRID >= 0.05) & (SS.FGRID <= 0.08)
    prof = np.nanmean(ms[band], 0); half = prof.max() / 2; above = LAGS[prof >= half]
    peakf = SS.FGRID[np.nanargmax(np.nanmean(ms[:, k0 - 1:k0 + 2], 1))]
    return dict(n=len(S), ridge_anom_at_0=float(prof[k0]), ridge_fwhm_h=float(above.max() - above.min() + 6) if len(above) else np.nan,
                peak_f_on_ridge=float(peakf), r1_on_ridge=float(np.nanmean(mr[band][:, k0 - 1:k0 + 2])), r1_off_ridge=float(np.nanmean(mr[band][:, :3])),
                lowest_f_median=float(np.nanmedian(low)), contrast_median=float(np.nanmedian(con)), profile=prof.round(3).tolist(), stack=np.round(ms, 3).tolist())

if __name__ == '__main__':
    out = {}; ev, X = EVL.real_windows('dev', 'confirmed'); S, R, low = [], [], []
    for e, x in zip(ev, X): s, r, l = stack_one(x, e['t0_rel'], e['D'] / 1e3); S.append(s); R.append(r); low.append(l)
    out['dev_real'] = summary(S, R, low, [e['contrast'] for e in ev])
    for vb in sys.argv[1:] or ['val1']:
        Xv, Yv = EVL.val_set(vb); idx = np.where((Yv[:, 4] >= 0.3) & (Yv[:, 0] >= 1000) & (Yv[:, 0] <= 8000))[0][:400]
        S, R, low = [], [], []
        for i in idx: s, r, l = stack_one(np.asarray(Xv[i], np.float32), Yv[i, 1], Yv[i, 0]); S.append(s); R.append(r); low.append(l)
        out['sim_' + vb] = summary(S, R, low, Yv[idx, 4])
    for k, v in out.items(): print(k, {a: b for a, b in v.items() if a not in ('stack',)})
    json.dump(dict(lags=LAGS.tolist(), f=SS.FGRID.tolist(), **out), open(os.path.join(EVL.ROOT, 'calib_dev.json'), 'w'))
