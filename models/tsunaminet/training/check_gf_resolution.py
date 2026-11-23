"""Resolution check: the released 8' Green's functions vs 4' (dt 10 s, NOAA MOST's propagation-database setup) at
one DART site (46409, Alaska) for every Pacific unit source: correlation in the network's input window and the ratio of
the 3-hour maximum, by source distance."""
import os, json, numpy as np, tsim
from _paths import BANK, DATA
a = np.load(os.path.join(BANK, 'gf4', '46409_0.npz')); b = np.load(os.path.join(BANK, 'gf8', '46409_0.npz'))
A = a['G'].astype(float); B = b['G'].astype(float)
g = tsim.Geometry(); d = tsim.hav(float(a['lat']), float(a['lon']), g.centers[:, 1], g.centers[:, 0])
ma = np.abs(A).max(1); ok = ma > 0.3; ta = np.argmax(np.abs(A) > 0.02 * ma[:, None], 1)
R = np.full(len(A), np.nan); Q = np.full(len(A), np.nan); DT = np.full(len(A), np.nan)
for i in np.nonzero(ok)[0]:
    s = slice(max(ta[i] - 20, 0), ta[i] + 70); R[i] = np.corrcoef(A[i, s], B[i, s])[0, 1]
    s2 = slice(ta[i], ta[i] + 180); Q[i] = np.abs(B[i, s2]).max() / np.abs(A[i, s2]).max()
    DT[i] = np.argmax(np.abs(B[i, s2])) - np.argmax(np.abs(A[i, s2]))
out = []
for lo, hi in [(0, 1500), (1500, 3000), (3000, 6000), (6000, 15000)]:
    m = ok & (d >= lo) & (d < hi)
    out.append(dict(dist_km=[lo, hi], n=int(m.sum()), corr_median=float(np.nanmedian(R[m])), corr_p10=float(np.nanpercentile(R[m], 10)),
                    max3h_ratio_median=float(np.nanmedian(Q[m])), max3h_ratio_iqr=[float(x) for x in np.nanpercentile(Q[m], [25, 75])],
                    log10_ratio_sd=float(np.nanstd(np.log10(Q[m]))), peak_time_shift_median_min=float(np.nanmedian(DT[m]))))
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(DATA, 'checks', 'gf_resolution_check.json'), 'w'), indent=1)
