"""Classical single-station estimates on the same windows, for comparison:
  distance   S - P time (an STA/LTA pick on the horizontals after P) inverted with ak135 for a 33 km deep source
  magnitude  surface-wave Ms (IASPEI): log10(A/T) + 1.66 log10(distance) + 3.3, A the largest 15-30 s vertical displacement
             in the Rayleigh window (micrometres), using that distance
  back-azimuth  P-wave polarization: the principal horizontal direction of the first 15 s of P, the sign set by the vertical
  depth      none: no standard one-station method; the exam compares with guessing the median"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
from scipy.signal import butter, sosfiltfilt
from obspy.signal.trigger import recursive_sta_lta
from obspy.taup import TauPyModel
sys.path.insert(0, _P.SRC); import pipeline as PL
TP = TauPyModel('ak135'); DG = np.arange(20, 100.1, 0.5)
def _first(ph, d):
    a = TP.get_travel_times(33, d, [ph]); return a[0].time if a else np.nan
SP = np.array([_first('S', d) - _first('P', d) for d in DG])
SOS20 = butter(2, [1 / 30, 1 / 15], 'band', fs=1.0, output='sos'); SOSP = butter(2, [0.03, 0.3], 'band', fs=1.0, output='sos')

def one(W, ls):
    X = W.astype(float) * 10 ** ls; P = PL.PRE
    h = np.sqrt(X[1] ** 2 + X[2] ** 2); c = recursive_sta_lta(sosfiltfilt(SOSP, h), 10, 100)
    lo = P + 150; s = lo + int(np.argmax(c[lo:P + 1000])); sp = s - P; ok = np.isfinite(SP)
    dist = float(np.interp(sp, SP[ok], DG[ok])) if SP[ok].min() <= sp <= SP[ok].max() else np.nan
    disp = np.cumsum(X[0]); disp = sosfiltfilt(SOS20, disp - np.mean(disp))
    if np.isfinite(dist):
        tR = P + int(dist * 111.19 / 3.7 - _first('P', dist)); a, b = max(P, tR - 350), min(len(disp), tR + 450)       # Rayleigh group velocity ~3.3-4.2 km/s, relative to P
        A = np.abs(disp[a:b]).max() * 1e6 if b > a else np.nan; ms = float(np.log10(A / 20) + 1.66 * np.log10(dist) + 3.3) if A > 0 else np.nan
    else: ms = np.nan
    z, n, e = (sosfiltfilt(SOSP, x)[P:P + 15] for x in X); Cm = np.cov(np.stack([n, e])); w, v = np.linalg.eigh(Cm); u = v[:, -1]
    if np.dot(u, [np.sum(n * z), np.sum(e * z)]) > 0: u = -u        # P arrives moving away from the source: up with motion away from it
    baz = float(np.degrees(np.arctan2(u[1], u[0])) % 360)
    return dict(dist=dist, mw=ms, baz=baz)

def run(idx):
    z = np.load(_P.WORK + '/data/real.npz'); return [one(z['x'][i], z['ls'][i]) for i in idx]

def score(idx, B, J):
    """errors against the USGS catalog; Ms is offset by its mean bias on the dev events (a fair, calibrated classical magnitude)"""
    y = lambda k: np.array([J['meta'][i][k] for i in idx], float); mw, d, baz = y('mw'), y('dist'), y('baz') if 'baz' in J['meta'][idx[0]] else None
    bd = np.array([b['dist'] for b in B]); bm = np.array([b['mw'] for b in B]); bb = np.array([b['baz'] for b in B]); ok = np.isfinite(bd)
    out = dict(n=len(idx), dist_found=float(ok.mean()), dist_mae=float(np.nanmean(np.abs(bd - d))), ms_minus_mw=float(np.nanmean(bm - mw)))
    if baz is not None:
        e = np.abs((bb - baz + 180) % 360 - 180); out.update(baz_med_deg=float(np.median(e)), baz_within_30=float(np.mean(e < 30)))
    return out, bm - mw

if __name__ == '__main__':
    J = json.load(open(_P.WORK + '/data/real_meta.json')); res = {}
    dev = [i for i, s in enumerate(J['split']) if s == 'dev']; exam = [i for i, s in enumerate(J['split']) if s == 'exam']
    Bd = run(dev); rd, dm = score(dev, Bd, J); sh = np.array([J['meta'][i]['depth'] <= 70 for i in dev]); off = float(np.nanmean(dm[sh]))
    Be = run(exam); re, em = score(exam, Be, J); she = np.array([J['meta'][i]['depth'] <= 70 for i in exam])
    for name, r, m, s in (('dev', rd, dm, sh), ('exam', re, em, she)):
        r['ms_mae_raw'] = float(np.nanmean(np.abs(m))); r['ms_mae_cal'] = float(np.nanmean(np.abs(m - off)))
        r['ms_mae_cal_shallow'] = float(np.nanmean(np.abs(m[s] - off))); r['ms_mae_cal_deep'] = float(np.nanmean(np.abs(m[~s] - off))); res[name] = r
    res['ms_offset_from_dev'] = off; json.dump(dict(res, per_record=dict(exam=Be)), open(_P.WORK + '/baseline_exam.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
