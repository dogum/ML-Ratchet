"""Surface-wave timing on the dev events: real surface waves arrive later than in the 1-D ak135 model (slower crust and
oceans along real paths). For each dev record of a shallow M6+ event with a USGS moment tensor, find the time warp after
the S arrival that best lines up the synthetic's envelope (0.01-0.05 Hz, vertical) with the recording's."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import compare as CP, pipeline as PL
from scipy.signal import butter, sosfiltfilt
from obspy.taup import TauPyModel
TP = TauPyModel('ak135'); D = _P.WORK + '/data/'; SOS = butter(2, [0.01, 0.05], 'band', fs=1.0, output='sos')

def warp(x, tS, w):
    t = np.arange(len(x)); src = np.where(t < tS, t, tS + (t - tS) / w); return np.interp(src, t, x, left=0, right=0)

def env(x): return np.convolve(np.abs(sosfiltfilt(SOS, x)), np.ones(41) / 41, 'same')

if __name__ == '__main__':
    J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz'); W = np.linspace(0.85, 1.5, 66); out = []
    for i, (m, s) in enumerate(zip(J['meta'], J['split'])):
        if s != 'dev' or m['mw'] < 5.9 or m['depth'] > 100: continue
        M, _ = CP.usgs_mt(m['id'])
        if M is None: continue
        X = CP.synth_for(m, M)[0]; R = z['x'][i].astype(float)[0]; L = min(len(X), len(R)); X, R = X[:L], R[:L]
        tp = TP.get_travel_times(m['depth'], m['dist'], ['P'])[0].time; ts = TP.get_travel_times(m['depth'], m['dist'], ['S'])[0].time; tS = PL.PRE + int(ts - tp)
        er = env(R); c = [np.corrcoef(env(warp(X, tS, w)), er)[0, 1] for w in W]; k = int(np.argmax(c))
        out.append(dict(i=i, id=m['id'], mw=m['mw'], dist=m['dist'], station=m['station'], w=float(W[k]), cc=float(c[k]), cc_1=float(c[np.argmin(np.abs(W - 1))])))
        print(out[-1], flush=True)
    w = np.array([o['w'] for o in out if o['cc'] > 0.5]); json.dump(dict(records=out, w_median=float(np.median(w)), w_p16=float(np.percentile(w, 16)), w_p84=float(np.percentile(w, 84)), n=len(w)), open(D + 'sw_calibration.json', 'w'), indent=1)
    print('warp after S: median %.3f, 16-84%% %.3f-%.3f, n %d (cc > 0.5)' % (np.median(w), np.percentile(w, 16), np.percentile(w, 84), len(w)))
