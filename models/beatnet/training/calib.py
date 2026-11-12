"""Dev calibration numbers: the same measurements on the real dev records and on a simulated bank.
  t_peak_ms   latency of the T-wave peak after the fiducial (median N window, +150..+550 ms)
  p_amp       P-wave height on the median N window (max in -300..-80 ms minus the level at -330 ms), in units of QRS size
  t_amp       T height on the median window, in units of QRS size
  noise       median |first difference| in -300..-220 ms (per beat), a noise-floor proxy
  v_width     classical QRS width of V beats (ms)"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import bnet as BN, train as TR, common as C, baselines as BL
ms = lambda m: 225 + int(round(m / 1000 * 180))

def measure(W, y):
    out = {}; N = W[y == 0].astype(np.float32); med = np.median(N, 0)
    out['t_peak_ms'] = float((np.argmax(med[ms(150):ms(550)]) + ms(150) - 225) / 180 * 1000); out['t_amp'] = float(med[ms(150):ms(550)].max() - med[ms(-330)])
    out['p_amp'] = float(med[ms(-300):ms(-80)].max() - med[ms(-330)]); out['p_peak_ms'] = float((np.argmax(med[ms(-300):ms(-80)]) + ms(-300) - 225) / 180 * 1000)
    out['noise_med'] = float(np.median(np.abs(np.diff(N[:, ms(-300):ms(-220)], axis=1)))); out['noise_p90'] = float(np.percentile(np.median(np.abs(np.diff(N[:, ms(-300):ms(-220)], axis=1)), 1), 90))
    V = W[y == 2].astype(np.float32)
    if len(V): out['v_width_ms'] = float(np.median(BL.qrs_width(V[:3000]))); mv = np.median(V, 0); out['v_t_amp'] = float(np.abs(mv[ms(150):ms(500)]).max())
    out['n_width_ms'] = float(np.median(BL.qrs_width(N[:3000])))
    return out

if __name__ == '__main__':
    Z = BN.real_bank('mitdb'); _, idx = BN.split_idx('dev'); res = {'dev_records': C.dev_records()}
    per = [measure(Z['w'][idx][Z['rec'][idx] == r], Z['y'][idx][Z['rec'][idx] == r]) for r in C.dev_records()]
    res['dev_per_record'] = per; res['dev'] = {k: float(np.median([p[k] for p in per if k in p])) for k in per[0]}
    import glob; S = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else BN.VAL))]) for k in ('w', 'y', 'info')}
    res['sim'] = measure(S['w'], S['y']); pid = S['info'][:, 2]; pv = []
    for q in np.unique(pid[S['y'] == 2]):
        m = (pid == q) & (S['y'] == 2)
        if m.sum() >= 3: V = S['w'][m].astype(np.float32); mv = np.median(V, 0); pv.append((float(np.median(BL.qrs_width(V))), float(np.abs(mv[ms(150):ms(500)]).max())))
    res['sim']['v_width_ms_per_patient'] = float(np.median([a for a, b in pv])); res['sim']['v_t_amp_per_patient'] = float(np.median([b for a, b in pv]))
    vr = [p for p in per if 'v_width_ms' in p]; res['dev']['v_width_ms'] = float(np.median([p['v_width_ms'] for p in vr])); res['dev']['v_t_amp'] = float(np.median([p['v_t_amp'] for p in vr]))
    print(json.dumps({'dev': res['dev'], 'sim': res['sim']}, indent=1)); json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else _P.WORK + '/calib_v1.json', 'w'), indent=1)
