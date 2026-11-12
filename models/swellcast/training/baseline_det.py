"""Detection with the classical ridge search: its ridge score (mean log-energy anomaly along the best dispersion line).
Threshold = 10% false alarms on the frozen REAL null set (quiet 2024 windows), the same rule the network uses.
Reports detection rate on dev/exam events and the false-alarm rate on held-out quiet 2025 windows."""
import os, json, numpy as np
import classical as CL, evaluate as E

def scores(X): return np.array([CL.read(np.asarray(x, np.float32))['score'] for x in X])

if __name__ == '__main__':
    out = {}; sn = scores(np.load(E.BANK + 'null_val1_x.npy', mmap_mode='r')); thr = float(np.quantile(sn, 0.9)); out['thr'] = thr
    out['quiet_false_alarm'] = float(np.mean(scores(np.load(E.BANK + 'quiet_test_x.npy', mmap_mode='r')) > thr))
    for split, vis in (('dev', 'confirmed'), ('exam', 'confirmed'), ('exam', True), ('exam', None)):
        ev, X = E.real_windows(split, vis); out[f'{split}_{vis}_detect_rate'] = float(np.mean(scores(X) > thr)); print(split, vis, out[f'{split}_{vis}_detect_rate'], flush=True)
    print(out); json.dump(out, open(os.path.join(E.ROOT, 'baseline_det.json'), 'w'), indent=1)
