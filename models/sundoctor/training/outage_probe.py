"""Simulation probe (15 Nov 2026): does SunDoctor catch an inverter that trips and stays dark for several days?

The v1/v2 simulators only switch an inverter off for single days (3% of days, independently), so a dark spell of
several days is almost never seen in training (3 days in a row: ~3e-5 per window). This probe simulates fortnights in
which an inverter trips and stays off for 2-14 days (simbatch2.LONG_OUT) and scores them with the rule the real exam
uses for 'inverter offline caught': the mean outage share over the first three fully dark days is above 0.5.

  python outage_probe.py <weights.npz> [n] [seed]       numpy weights (the release, or an exported rung)
Writes work/probes/outage_probe_<name>.json. Simulation only: no real data is read."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import simbatch2 as SB, train2 as T, sundoctor

def probe_set(n=600, seed=4242, D=14):
    fn = f'{_P.WORK}/probes/outage_probe_set_{n}_{seed}.npz'
    if os.path.exists(fn):
        z = np.load(fn); return z['X'], z['Y'], {k[2:]: z[k] for k in z.files if k.startswith('m_')}
    os.makedirs(os.path.dirname(fn), exist_ok=True); SB.LONG_OUT = 1.0; SB.FRONTS = True; rng = np.random.default_rng(seed); Xs, Ys, Ms = [], [], []
    for b in range(n // 100):
        X, Y, G, m = T.batch(rng, 100, D, home=np.arange(100) % 2 == 1); Xs.append(X.astype(np.float32)); Ys.append(Y); Ms.append(m)
    SB.LONG_OUT = 0.0; SB.FRONTS = False
    M = {k: np.concatenate([m[k] for m in Ms]) for k in ('tracker', 'home', 'snow_any')}
    X, Y = np.concatenate(Xs), np.concatenate(Ys); np.savez_compressed(fn, X=X.astype(np.float16), Y=Y, **{'m_' + k: v for k, v in M.items()})
    return X, Y, M

def score(P, Y, M):
    """per fortnight: the first day d with >= 3 fully dark days in a row (outage label > 0.9), d <= D - 3"""
    out_lab = Y[:, SB.DAY_LABELS.index('outage')]; rows = []
    for i in range(len(Y)):
        dark = out_lab[i] > 0.9; d = next((d for d in range(len(dark) - 2) if dark[d:d + 3].all()), None)
        if d is None: continue
        rows.append(dict(i=i, day=d, caught=bool(np.mean(P['outage'][i, d:d + 3]) > 0.5), mean3=float(np.mean(P['outage'][i, d:d + 3])),
                         frac3=float(np.mean(P['frac'][i, d:d + 3])), snow3=float(np.mean(P['snow'][i, d:d + 3])), home=bool(M['home'][i]), tracker=bool(M['tracker'][i])))
    out = {}
    for name, sel in (('all', lambda r: True), ('with_sensor', lambda r: not r['home']), ('with_sensor_fixed', lambda r: not r['home'] and not r['tracker']),
                      ('power_only', lambda r: r['home'])):
        rr = [r for r in rows if sel(r)]
        out[name] = dict(n=len(rr), caught=int(sum(r['caught'] for r in rr)), rate=float(np.mean([r['caught'] for r in rr])) if rr else None,
                         median_outage_pred=float(np.median([r['mean3'] for r in rr])) if rr else None, median_frac_pred=float(np.median([r['frac3'] for r in rr])) if rr else None)
    return out, rows

if __name__ == '__main__':
    w = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 600; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 4242
    X, Y, M = probe_set(n, seed); net = sundoctor.SunDoctorNet(w); P = net.predict(X.astype(np.float32))
    out, rows = score(P, Y, M); print(json.dumps(out, indent=1))
    json.dump(dict(weights=os.path.basename(w), n=n, seed=seed, summary=out, rows=rows), open(f'{_P.WORK}/probes/outage_probe_{os.path.basename(w).replace(".npz", "")}.json', 'w'), indent=1)
