"""Simulation-side check for rung v2-r14 (no real data): how much energy the bank's old SC03 / 20 F cold-FTP draws
missed against the EPA test conditions (dynoproc.py), on light-duty vehicles of the training fleet.
python check_dynoproc.py [n vehicles]   -> work/check_dynoproc.json"""
import _paths as _P
import sys, json, glob
sys.path.insert(0, _P.SRC)
from common import *
import numpy as np
from roadsim.vehicle import fill_spec
import bank as BK, physics as PHY, dynoproc as DP


def main(n=40):
    cy = BK.cycles()
    V = []
    for cp in sorted(glob.glob(f'{EN}/data/fleet_train2/chunk_*.json')):
        V += [x for x in json.load(open(cp)) if x['spec']['class'] not in HD]
        if len(V) >= 4 * n: break
    rng = np.random.default_rng(5)
    V = [V[i] for i in rng.choice(len(V), n, replace=False)]
    rows = []
    for x in V:
        spec = x['spec']; arch = spec['architecture']; full = fill_spec(spec)
        b = full.get('battery') or {}; lo, hi = (b.get('soc_window') or [0.2, 0.9])
        soc = None if arch == 'conv' else 0.5 * (lo + hi)
        r = dict(arch=arch, cls=spec['class'])
        for name, T in (('SC03', 35.0), ('FTP_cold_20F', -6.7)):
            tr = BK._fill(cy[name]); N = len(tr['v'])
            import routes
            tr = dict(tr); tr.update(T_amb=np.full(N, T, np.float32), wind_speed=np.zeros(N, np.float32), grade=np.zeros(N, np.float32),
                                     rho_air=routes.rho_air(np.full(N, T), np.zeros(N)).astype(np.float32))
            temps = warm_temps(T) if name == 'SC03' else None
            o_old = PHY.sim(spec, tr, dyno=True, hvac=True, init_soc=soc, init_temps=temps, record=True, rng=1)
            o_new = DP.run(spec, tr, name, soc, temps, 1)
            e_old = PHY.targets(o_old, spec, tr['T_amb'], True)[0]; e_new = PHY.targets(o_new, spec, tr['T_amb'], True)[0]
            r[name] = dict(old_MJ=e_old, new_MJ=e_new, ratio=e_new / e_old,
                           hvac_old_W=float(np.mean(o_old['hvac_power'])), hvac_new_W=float(np.mean(o_new['hvac_power'])))
        rows.append(r); print(r['arch'], r['cls'], {k: round(r[k]['ratio'], 3) for k in ('SC03', 'FTP_cold_20F')}, flush=True)
    summ = {}
    for name in ('SC03', 'FTP_cold_20F'):
        for g in ('ICE', 'HEV', 'PHEV', 'EV'):
            q = [r[name]['ratio'] for r in rows if arch_group(r['arch']) == g]
            if q: summ[f'{name}_{g}'] = dict(n=len(q), median_ratio=float(np.median(q)), p25=float(np.quantile(q, 0.25)), p75=float(np.quantile(q, 0.75)))
    json.dump(dict(rows=rows, summary=summ), open(f'{EN}/check_dynoproc.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
