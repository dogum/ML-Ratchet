"""(15 Nov 2026, round 2) the numpy release module (hybridpilot.py, DEFAULTS = v2.2) against the training policy (hp_policy,
the incumbent's params) on frozen and dev trips: fuel and final SOC must match.
usage: r2_numpy_check.py N_PER_SET   (run from WORK with HP_PHYS set)"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import evaluate as E, sets
from hp_env import PHYS
from roadsim import simulate
import hybridpilot as HP
import subprocess

if __name__ == '__main__':
    n = int(sys.argv[1])
    inc = json.loads(subprocess.check_output([sys.executable, os.path.join(os.path.dirname(__file__), 'incumbent.py'), 'params']))
    ck = subprocess.check_output([sys.executable, os.path.join(os.path.dirname(__file__), 'incumbent.py'), 'ckpt']).decode().strip()
    worst = 0.0; k = 0
    for st in ('dev', 'frozen'):
        cases = sets.get(st)
        rng = np.random.default_rng(0)
        pick = [cases[i] for i in sorted(rng.choice(len(cases), n, replace=False))]
        pick += [c for c in cases if c['spec']['class'] == 'bus'][:4] + [c for c in cases if E.is_mild(c['spec'])][:2]
        pick += [c for c in cases if c['spec']['architecture'].startswith('phev')][:6] + [c for c in cases if c['spec']['architecture'] == 'phev_series'][:6]
        for c in pick:
            nav = c['nav']
            a = simulate(c['spec'], c['trip'], strategy=E.HybridPilot(**E.case_params(c, ck, inc)), init_soc=c['soc0'],
                         init_temps=c['temps'], record=True, options=PHYS)
            b = simulate(c['spec'], c['trip'], strategy=HP.make_roadsim_strategy(nav=nav), init_soc=c['soc0'],
                         init_temps=c['temps'], record=True, options=PHYS)
            d = abs(a['summary']['fuel_g'] - b['summary']['fuel_g']); ds = abs(float(a['soc'][-1]) - float(b['soc'][-1]))
            worst = max(worst, d); k += 1
            print(st, c['case_id'][:48], f"{a['summary']['fuel_g']:.4f} {b['summary']['fuel_g']:.4f} dsoc {ds:.2e}", flush=True)
    print('trips', k, 'worst |fuel diff| g', worst)
