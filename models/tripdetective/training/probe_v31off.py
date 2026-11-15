"""Probe (15 Nov 2026): engine-off share while moving of the same generated hybrids under physics v2 and v3.1
(generator hardware as each preset says), through the OBD emulator. usage: probe_v31off.py <n_cars> <seed>"""
from common import *
import copy, fleet, routes, obdlog, priors
from roadsim import simulate
from roadsim.generator import generate_vehicle
from roadsim.options import resolve
n, seed = int(sys.argv[1]), int(sys.argv[2])
res = {}
for phys in ['roadsim-physics-v2', 'roadsim-physics-v3.1']:
    o = resolve(phys); hw = o.get('gen_hybrid_hw', 'v1'); pack = bool(o.get('gen_phev_pack_rule', False))
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        arch = ['hev_powersplit', 'hev_parallel_p2', 'hev_series'][i % 3]
        spec = generate_vehicle(rng, 'car', arch, True, 'v1', hw, pack)
        cfg = obdlog.logger_config(arch, rng); cfg['rpm'] = True
        offs = []
        for k in range(4):
            route, src = routes.draw_route(rng); simr = routes.perturb_for_sim(route, rng)
            lo, hi = spec['battery'].get('soc_window', [0.3, 0.8])
            r = simulate(spec, simr, init_temps=WARM, init_soc=float(rng.uniform(lo + .05, hi - .05)), rng=int(rng.integers(1 << 30)), options=phys)
            v = np.asarray(r['v']); on = np.asarray(r['eng_on']) > .5; mv = v > 5
            if mv.sum() > 10: offs.append(1 - on[mv].mean())
        out.append((arch, spec['engine']['displacement_L'], spec['engine']['bsfc_map'], np.mean(offs)))
    res[phys] = out
for i in range(n):
    a, b = res['roadsim-physics-v2'][i], res['roadsim-physics-v3.1'][i]
    print(f'{a[0]:16s} {a[1]:.1f} L {a[2]:28s} v2 off {a[3]:.3f} | v3.1 {b[2]:28s} off {b[3]:.3f}')
for phys in res:
    x = np.array([r[3] for r in res[phys]]); print(phys, 'median', np.median(x).round(3), 'pct 10/90', np.percentile(x, [10, 90]).round(3))
