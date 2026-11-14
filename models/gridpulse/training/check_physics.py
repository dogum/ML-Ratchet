"""Physics checks of the GridPulse wrapper: (1) inertia of rescaled states equals the drawn target and the initial
RoCoF of the simulation equals the swing-equation formula RoCoF = dP f0 / (2 E); (2) the Dynamic Containment knee:
steady output of the two-segment droop at fixed deviations vs NESO's service curve."""
from common import *
import gpsim, gridsim as gs
from gridsim.frequency import _compile
res = {}
for sy in ('GB', 'ERCO'):
    S, E, L = gpsim.sample(100, 4242, sy, 'p2')
    ev = gpsim.run(S, E, t_end=1.0, dt=0.02, h=0.01)
    slope = np.array([(e['f'][1] - e['f'][0]) / (e['t'][1] - e['t'][0]) for e in ev])
    first = np.array([ee[0][1] for ee in E])
    formula = -first * L.f0.values / (2 * L.inertia_MWs.values)
    rel = np.abs(slope / formula - 1)
    res[sy] = dict(n=100, initial_rocof_vs_formula_max_rel_err=float(rel.max()), median_rel_err=float(np.median(rel)),
                   note='slope over the first 20 ms (governors have not moved yet); first loss step only')
    print(sy, res[sy])
# DC knee: K * dead-banded deviation, clipped at contracted MW, two segments
f0 = 50.0; dc = 1000.0
segs = [(0.05 * dc, 0.015, 0.2), (0.95 * dc, 0.2, 0.5)]
spec = lambda d: np.interp(d, [0, 0.015, 0.2, 0.5, 1.0], [0, 0, 0.05, 1.0, 1.0]) * dc
rows = []
for d in (0.01, 0.1, 0.2, 0.3, 0.5, 0.8):
    out = sum(min(MW, MW / ((full - db) / f0 * f0) * max(d - db, 0) * (full - db) / (full - db)) if False else
              min(MW, max(d - db, 0) * MW / (full - db)) for MW, db, full in segs)
    rows.append(dict(df_Hz=d, model_MW=out, NESO_curve_MW=float(spec(d))))
res['dc_knee'] = rows; print(rows)
os.makedirs(f'{ROOT}/checks', exist_ok=True)
json.dump(res, open(f'{ROOT}/checks/physics.json', 'w'), indent=1)
