"""Export FirePilot's value network, the three example fires and Python reference cases for the browser instrument.

Writes site/public/instruments/firepilot/:
  weights.{json,bin}   the released value network: W0-W2, b0-b2 as float32 (as stored; 6,338 weights, 25 KB). The
                       feature scaling mu, sd (float64 in the npz) and sig_scale go in the JSON meta, so the browser
                       standardises with the same float64 numbers as Python.
  fires.{json,bin}     the three exam fires of examples/: detected area (u8), forecast arrival time and flame length
                       (float32, as stored), the line-cost class per cell (u8; the float32 cost per class is in the
                       meta) and the crew clock (float32); the exam's plans and simulated truth rows in the meta
  reference.json       firepilot.py's own features (31), the network's mean and sigma, km of line built, the
                       connectivity proxy and the number of line cells, for 10 plans on each fire at the two exam
                       budgets and one budget the exam never used (650 m/h, the perturbed variant)

Run from the repository root:  python models/firepilot/web/export_instrument.py
"""
import json
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[1]
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(REPO))
from mlratchet.webexport import instrument_dir, write_json, write_pack  # noqa: E402
import firepilot as fp  # noqa: E402

BUDGETS = [('300', 300.0), ('1000', 1000.0), ('650 (not an exam budget)', 650.0)]


def fixed_plans(rec):
    """The 5 baseline plans, the exam's FirePilot plans at both budgets and 3 prior draws (seed 1)."""
    plans = [(k, dict(v)) for k, v in fp.BASELINES.items()]
    plans += [(f'firepilot exam plan, {b} m/h', dict(rec['budgets'][b]['firepilot_plan'])) for b in ('300', '1000')]
    rs = np.random.default_rng(1)
    plans += [(f'prior draw {i + 1}', fp.sample_plan(rs)) for i in range(3)]
    return plans


def outputs(ctx, net, plan):
    x, pe = ctx.features(plan)
    m, s = net(x[None, :])
    n_line = len(fp.line_cells(ctx.fr, plan, None)[0])
    return [float(v) for v in x] + [float(m[0]), float(s[0]), float(pe['km']), float(pe['s_proxy']), float(n_line)]


if __name__ == '__main__':
    out = instrument_dir('firepilot')
    z = np.load(MODEL / 'firepilot_weights.npz')
    assert 'n_members' not in z.files                       # the released r1 is a single network
    W = {k: np.asarray(z[k], np.float32) for k in ('W0', 'W1', 'W2', 'b0', 'b1', 'b2')}
    assert all(np.array_equal(W[k], z[k]) for k in W)     # stored as float32: the pack is exact
    write_pack(out / 'weights', W, dtype='f32',
               meta=dict(mu=z['mu'].tolist(), sd=z['sd'].tolist(), sig_scale=float(z['sig_scale']), version='1.1',
                         params=int(sum(v.size for v in W.values()))))
    net = fp.ValueNet(str(MODEL / 'firepilot_weights.npz'))

    names = sorted(p.stem for p in (MODEL / 'examples').glob('*.json'))
    costs = sorted({float(v) for g, v in fp.COST_MULT.items()})
    arrays, fires, reference = {}, [], []
    for i, name in enumerate(names):
        rec = json.loads((MODEL / 'examples' / f'{name}.json').read_text(encoding='utf-8'))
        e = np.load(MODEL / 'examples' / f'{name}.npz')
        cost = e['cost']
        code = np.full(cost.shape, 255, np.uint8)
        for k, c in enumerate(costs):
            code[cost == np.float32(c)] = k
        assert (code < 255).all()
        arrays.update({f'b{i}': e['burned0'].astype(np.uint8), f'a{i}': e['arr_f'], f'f{i}': e['fl_f'],
                       f'c{i}': code, f't{i}': e['clock_t'], f'w{i}': e['clock_w']})
        ny, nx = e['burned0'].shape
        fires.append(dict(id=rec['fire_id'], name=rec['fire_name'], period=rec['period'], k=rec['k'], res=rec['res'],
                          head=rec['head'], ny=ny, nx=nx, a0_ha=rec['a0_ha'], forecast_none_ha=rec['forecast_none_ha'],
                          budgets=rec['budgets']))
        for label, rate in BUDGETS:
            ctx = fp.Context.from_store(e, dict(rate=rate, res=rec['res'], head=rec['head']))
            for pname, plan in fixed_plans(rec):
                reference.append(dict(example=i, variant=f'{rate:.0f} m/h · {pname}', fire=i, rate=rate, plan=plan,
                                      outputs=outputs(ctx, net, plan)))
        print(f"{name} {rec['fire_name']}: {ny} x {nx} cells, {len(BUDGETS) * len(fixed_plans(rec))} cases")

    summ = json.loads((MODEL / 'training' / 'results' / 'exam_summary.json').read_text(encoding='utf-8'))['planner']
    exam = {b: {k: summ[f'B|{b}|all'][k]['saved_mean'] for k in
                ('direct_head', 'indirect_1km', 'indirect_2km', 'anchor_flank', 'r0', 'mpc_forecast', 'firepilot', 'pf_bound')}
            for b in ('300', '1000')}
    write_pack(out / 'fires', arrays, dtype='f32', keep={k: 'u8' for k in arrays if k[0] in 'bc'},
               meta=dict(fires=fires, cost_classes=[float(np.float32(c)) for c in costs], exam_means=exam,
                         horizon_h=fp.HORIZON_H, fl_hold_m=fp.FL_HOLD_M))
    for stale in ('examples.json', 'examples.bin'):
        (out / stale).unlink(missing_ok=True)
    write_json(out / 'reference.json', dict(cases=reference, tolerance=1e-6))
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f'{len(names)} fires, {len(reference)} parity cases, {size / 1e6:.2f} MB')
