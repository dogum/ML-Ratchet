"""Build the FirePilot showpiece page (web/firepilot.html) from the exam outputs.

  python3 build_page.py <final_model> [exam_dir]

Embeds: exam summary, fidelity, ratchet, the value net, and a seeded draw of exam fires (stratified by size tercile,
never hand-picked) with their planner inputs (gzip + base64) so the network plans live in the browser (web/fp.js).
Also writes web/page_check.json with Python features for the embedded inputs; check_js.js compares them.
"""
import os
import sys
import json
import gzip
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'training'))
import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
import numpy as np

from fp.planner import Context, ValueNet, BASELINES, sample_plan
from fp.core import GROUPS, COST_MULT, FL_HOLD_M

ROOT = _paths.WORK
WEB = os.path.join(ROOT, 'web')   # outputs; the sources (template.html, fp.js) are read from HERE
GCODES = ['NB', 'GR', 'GS', 'SH', 'TU', 'TL', 'SB', 'A_grass', 'A_shrub', 'A_timber', 'A_slash']
METHODS = ['firepilot', 'anchor_flank', 'indirect_1km', 'indirect_2km', 'direct_head', 'r0', 'mpc_forecast', 'pf_bound']


def b64gz(a):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()


def gcode(fuel):
    out = np.zeros(fuel.shape, np.uint8)
    for k, g in enumerate(GCODES[1:], 1):
        out[np.isin(fuel, list(GROUPS[g]))] = k
    return out


def quantized_inputs(z, rate_w):
    """The page's inputs, and the same values as Python arrays for the reference features."""
    arr = z['arr_f']
    aq = np.where(np.isfinite(arr), np.minimum(np.round(arr * 16), 65534), 65535).astype(np.uint16)
    flq = np.minimum(np.round(z['fl_f'] * 20), 255).astype(np.uint8)
    g = gcode(z['fuel'])
    costv = np.array([np.float32(COST_MULT[c]) for c in GCODES], np.float32)
    W = rate_w.astype(np.float32)
    py = dict(arr=np.where(aq == 65535, np.nan, aq / 16.0), fl=flq * 0.05, cost=costv[g], W=W.astype(np.float64))
    return dict(aq=aq, flq=flq, g=g, W=W), py


def pick_fires(recs, n=6, seed=20261122):
    """Seeded draw, 2 per size tercile of the detected area (no hand-picking); windows above 260k cells are skipped
    for page size, and the skip is recorded."""
    ok = [r for r in recs if 'skip' not in r and r.get('stage_b')]   # page shows MPC / PF rows too
    a0 = np.array([r['a0'] for r in ok])
    q = np.quantile(a0, [1 / 3, 2 / 3])
    rs = np.random.default_rng(seed)
    out, skipped = [], []
    for t in range(3):
        pool = [r for r, a in zip(ok, a0) if (a > q[t - 1] if t else True) and (a <= q[t] if t < 2 else True)]
        order = rs.permutation(len(pool))
        got = 0
        for i in order:
            r = pool[i]
            if r['shape'][0] * r['shape'][1] > 260000:
                skipped.append(r['fire_id'])
                continue
            out.append(r)
            got += 1
            if got == n // 3:
                break
    return out, skipped


def main():
    final = sys.argv[1]
    exd = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results', 'exam')
    import firedata as fd
    ft = fd.fires('exam').set_index('Event_ID')
    from exam import merged
    recs = [merged(f[:-5], exd) for f in sorted(os.listdir(exd)) if f.endswith('.json') and not f.endswith('.B.json')]
    summ = json.load(open(_paths.latest('results/exam_summary.json'))) if os.path.exists(
        _paths.latest('results/exam_summary.json')) else {}
    chosen, skipped = pick_fires(recs) if len(recs) > 3 else (recs, [])
    net = ValueNet(os.path.join(ROOT, 'models', final + '.npz'))
    zn = np.load(os.path.join(ROOT, 'models', final + '.npz'))
    nm = int(zn['n_members']) if 'n_members' in zn.files else 0
    sfx = [f'_{k}' for k in range(nm)] if nm else ['']
    netj = dict(mu=zn['mu'].tolist(), sd=zn['sd'].tolist(), sig_scale=float(zn['sig_scale']) if 'sig_scale' in zn.files else 1.0,
                members=[dict(W=[zn[f'W{i}{s}'].ravel().tolist() for i in range(3)],
                              b=[zn[f'b{i}{s}'].tolist() for i in range(3)]) for s in sfx])
    fires, check = [], []
    for r in chosen:
        fid = r['fire_id']
        z = np.load(os.path.join(exd, fid + '_ctx.npz'))
        m300 = np.load(os.path.join(exd, f'{fid}_300_masks.npz'))
        q, py = quantized_inputs(z, m300['clock_w'])
        ny, nx = z['burned0'].shape
        meta = ft.loc[fid] if fid in ft.index else None
        F = dict(id=fid, name=str(meta.Incid_Name).title() if meta is not None else fid,
                 period=str(meta.exam_period) if meta is not None else 'dev', region=str(meta.region) if meta is not None else '',
                 year=int(meta.year) if meta is not None else 0, ny=ny, nx=nx, res=90.0, head=float(z['head']), k=r['k'],
                 a0_ha=r['a0'], hidden=r['hidden'], budgets={}, fidelity=r['fidelity']['runs'],
                 obs_h=z['obs_h'].tolist())
        F['blob'] = dict(g=b64gz(q['g']), b0=b64gz(z['burned0'].astype(np.uint8)), aq=b64gz(q['aq']), flq=b64gz(q['flq']),
                         W=b64gz(q['W']), none72=b64gz(z['none72']), over=b64gz(np.packbits(z['fl_t'] > FL_HOLD_M)),
                         obs=[b64gz(o) for o in z['obs']] if len(z['obs']) else [])
        F['nT'] = int(len(z['clock_t']))
        for b, rr in r['budgets'].items():
            mz = np.load(os.path.join(exd, f'{fid}_{b}_masks.npz'))
            F['budgets'][b] = dict(area_none=rr['area_none'], pred=rr['firepilot_pred'], sigma=rr['firepilot_sigma'],
                                   rows={k: dict(saved=v['saved'], area=v['area'], km=v['km'], plan=v['plan'],
                                                 n_overrun=v['n_overrun']) for k, v in rr['rows'].items()},
                                   masks={k: b64gz(mz[k]) for k in mz.files if k != 'clock_w'})
        fires.append(F)
        # Python reference on the quantized inputs
        ctx = Context(z['burned0'], py['arr'], py['fl'], py['cost'].astype(np.float32), 300.0,
                      (z['clock_t'].astype(np.float64), py['W']), 90.0)
        ctx.head = float(z['head'])
        from fp.core import Frame
        ctx.fr = Frame(ctx.burned0, 90.0, ctx.head)
        rs = np.random.default_rng(1)
        plans = list(BASELINES.values()) + [sample_plan(rs) for _ in range(6)]
        X = np.array([ctx.features(p)[0] for p in plans])
        m, s = net(X)
        check.append(dict(id=fid, plans=plans, X=X.tolist(), m=m.tolist(), s=s.tolist()))
    # page view of the summary: stage-A rows per budget|period, with the stage-B methods (MPC, PF bound) added
    if summ.get('planner'):
        pv = {}
        for key, row in summ['planner'].items():
            sub, bk, per = key.split('|')
            if sub != 'A':
                continue
            r = dict(row)
            rb = summ['planner'].get(f'B|{bk}|{per}')
            if rb:
                for m in ('mpc_forecast', 'pf_bound'):
                    r[m] = rb[m]
                for k2 in ('firepilot_beats_mpc_forecast', 'firepilot_beats_pf_bound', 'firepilot_share_of_pf'):
                    r[k2] = rb.get(k2)
                r['nB'] = rb['n']
                r['firepilot_on_B'] = rb['firepilot']
            pv[f'{bk}|{per}'] = r
        summ = dict(summ, planner=pv)
    data = dict(fires=fires, net=netj, summary=summ, skipped_for_size=skipped,
                ratchet=json.load(open(_paths.latest('ratchet.json'))),
                fid_dev=json.load(open(_paths.latest('results/fidelity_dev_p1.json')))['summary'],
                fid_dev_p0=json.load(open(_paths.latest('results/fidelity_dev_p0.json')))['summary'],
                extra=json.load(open(_paths.latest('results/page_extra.json'))) if os.path.exists(
                    _paths.latest('results/page_extra.json')) else {})
    json.dump(check, open(os.path.join(WEB, 'page_check.json'), 'w'))
    tpl = open(os.path.join(HERE, 'template.html')).read()
    js = open(os.path.join(HERE, 'fp.js')).read()
    html = tpl.replace('/*FPJS*/', js).replace('/*DATA*/', 'const DATA = ' + json.dumps(data, separators=(',', ':')) + ';')
    open(os.path.join(WEB, 'firepilot.html'), 'w').write(html)
    json.dump(data, open(os.path.join(WEB, 'page_data.json'), 'w'), separators=(',', ':'))
    print('page', len(html) // 1024, 'KB', [f['name'] for f in fires], 'skipped', skipped)


if __name__ == '__main__':
    main()
