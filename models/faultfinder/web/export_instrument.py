"""Export FaultFinder's live instrument for the website: the network in float16, twelve real exam days as raw
[13, 96] trend-log arrays, and reference outputs for the browser parity test.

    python models/faultfinder/web/export_instrument.py      # writes site/public/instruments/faultfinder/

Days: the four in examples/ (the release's seeded draw, one per source), read through faultfinder.raw_day exactly
as the demo reads them, and eight more drawn with seed 20261117 from the 217 exam days embedded in lab/index.html
(the original page; webdata.py wrote them rounded to 0.01 degC and 0.001): per source one faulty day whose fault
FaultFinder can name and one healthy day. Each day carries its dataset label, the rule baseline's verdict (APAR for
AHUs, fan-on-while-unoccupied for the RTU, at the dev-tuned threshold, from the page) and its data source and licence.
No dev day is used. The building platform is not needed.

perturb() below is the one definition of what each control does to the input (net.ts mirrors it):
  offset  add a constant (K) to one temperature sensor: a sensor bias
  damper  rewrite the mixed-air temperature from the mixing-box balance T_ma = T_ra + f (T_oa - T_ra) while the fan
          runs, with the outdoor-air damper stuck at fraction f; the logged damper command is left as it was (a BMS
          logs commands, not positions), and so is everything downstream
  noise   add sigma x a fixed standard-normal pattern (per day, seeded) to the five measured temperatures
  drop    remove the mixed-air sensor (the channel becomes missing)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, REPO)

from faultfinder import FaultFinder, RAWC, R, CLASSES, LABELS, featurize, forward, raw_day  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("faultfinder")
SEED = 20261117
TEMPS = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone']
SOURCES = ['Iowa ERS (lab)', 'LBNL FLEXLAB (lab)', 'ORNL RTU (lab)', 'Nesbitt Hall (real building)']
GHALAMSIAH = 'via N. Ghalamsiah (2025), figshare doi:10.6084/m9.figshare.29297999.v3, CC BY 4.0'
LBNL = 'LBNL fault detection data synthesis (Granderson, Lin et al.), figshare doi:10.6084/m9.figshare.11752740, CC0'
CREDIT = {'ers_rp1312': f'ASHRAE RP-1312 (J. Wen and S. Li) {GHALAMSIAH}', 'nesbitt': f'Drexel Nesbitt Hall (Y. Chen) {GHALAMSIAH}',
          'lbnl_mzvav2_ers_exp': LBNL, 'lbnl_szcav_flexlab': LBNL, 'lbnl_szvav_flexlab': LBNL, 'lbnl_rtu_ornl': LBNL}
FACILITY = {'Iowa ERS (lab)': 'Iowa Energy Resource Station', 'LBNL FLEXLAB (lab)': 'LBNL FLEXLAB, Berkeley',
            'ORNL RTU (lab)': 'ORNL Flexible Research Platform', 'Nesbitt Hall (real building)': 'Nesbitt Hall, Drexel, Philadelphia'}


def perturb(x, nz, p):
    """The instrument's controls on one raw day x [13, 96] (NaN = missing). nz: [5, 96] standard normal, TEMPS order.
    p: {sensor, offset (K), damper (outdoor-air fraction 0..1, or None), noise (K), drop_ma (bool)}."""
    x = np.asarray(x, np.float64).copy()
    f = p.get('damper')
    if f is not None:
        on = x[R['fan_on']]
        run = np.where(np.isfinite(on), on > 0.5, True)
        ok = run & np.isfinite(x[R['T_oa']]) & np.isfinite(x[R['T_ra']])
        x[R['T_ma']] = np.where(ok, x[R['T_ra']] + f * (x[R['T_oa']] - x[R['T_ra']]), x[R['T_ma']])
    if p.get('offset'):
        x[R[p['sensor']]] = x[R[p['sensor']]] + p['offset']        # NaN stays NaN
    if p.get('noise'):
        for j, c in enumerate(TEMPS):
            x[R[c]] = x[R[c]] + p['noise'] * nz[j]
    if p.get('drop_ma'):
        x[R['T_ma']] = np.nan
    return x


def outputs(W, T, x):
    """Raw network outputs for one day: 19 class logits, 19 parameters, the energy output, then P(fault)."""
    lo, pa, wa = forward(W, featurize(x)[None].astype(np.float32))
    f = lo[0, 1:].astype(np.float64)
    m = f.max()
    d = m + np.log(np.exp(f - m).sum()) - float(lo[0, 0])
    return [float(v) for v in lo[0]] + [float(v) for v in pa[0]] + [float(wa[0]), float(1 / (1 + np.exp(-d / T)))]


def page_runs():
    lines = open(os.path.join(MODEL, 'lab', 'index.html'), encoding='utf-8').read().split('\n')
    line = next(s for s in lines if s.startswith('const DATA = '))
    D = json.loads(line[len('const DATA = '):].rstrip().rstrip(';'))
    return {r['id']: r for r in D['runs']}


def main():
    ff = FaultFinder()
    T = ff.T
    W = {k: v for k, v in ff.W.items() if k != 'temperature'}
    n_params = int(sum(v.size for v in W.values()))
    write_pack(OUT / 'weights', W, dtype='f16',
               meta={'temperature': T, 'classes': CLASSES, 'labels': LABELS, 'n_params': n_params, 'channels': RAWC})
    Wr = {k: v.astype(np.float32) for k, v in roundtrip(W, 'f16').items()}

    runs = page_runs()
    labels = json.load(open(os.path.join(MODEL, 'examples', 'labels.json')))
    days = []
    for fn in sorted(labels):
        m = labels[fn]
        r = runs[m['run_id']]
        x = raw_day(pd.read_csv(os.path.join(MODEL, 'examples', fn)))
        days.append((r, x, f'examples/{fn} (release draw, seed 7)'))
    have = {d[0]['id'] for d in days}
    rng = np.random.default_rng(SEED)
    for src in SOURCES:
        pool = sorted(runs.values(), key=lambda r: r['id'])
        for kind in ('fault', 'healthy'):
            g = [r for r in pool if r['src'] == src and r['id'] not in have
                 and ((r['fault'] != 'none' and r['diag'] not in ('none', 'other')) if kind == 'fault' else r['fault'] == 'none')]
            r = g[int(rng.integers(len(g)))]
            x = np.array([[np.nan if v is None else v for v in row] for row in r['x']], np.float32)
            days.append((r, x, f'lab page exam days, seeded draw {SEED} ({kind}, {src})'))
            have.add(r['id'])
    order = sorted(range(len(days)), key=lambda i: (SOURCES.index(days[i][0]['src']), days[i][0]['fault'] == 'none', days[i][0]['id']))
    days = [days[i] for i in order]

    variants = [('raw', {}), ('Tsa+3', {'sensor': 'T_sa', 'offset': 3.0}), ('Tma-2', {'sensor': 'T_ma', 'offset': -2.0}),
                ('Toa+4', {'sensor': 'T_oa', 'offset': 4.0}), ('damper1', {'damper': 1.0}), ('damper0.15', {'damper': 0.15}),
                ('noise1', {'noise': 1.0}), ('dropMA', {'drop_ma': True}),
                ('combo', {'sensor': 'T_ra', 'offset': -1.5, 'damper': 0.5, 'noise': 0.5, 'drop_ma': False})]
    xs, meta, cases, f16 = {}, [], [], []
    for i, (r, x, how) in enumerate(days):
        nz = np.random.default_rng(SEED + 1 + i).standard_normal((len(TEMPS), 96)).astype(np.float32)
        xs[f'x{i}'] = np.asarray(x, np.float32)
        xs[f'n{i}'] = nz
        x32 = xs[f'x{i}']
        for name, p in variants:
            xp = perturb(x32, nz, p)
            o = outputs(Wr, T, xp)
            cases.append({'example': i, 'variant': name, 'p': p, 'outputs': o})
            f16.append(abs(o[-1] - outputs(W, T, xp)[-1]))
        py = ff.predict_raw(np.asarray(x32, np.float64)[None])[0]
        diag = r['diag']
        meta.append({
            'id': r['id'], 'source': r['src'], 'facility': FACILITY[r['src']], 'dataset': r['ds'], 'unit': r['unit'],
            'date': r['date'], 'fault': r['fault'], 'severity': r['sev'],
            'diag_class': None if diag in ('none', 'other') else diag, 'simulable': diag != 'other',
            'rule': {'name': r['apar']['rule'], 'flag': r['apar']['flag'], 'score': r['apar']['score'],
                     'threshold': r['apar']['thr'], 'rules_fired': r['apar']['rules']},
            'page_p': r['py']['p'], 'python_p': py['p_fault'], 'credit': CREDIT[r['ds']], 'drawn': how,
            'has': [bool(np.isfinite(x32[k]).any()) for k in range(len(RAWC))],
        })
    write_pack(OUT / 'examples', xs, dtype='f32', meta=meta)
    write_json(OUT / 'reference.json', {'cases': cases, 'tolerance': 2e-3})
    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    dp = max(abs(m['python_p'] - m['page_p']) for m in meta)
    print(f'wrote {OUT}: {len(days)} days, {len(cases)} parity cases, {tot / 1e6:.2f} MB total; '
          f'float16 vs float32 weights: max P(fault) difference {max(f16):.1e}; Python vs the page on these days: {dp:.1e}')
    for m in meta:
        print(f"  {m['id']:34s} {m['fault']:24s} P {m['python_p']:.3f}  rule {'flag' if m['rule']['flag'] else 'ok'}  ({m['drawn']})")


if __name__ == '__main__':
    main()
