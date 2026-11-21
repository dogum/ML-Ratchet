"""Build the showpiece page: inline engine.js + data (final weights, exam results, a fixed stratified draw of exam
dams, ratchet). usage: python3 export_web.py weights.npz texts.json -> web/reservoirpilot.html"""
import sys, json, pickle
import numpy as np, pandas as pd
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from check_js import net_json
from task import wd
ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)


def r4(a, s=4):
    return [float(f'{v:.{s}g}') for v in np.asarray(a, float)]


def draw_dams(tasks, seed=20261118):
    """Fixed random draw within strata: 2 per main use (fcon, irri, elec, supp), 1 from the small uses, 2 CDEC."""
    rng = np.random.default_rng(seed); ex = [t for t in tasks if t['split'] == 'exam']; out = []
    for use in ('fcon', 'irri', 'elec', 'supp'):
        pool = [t for t in ex if t['main_use'] == use]; out += [pool[i] for i in rng.choice(len(pool), 2, replace=False)]
    pool = [t for t in ex if t['main_use'] in ('navi', 'recr', 'other')]; out.append(pool[int(rng.integers(len(pool)))])
    cd = [t for t in tasks if t['split'] == 'cdec']; out += [cd[i] for i in rng.choice(len(cd), 2, replace=False)]
    return out


def main(w, texts):
    tasks = pickle.load(open(f'{ROOT}/data/real_exam.pkl', 'rb'))
    meta = wd.reservoir_meta().set_index('res_id')
    ex = json.load(open(f'{ROOT}/exam/exam_results.json'))
    per = pd.read_parquet(f'{ROOT}/exam/exam_per_dam.parquet')
    T = json.load(open(texts))
    dams = []
    for tk in draw_dams(tasks):
        m = meta.loc[tk['res_id']]; dates = tk['dates']
        wy = [int(d.year + (d.month >= 10)) for d in dates]
        dams.append({'id': tk['res_id'], 'cdec': tk.get('cdec_id', ''), 'name': str(m.dam_name).title(), 'state': str(m.state),
                     'use': tk['main_use'], 'label': str(m.dam_name).title() + (' · CDEC' if tk.get('cdec_id') else ''),
                     'cap': tk['cap'], 'capYears': tk['cap'] / max(tk['cal_I_mean'] * 365.25 * 0.0864, 1e-9), 'Qmax': tk['Qmax'], 'S0': tk['S0'],
                     'start': str(dates[0].date()), 'end': str(dates[-1].date()), 'doy': [int(v) for v in dates.dayofyear],
                     'wy': wy, 'I': r4(tk['I']), 'Dcurve': r4(tk['Dcurve']), 'Icurve': r4(tk['Icurve']),
                     'Robs': r4(tk['R_obs']), 'Sobs': r4(tk['S_obs'])})
    e = per[per.split == 'exam']
    piv = e.pivot_table(index='res_id', columns='policy', values='J')
    scatter = [[i, float(f"{r.pilot:.4g}"), float(f"{r.sop:.4g}")] for i, r in piv.iterrows()]
    t = ex['resopsus_exam']
    head = {'pilot': t['pilot']['J'], 'sop': t['sop']['J'], 'hist': t['historical']['J'], 'dp': t['dp']['J']}
    rj = json.load(open(f'{ROOT}/ratchet.json'))
    ratchet = [{'id': r['id'], 'desc': r['desc'], 'g1': r['sim'].get('g1', {}).get('S_J'), 'g4': r['sim'].get('g4', {}).get('S_J'),
                'dev': r['dev'].get('J_median'), 'kept': r['kept'], 'decision': r['decision']} for r in rj['rungs']]
    data = {'headline': head, 'net': net_json(w), 'dams': dams, 'exam': ex, 'scatter': scatter, 'ratchet': ratchet, **T}
    html = open(f'{_paths.SRC}/template.html').read()
    html = html.replace('/*ENGINE*/', open(f'{_paths.SRC}/engine.js').read())
    html = html.replace('/*DATA*/', json.dumps(data, separators=(',', ':')))
    open(f'{ROOT}/web/reservoirpilot.html', 'w').write(html)
    print('bytes', len(html), 'dams', [d['label'] for d in dams])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
