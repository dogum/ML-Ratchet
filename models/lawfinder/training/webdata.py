"""Collect everything the LawFinder page needs into web/data.json (small: sampled clouds, exam points, ladders)."""
import json, os, importlib
import numpy as np, pandas as pd
import lawsearch as ls
import lawratchet as lr
import pretty

ROOT = lr.ROOT
LAWS = ['water', 'fire', 'grid_inertia', 'grid_nadir', 'bldg', 'road']
QUESTION = {
    'water': 'What share of the rain and snow leaves a basin as river flow, over a decade?',
    'fire': 'How much new ground does a running wildfire burn in the next 24 hours?',
    'grid_inertia': 'How much spinning inertia does the GB grid have at a given demand?',
    'grid_nadir': 'How far does GB frequency fall after a power station trips?',
    'bldg': 'How much energy does a single-family home use per square metre in a year?',
    'road': 'How much energy does a car use per kilometre on a given trip?',
}
PLATFORM = {'water': 'WaterSim', 'fire': 'FireSim', 'grid_inertia': 'GridSim', 'grid_nadir': 'GridSim', 'bldg': 'BuildingSim', 'road': 'RoadSim'}
XLOG = {'fire': ['P', 'R0'], 'grid_nadir': [], 'bldg': [], 'road': []}
XDEF = {'water': 'phi', 'fire': 'P', 'grid_inertia': 'D', 'grid_nadir': 'L', 'bldg': 'H', 'road': 'v'}


def r(v, n=4):
    return None if v is None or not np.isfinite(v) else float(f'{v:.{n}g}')


def sample(df, n, seed=0):
    return df.sample(min(n, len(df)), random_state=seed) if len(df) > n else df


def law_block(name):
    L = importlib.import_module(f'law_{name}')
    R = json.load(open(f'{ROOT}/results/{name}_ratchet.json'))
    exf = {'grid_inertia': 'exam_grid.json', 'grid_nadir': 'exam_grid.json',
           'road': 'exam_road_reported.json'}.get(name, f'exam_{name}.json')   # road: run 1 with its errata
    E = json.load(open(f'{ROOT}/results/{exf}')) if os.path.exists(f'{ROOT}/results/{exf}') else None
    if E is not None and name.startswith('grid_'):
        E = dict(E[name.split('_')[1]], k_demand_GVAs_per_GW=E['k_demand_GVAs_per_GW'], k_swing=E['k_swing'])
    ver = L.VERSIONS[-1]
    va = L.load_bank(ver, 'val')
    Xv, yv = L.X_of(va), L.target(va)
    dev = L.load_dev(); Xd, yd = L.X_of(dev), L.target(dev)
    dec = {d['id']: d for d in R['history'][-1]['decisions']}
    ladder = []
    for g in R['rungs']:
        t = ls.from_json(g['tree']); c = np.array(g['consts'])
        ladder.append({'id': g['id'], 'version': g['version'], 'complexity': g['complexity'],
                       'text': pretty.text(t, c, L.VARS), 'mathml': pretty.mathml(t, c, L.VARS), 'js': g['js'],
                       'sim': r(g['sim']), 'dev': r(g['dev']), 'decision': dec[g['id']]['decision'],
                       'why': dec[g['id']]['why'], 'exam': (E or {}).get('rungs', {}).get(g['id']),
                       'scores': {k: {kk: r(vv) for kk, vv in v.items()} for k, v in g.get('scores', {}).items()}})
    rel = next(g for g in ladder if g['id'] == R['released'])
    cloud = pd.DataFrame(Xv, columns=L.VARS); cloud['y'] = yv
    cloud = sample(cloud, 1400)
    devp = pd.DataFrame(Xd, columns=L.VARS); devp['y'] = yd
    pts = None
    if E is not None:
        pf = {'grid_inertia': 'exam_grid_inertia_points', 'grid_nadir': 'exam_grid_nadir_points'}.get(name, f'exam_{name}_points')
        P = pd.read_parquet(f'{ROOT}/results/{pf}.parquet')
        if name == 'grid_inertia':
            P = P.rename(columns={'inertia_GVAs': 'inertia_GVAs'})
        Xe = L.X_of(P); ye = L.target(P)
        pe = pd.DataFrame(Xe, columns=L.VARS); pe['y'] = ye
        pe = pe[np.isfinite(pe).all(1)]
        pts = sample(pe, 1500, 1)
    out = {'id': name, 'title': L.TITLE, 'question': QUESTION[name], 'platform': PLATFORM[name],
           'vars': [{'id': v, **L.VAR_INFO[v], 'log': v in XLOG.get(name, []),
                     'median': r(float(np.nanmedian(Xd[:, i]))) } for i, v in enumerate(L.VARS)],
           'xdef': XDEF[name], 'target': L.TARGET, 'released': rel, 'ladder': ladder, 'bars': R['bars'],
           'rule': R['rule'], 'history': R['history'],
           'cloud': {k: [r(v, 4) for v in cloud[k]] for k in cloud.columns},
           'dev': {k: [r(v, 4) for v in devp[k]] for k in devp.columns},
           'exam_points': None if pts is None else {k: [r(v, 4) for v in pts[k]] for k in pts.columns},
           'exam': E}
    return out


def main():
    data = {'laws': []}
    for n in LAWS:
        if os.path.exists(f'{ROOT}/results/{n}_ratchet.json'):
            data['laws'].append(law_block(n))
            print('ok', n)
    os.makedirs(f'{ROOT}/web', exist_ok=True)
    json.dump(data, open(f'{ROOT}/web/data.json', 'w'), separators=(',', ':'))
    print('size KB', os.path.getsize(f'{ROOT}/web/data.json') / 1e3)


if __name__ == '__main__':
    main()
