"""Build release/ (lawfinder_laws.json, examples/, training/) and the for_repo/models/lawfinder layout."""
import json, os, shutil, importlib, glob
import numpy as np, pandas as pd
import lawsearch as ls
import lawratchet as lr
import pretty

ROOT = lr.ROOT
REL = f'{ROOT}/release'
PLATFORM = {'water': 'WaterSim', 'fire': 'FireSim', 'grid_inertia': 'GridSim', 'grid_nadir': 'GridSim', 'bldg': 'BuildingSim', 'road': 'RoadSim'}
TB = {'water': ('budyko_fu', 'w_dev', 'fu_dev'), 'fire': ('rothermel_x_perimeter', 'k_dev', 'rothermel_dev'),
      'grid_inertia': ('k_times_demand', 'k_demand_GVAs_per_GW', 'k_x_demand'),
      'grid_nadir': ('swing_ramp', 'k_swing', 'swing_ramp'), 'bldg': ('degree_day_signature', 'coef_dev', 'signature_dev'),
      'road': ('road_load', 'eta_dev', 'roadload_dev')}
TB_LABEL = {'water': 'Budyko-Fu, w fitted on dev', 'fire': 'Rothermel head ROS x perimeter x 24 h, k fitted on dev',
            'grid_inertia': 'k x demand, k fitted on dev', 'grid_nadir': 'swing + ramp response, k L^2/E, k fitted on dev',
            'bldg': 'degree-day signature a + b HDD + c CDD, fitted on dev',
            'road': 'road-load energy / eta, eta per powertrain fitted on dev'}
POINTS = {'grid_inertia': 'exam_grid_inertia_points', 'grid_nadir': 'exam_grid_nadir_points'}
STRATA_N = 5


def exam_of(name):
    f = 'exam_grid.json' if name.startswith('grid_') else ('exam_road_reported.json' if name == 'road' else f'exam_{name}.json')
    p = f'{ROOT}/results/{f}'
    if not os.path.exists(p):
        return None
    E = json.load(open(p))
    return E[name.split('_')[1]] if name.startswith('grid_') else E


def law_entry(name, laws):
    """add law `name` to `laws` and write release/examples/<name>.csv; False if its ratchet or exam is missing"""
    pr = f'{ROOT}/results/{name}_ratchet.json'
    E = exam_of(name)
    if not os.path.exists(pr) or E is None:
        return False
    L = importlib.import_module(f'law_{name}')
    R = json.load(open(pr))
    rel = next(r for r in R['rungs'] if r['id'] == R['released'])
    kn = lr.knee(R)
    t = ls.from_json(rel['tree']); c = np.array(rel['consts'])
    tk = ls.from_json(kn['tree']); ck = np.array(kn['consts'])
    form, key, row = TB[name]
    laws[name] = {'title': L.TITLE, 'platform': PLATFORM[name], 'vars': L.VARS, 'var_info': L.VAR_INFO,
                  'target': L.TARGET, 'rung': rel['id'], 'expr': rel['expr'], 'text': pretty.text(t, c, L.VARS),
                  'tree': rel['tree'], 'consts': rel['consts'], 'sigma': R['bars']['sigma'],
                  'sigma_sim': R['bars']['sigma_sim'], 'widen_dev': R['bars']['widen_dev'],
                  'simulator_law': {'rung': kn['id'], 'text': pretty.text(tk, ck, L.VARS), 'tree': kn['tree'],
                                    'consts': kn['consts']},
                  'textbook': {'form': form, 'const': E.get(key) if key in E else None, 'label': TB_LABEL[name]},
                  'exam': {k: {kk: vv for kk, vv in v.items()} for k, v in E['rows'].items()}}
    if name.startswith('grid_'):
        G = json.load(open(f'{ROOT}/results/exam_grid.json'))
        laws[name]['textbook']['const'] = G[key]
    # examples: fixed random draw within strata (terciles of the first input), never picked by result
    P = pd.read_parquet(f"{ROOT}/results/{POINTS.get(name, f'exam_{name}_points')}.parquet")
    X = L.X_of(P); y = L.target(P)
    df = pd.DataFrame(X, columns=L.VARS); df['truth'] = y
    df['textbook'] = P[f'pred_{row}'].values
    df['id'] = (P['basin'] if 'basin' in P else P['fire_id'].astype(str) + '_k' + P['k'].astype(str) if 'fire_id' in P
                else P['event_id'] if 'event_id' in P else P['id']).values
    df = df[np.isfinite(df[L.VARS + ['truth', 'textbook']]).all(1)].reset_index(drop=True)
    q = pd.qcut(df[L.VARS[0]].rank(method='first'), 3, labels=False)
    rng = np.random.default_rng(20261122)
    pick = []
    for s in range(3):
        idx = np.where(q == s)[0]
        pick += list(rng.choice(idx, size=min(len(idx), 2 if s < 2 else 1), replace=False))
    ex = df.iloc[sorted(pick)]
    ex.to_csv(f'{REL}/examples/{name}.csv', index=False, float_format='%.6g')
    return True


def main(only=None):
    """only: law names to (re)build and merge into the shipped lawfinder_laws.json (e.g. --only road: the other laws'
    exam point files are not shipped, so they cannot be rebuilt from this folder)"""
    laws = {}
    os.makedirs(f'{REL}/examples', exist_ok=True)
    if only:
        from _paths import MODEL
        laws = json.load(open(f'{MODEL}/lawfinder_laws.json'))['laws']
    for name in (only or PLATFORM):
        law_entry(name, laws)
    json.dump({'version': '1.0', 'released': '2026-11-22', 'laws': laws}, open(f'{REL}/lawfinder_laws.json', 'w'),
              indent=1, default=float)
    # training scripts
    tdir = f'{REL}/training'
    os.makedirs(tdir, exist_ok=True)
    from _paths import SRC
    for f in glob.glob(f'{SRC}/*.py') + glob.glob(f'{SRC}/*.sh'):
        shutil.copy(f, tdir)
    for f in glob.glob(f'{ROOT}/results/*_ratchet.json') + glob.glob(f'{ROOT}/results/*_search_*.json') + \
            glob.glob(f'{ROOT}/results/exam_*.json'):
        os.makedirs(f'{tdir}/results', exist_ok=True); shutil.copy(f, f'{tdir}/results')
    shutil.copy(f'{ROOT}/splits.json', REL)
    print('laws', list(laws))


if __name__ == '__main__':
    import sys
    main(sys.argv[sys.argv.index('--only') + 1].split(',') if '--only' in sys.argv else None)
