"""Fixed case sets: frozen (simulated vehicles x held-out real-noise / synthetic / cycle routes), dev gate (VED dev
hybrid routes + ZTBus dev legs), exam (VED exam hybrid routes + ZTBus exam legs + HD cycles). Cached as pickles.
Physics version is recorded with each set (RoadSim physics + any HybridPilot patch)."""
import sys, os, pickle, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hp_env import *

PHYS = 'roadsim-physics-v1'


def _ztbus(split, n=None, seed=0, max_s=2400):
    import realdata as rd
    out = []
    for meta, arr in rd.load_trips(split, source='ZTBus', limit=n, seed=seed):
        tr = {k: np.nan_to_num(np.asarray(arr[k], float)) for k in ['t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'surface'] if k in arr}
        tr['surface'] = tr['surface'].astype(int)
        # start at the first stop after 60 s, take max_s seconds (bus legs are hours long)
        tr = {k: x[:max_s] for k, x in tr.items()}
        tr['_name'] = meta['trip_id']; tr['_kind'] = 'ztbus_' + split; tr['_vehicle'] = meta.get('vehicle_id')
        out.append(tr)
    return out


def _case(spec, tr, rng, nav=None, warm=None):
    soc0, temps = init_conditions(rng, spec, tr, warm_p=0.6)
    if warm is not None:
        temps = dict(WARM) if warm else None
    return dict(spec=spec, trip=tr, soc0=soc0, temps=temps, nav=bool(rng.random() < 0.5) if nav is None else nav,
                case_id=f"{spec['id']}|{tr['_name']}")


def build_frozen():
    rng = np.random.default_rng(SEED['frozen'])
    fleet = fleet_balanced('frozen', 2)
    _, fro_v = train_noise_partition()
    ved = ved_routes('train_noise', vehicles=fro_v)
    cases = []
    for i, spec in enumerate(fleet):
        hd = is_hd(spec['class'])
        # 1: a real-noise route (frozen train_noise vehicles)
        if spec['class'] != 'truck_hd':
            cases.append(_case(spec, copy.deepcopy(ved[int(rng.integers(len(ved)))]), rng))
        # 2: a synthetic route of the vehicle's kind
        kind = 'bus' if spec['class'] == 'bus' else ('highway' if spec['class'] == 'truck_hd' else ['urban', 'suburban', 'mixed', 'hilly'][i % 4])
        v, g = synth_route(rng, kind, hd=hd)
        cases.append(_case(spec, trip_from_trace(v, g, draw_weather(rng), f'fsynth_{kind}_{i}', 'synth_' + kind), rng))
        # 3: a cycle with random weather (cold start half the time)
        name = (CYCLES_HD if hd else CYCLES_LD)[i % 6]
        tr = make_trip(name, **draw_weather(rng)); tr['_name'] = name; tr['_kind'] = 'cycle'
        cases.append(_case(spec, tr, rng, warm=bool(i % 2)))
    return cases


def build_dev():
    rng = np.random.default_rng(SEED['dev'])
    import realdata as rd
    cases = []
    ved = ved_routes('dev', arch=['hev_powersplit', 'phev_series', 'phev_powersplit'], label_ok=True, limit=None)
    byv = {}
    for t in ved:
        byv.setdefault(t['_vehicle'], []).append(t)
    for vid in sorted(byv):
        tw = twin_spec(vid)
        L = byv[vid]
        for j in rng.choice(len(L), min(6, len(L)), replace=False):
            cases.append(_case(tw, copy.deepcopy(L[int(j)]), rng, warm=True))
    fleet = fleet_balanced('dev', 1)
    allv = [t for v in sorted(byv) for t in byv[v]]
    zt = _ztbus('dev')
    for spec in fleet:
        if spec['class'] == 'bus' and zt:
            trs = [zt[int(rng.integers(len(zt)))]]
        else:
            trs = [allv[int(rng.integers(len(allv)))] for _ in range(2)]
        for tr in trs:
            cases.append(_case(spec, copy.deepcopy(tr), rng))
    return cases


def build_exam():
    """fixed random draw within strata (never cherry-picked): 2 trips per VED exam hybrid vehicle driven by its twin;
    held-out generated hybrids (3 per class x arch) on exam routes; buses on ZTBus exam legs; trucks also on HD cycles."""
    rng = np.random.default_rng(SEED['exam'])
    import realdata as rd
    idx = rd.load_trips('exam', arch=['hev_powersplit', 'hev_parallel_p2', 'phev_series', 'phev_powersplit'], label_ok=True, meta_only=True)
    idx = idx[(idx.duration_s >= 180)]
    pick = idx.sort_values('trip_id').sample(frac=1.0, random_state=SEED['exam']).groupby('vehicle_id').head(2).sort_values('trip_id')
    ved = ved_routes('exam', vehicles=None, label_ok=True, limit=None) if False else None
    routes = []
    for r in pick.itertuples():
        meta_arr = rd.load_trip(r.trip_id) if hasattr(rd, 'load_trip') else None
        meta, arr = meta_arr if isinstance(meta_arr, tuple) else (None, meta_arr)
        tr = {k: np.nan_to_num(np.asarray(arr[k], float)) for k in ['t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface']}
        tr['surface'] = tr['surface'].astype(int)
        tr = {k: x[:2700] for k, x in tr.items()}
        tr['_name'] = r.trip_id; tr['_kind'] = 'ved_exam'; tr['_vehicle'] = r.vehicle_id; tr['_arch'] = r.architecture
        routes.append(tr)
    cases = []
    for tr in routes:
        try:
            tw = twin_spec(tr['_vehicle'])
        except Exception as ex:
            print('twin fail', tr['_vehicle'], ex); continue
        c = _case(tw, tr, rng, warm=True); c['group'] = 'twin'
        cases.append(c)
    fleet = fleet_balanced('exam', 3)
    zt = _ztbus('exam', n=12, seed=SEED['exam'])
    for spec in fleet:
        cl = spec['class']
        if cl == 'bus':
            trs = [zt[int(rng.integers(len(zt)))] for _ in range(2)] + [routes[int(rng.integers(len(routes)))] for _ in range(2)]
        elif cl == 'truck_hd':
            trs = []
            for nm in ['hd_transient', 'gem_55', 'gem_65']:
                t = make_trip(nm, **draw_weather(rng)); t['_name'] = nm; t['_kind'] = 'cycle'; trs.append(t)
            trs.append(routes[int(rng.integers(len(routes)))])
        else:
            trs = [routes[int(rng.integers(len(routes)))] for _ in range(4)]
        for tr in trs:
            c = _case(spec, copy.deepcopy(tr), rng); c['group'] = 'generated'
            cases.append(c)
    return cases


def build_tune():
    """60 cases from the TRAINING distribution (training fleet, training routes, new seed) for policy-search fine-tuning."""
    fleet = gen_fleet(400, 'train')
    tr_v, _ = train_noise_partition()
    ved = ved_routes('train_noise', vehicles=tr_v)
    cs = train_cases(60, 9999, role='tune', fleet=fleet, ved=ved)
    for c in cs:
        c['case_id'] = c['case_id'] + '|' + c['spec']['id']
    return cs


def get(name):
    fn = f'{ROOT}/data/set_{name}_{PHYS}.pkl'
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    cases = {'frozen': build_frozen, 'dev': build_dev, 'exam': build_exam, 'tune': build_tune}[name]()
    pickle.dump(cases, open(fn, 'wb'))
    return cases


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        c = get(nm)
        from collections import Counter
        print(nm, len(c), Counter((x['spec']['class'], x['spec']['architecture']) for x in c).most_common(),
              'steps', sum(len(x['trip']['v']) for x in c))
