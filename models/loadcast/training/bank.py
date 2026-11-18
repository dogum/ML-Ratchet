"""Simulated training banks: generated buildings x real BDG2 site weather 2016-2017 (both years, one run).

    nice -n 5 python3 bank.py <phys> <name> <n> <seed>
      phys = v0  Phase 1 generic twins (generator building + meters decide the system, defaults otherwise)
             v1d as v1, with a deterministic generator spec per building (17 Nov 2026: v1's spec depended on the order
                 in which a process asked for them, so banks were not reproducible); physics from $BUILDINGSIM_PHYSICS
             v1  + dev-calibrated knobs drawn from the 52 dev fits (data/dev_knobs.json, or $LC_KNOBS): HVAC for electricity-only
                 buildings, operating schedule, outdoor air, VAV minimum/reheat, meter scale a and base load b

Buildings: use type drawn with BDG2-like weights; ComStock/CBECS-based generator spec at the site's climate zone;
floor area from the generator or (half the time) from the dev log-area distribution; which meters exist drawn from the
dev buildings' meter sets (80 %) or a fixed list (20 %). Clean meter series are stored (kW / (A * IREF)), float16;
noise is added later (dataset.py) so one simulation bank serves several noise rungs.
Output: data/bank_<name>/part_XXXX.npz (+ meta json lines), resumable.
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, lcsim as L
import buildingsim as bs

USE_P = {'education': 0.24, 'office': 0.15, 'other': 0.12, 'assembly': 0.12, 'lodging': 0.12, 'healthcare': 0.06,
         'retail': 0.05, 'warehouse': 0.05, 'food': 0.04, 'lab': 0.05}
EXTRA_CONFIGS = [['electricity'], ['electricity', 'gas'], ['electricity', 'chilledwater', 'gas'],
                 ['electricity', 'chilledwater', 'steam'], ['electricity', 'chilledwater', 'hotwater'],
                 ['electricity', 'steam'], ['electricity', 'hotwater'], ['electricity', 'chilledwater'],
                 ['electricity', 'steam', 'gas'], ['electricity', 'chilledwater', 'steam', 'gas']]
PART = 100


def dev_info():
    meta = C.bd.bdg2_meta(split='dev')
    cfgs = [C.meters_of(m) for m in meta.meters]
    la = np.log(meta.floor_area_m2.values)
    cz = {r.site_id: r.climate_zone for r in C.bd.bdg2_meta().drop_duplicates('site_id').itertuples()}
    kf = f"{C.ROOT}/data/{os.environ.get('LC_KNOBS', 'dev_knobs.json')}"
    kn = json.load(open(kf)) if os.path.exists(kf) else {}
    return cfgs, (la.mean(), la.std()), cz, kn


def sys_class(spec_filled, meters):
    hc = C.hvac_class_from_meters(meters)
    if hc != 'unknown': return hc
    s = spec_filled['system']; hs = (spec_filled.get('hvac') or {}).get('heat_source')
    if s == 'district_steam_chw': return 'district_all'
    if s in ('heat_pump_air', 'heat_pump_ground', 'baseboard_resistance') or hs in ('hp', 'elec', 'electric'): return 'all_electric'
    if s == 'ptac' and hs in (None,): return 'all_electric'
    if s in ('vav_chiller_boiler', 'fan_coil', 'radiant', 'hydronic_baseboard'): return 'gas_vav'
    return 'gas_rtu'


def sched_class(spec_filled, knobs):
    sc = spec_filled['occupancy_schedule']
    if isinstance(sc, dict) and sc.get('type') in ('24x7', 'lodging'): return '24x7'
    return {'default': 'standard', 'extended': 'extended', '24x7': '24x7'}[knobs.get('sched', 'default')]


def draw_knobs(rng, use, meters, kn, phys):
    ms = set(meters)
    elec_only = not (ms & {'chilledwater', 'steam', 'hotwater', 'gas'})
    if phys == 'v0' or not kn:
        return dict(sys='keep' if elec_only else 'auto')
    ids = list(kn)
    same = [b for b in ids if kn[b]['use'] == use]
    donor = kn[same[rng.integers(len(same))]] if same and rng.random() < 0.6 else kn[ids[rng.integers(len(ids))]]
    k = dict(donor['best']['knobs'])
    if elec_only:
        pool = [kn[b]['best']['knobs']['sys'] for b in ids if kn[b]['best']['knobs']['sys'] in ('auto', 'district', 'electric')]
        s = pool[rng.integers(len(pool))]
        k['sys'] = 'keep' if s == 'auto' else s
    elif 'gas' in ms and not (ms & {'chilledwater', 'steam', 'hotwater'}):
        k['sys'] = ['gas_rtu', 'gas_vav'][rng.integers(2)]
    else:
        k['sys'] = 'auto'
    k['oa'] = float(k['oa'] * np.exp(rng.normal(0, 0.2)))
    if 'vav_min' in k: k['vav_min'] = float(np.clip(k['vav_min'] + rng.normal(0, 0.08), 0.2, 1.0))
    for m in meters:
        per = donor['best']['per'] if m in donor['best']['per'] else None
        if per is None:
            pool = [b for b in ids if m in kn[b]['best']['per'] and kn[b]['best']['per'][m]['cv_fit'] < 200]
            if not pool:   # no dev fit for this meter type: borrow the pooled heating meters
                pool = [b for b in ids if any(x in kn[b]['best']['per'] for x in ('steam', 'hotwater', 'gas'))]
                d = kn[pool[rng.integers(len(pool))]]['best']['per']
                f = d[[x for x in ('steam', 'hotwater', 'gas') if x in d][0]]
            else:
                f = kn[pool[rng.integers(len(pool))]]['best']['per'][m]
        else:
            f = per[m]
        k[f'a_{m}'] = float(f['a'] * np.exp(rng.normal(0, 0.2)))
        k[f'b_{m}'] = float(f['b'] * np.exp(rng.normal(0, 0.2)))
    return k


_GEN = {}


def gen_spec_det(use, cz, gs):
    """One generator spec per building from its own seed: independent of process and order (phys v1d)."""
    try:
        r = bs.generate_buildings(1, seed=gs, primary_use=use, climate_zone=cz)
    except Exception:
        r = []
    return (r or bs.generate_buildings(1, seed=gs, primary_use=use))[0]


def gen_spec(use, cz, gs):
    """Generator specs are drawn 25 at a time per (use, climate zone) (the prior filtering is the slow part)."""
    key = (use, cz)
    if not _GEN.get(key):
        try:
            _GEN[key] = bs.generate_buildings(25, seed=gs, primary_use=use, climate_zone=cz)
        except Exception:
            _GEN[key] = []
        if not _GEN[key]:
            _GEN[key] = bs.generate_buildings(25, seed=gs, primary_use=use)
    return _GEN[key].pop()


def make_one(i, seed, phys, info):
    cfgs, (lam, las), czmap, kn = info
    rng = np.random.default_rng([seed, i])
    site = C.SITES[rng.integers(len(C.SITES))]
    cz = czmap[site]
    use = list(USE_P)[rng.choice(len(USE_P), p=np.array(list(USE_P.values())) / sum(USE_P.values()))]
    spec = (gen_spec_det if phys.endswith('d') else gen_spec)(use, cz, int(rng.integers(1 << 30)))
    spec['climate_zone'] = cz
    if rng.random() < 0.5 or not spec.get('floor_area_m2'):
        spec['floor_area_m2'] = float(np.exp(rng.normal(lam, las)))
        spec['n_floors'] = None
    spec['floor_area_m2'] = float(np.clip(spec['floor_area_m2'], 150, 150000))
    if spec.get('n_floors') is None:
        spec['n_floors'] = max(1, int(round(np.sqrt(spec['floor_area_m2']) / 30)))
    meters = cfgs[rng.integers(len(cfgs))] if rng.random() < 0.8 else EXTRA_CONFIGS[rng.integers(len(EXTRA_CONFIGS))]
    meters = list(meters) if 'electricity' in meters else ['electricity'] + list(meters)
    knobs = draw_knobs(rng, use, meters, kn, phys)
    s = L.make_spec(spec, meters, knobs)
    t0 = time.time()
    raw, out = L.simulate_site(s, site)
    y = L.post(raw, knobs, s['floor_area_m2'])
    fs = out['spec']
    A = fs['floor_area_m2']
    Y = {m: (y[m] / (A * C.IREF[m] / 1000.0)).astype(np.float16) for m in meters}
    rec = dict(i=i, site=site, use=use, area=A, year_built=fs.get('year_built'), meters=meters, knobs=knobs,
               system=fs['system'], heat_source=fs['hvac'].get('heat_source'), hvac_class=sys_class(fs, meters),
               sched_class=sched_class(fs, knobs), dir=int(rng.integers(2)), yb_shown=bool(rng.random() < 0.5),
               wall_s=time.time() - t0, eui=float(out['summary']['EUI_kWh_m2_yr']),
               ledger_ok=bool(out['ledger']['ok']), physics=str(out.get('physics', '1.0')))
    return rec, Y


def main(phys, name, n, seed):
    d = f'{C.ROOT}/data/bank_{name}'; os.makedirs(d, exist_ok=True)
    info = dev_info()
    if phys.startswith('v1') and not info[3]: raise SystemExit('need data/dev_knobs.json')
    t0 = time.time(); done_here = 0
    for p0 in range(0, n, PART):
        f = f'{d}/part_{p0 // PART:04d}.npz'
        if os.path.exists(f): continue
        if done_here >= int(os.environ.get('LC_MAXPARTS', 99)): break    # simulator caches grow: restart often
        done_here += 1
        recs, ys = [], {}
        for i in range(p0, min(p0 + PART, n)):
            try:
                rec, Y = make_one(i, seed, phys, info)
            except Exception as e:
                print('ERR', i, repr(e)[:200], flush=True); continue
            recs.append(rec)
            for m, v in Y.items(): ys[f'{rec["i"]}_{m}'] = v
        np.savez(f + '.tmp.npz', meta=json.dumps(recs, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), **ys)
        os.replace(f + '.tmp.npz', f)
        print(f'{name}: {min(p0 + PART, n)}/{n} buildings, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
