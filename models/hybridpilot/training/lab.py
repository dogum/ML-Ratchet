"""Precompute the page's lab: real routes driven by hybrids under each strategy, swept over temperature, mass and
preview length. Output web/lab.json (compact: per run engine-on bits, SOC bytes, cumulative fuel, summary numbers).
usage: lab.py CKPT 'json policy params'"""
import sys, os, json, copy, base64, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hp_env import *
from hp_score import run_case
from hp_policy import MLP
import sets

STRATS = ['rule', 'ecms', 'aecms', 'hybridpilot', 'dp']


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pick_route(vid, min_s=900, max_s=1800):
    import realdata as rd
    d = rd.load_trips('dev', vehicles=[vid], label_ok=True, meta_only=True)
    d = d[(d.duration_s >= min_s)].sort_values('trip_id')
    r = d.iloc[0] if len(d) else rd.load_trips('dev', vehicles=[vid], label_ok=True, meta_only=True).sort_values('duration_s').iloc[-1]
    meta, a = rd.load_trip(r.trip_id)
    tr = {k: np.nan_to_num(np.asarray(a[k], float))[:max_s] for k in ['t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface']}
    tr['surface'] = tr['surface'].astype(int); tr['_name'] = r.trip_id; tr['_kind'] = 'ved_dev'
    return tr, meta


def combos():
    rng = np.random.default_rng(SEED['lab'])
    aa, m1 = pick_route('VED_258')
    volt_r, m2 = pick_route('VED_537')
    zt = sets._ztbus('dev', max_s=1800)
    zt = sorted(zt, key=lambda t: t['_name'])[0]
    p2 = generate_vehicles(1, seed=int(rng.integers(1 << 30)), cls='suv', arch='hev_parallel_p2')[0]; p2['id'] = 'lab_suv_p2'
    bus = generate_vehicles(1, seed=int(rng.integers(1 << 30)), cls='bus', arch='hev_series')[0]; bus['id'] = 'lab_bus_series'
    trk = generate_vehicles(1, seed=int(rng.integers(1 << 30)), cls='truck_hd', arch='hev_parallel_p2')[0]; trk['id'] = 'lab_truck_p2'
    gem = make_trip('gem_55', T_amb=15.0); gem['_name'] = 'gem_55'; gem['_kind'] = 'cycle'
    udds = make_trip('udds', T_amb=5.0); udds['_name'] = 'udds'; udds['_kind'] = 'cycle'
    prius = twin_spec('VED_258'); volt = twin_spec('VED_537')
    return [
        dict(key='prius_aa', label='Power-split car (Prius-like twin of VED_258), Ann Arbor route', spec=prius, trip=aa, soc0=0.6, temps=dict(WARM), route=f"VED dev trip {aa['_name']}"),
        dict(key='prius_cold', label='Power-split car, UDDS city cycle from a cold soak at 5 C', spec=prius, trip=udds, soc0=0.6, temps=None, route='EPA UDDS'),
        dict(key='p2_aa', label='P2 parallel SUV (generated), same Ann Arbor route', spec=p2, trip=aa, soc0=0.55, temps=dict(WARM), route=f"VED dev trip {aa['_name']}"),
        dict(key='volt_aa', label='Series plug-in (Volt-like twin of VED_537), Ann Arbor route, battery at 45 %', spec=volt, trip=volt_r, soc0=0.45, temps=dict(WARM), route=f"VED dev trip {volt_r['_name']}"),
        dict(key='bus_zh', label='Series hybrid city bus (generated), Zurich trolleybus route (ZTBus dev leg)', spec=bus, trip=zt, soc0=0.55, temps=dict(WARM), route=f"ZTBus dev leg {zt['_name']}"),
        dict(key='truck_gem', label='P2 hybrid Class 8 tractor (generated), 55 mph over the EPA GEM grade profile', spec=trk, trip=gem, soc0=0.55, temps=dict(WARM), route='40 CFR 1037 App. D grade at 55 mph'),
    ]


def variants(cb):
    V = [dict(tag='base')]
    for T in (-10, 0, 10, 25, 35):
        V.append(dict(tag=f'T{T}', T=T))
    for mf in (0.9, 1.2, 1.4):
        V.append(dict(tag=f'm{mf}', mass=mf))
    return V


def apply(cb, v):
    spec = copy.deepcopy(cb['spec']); trip = copy.deepcopy(cb['trip'])
    if 'T' in v:
        trip['T_amb'] = np.full(len(trip['v']), float(v['T']))
        trip.pop('rho_air', None)
    if 'mass' in v:
        from roadsim.sim import get_vehicle
        m0 = get_vehicle(cb['spec']).mass
        spec['payload_kg'] = float(spec.get('payload_kg', 0.0) or 0.0) + (v['mass'] - 1.0) * m0
        spec['id'] = spec['id'] + f"_m{v['mass']}"
    return spec, trip


def run(ckpt, params, previews=(0, 10, 30, 60)):
    net = MLP.load(ckpt)
    out = dict(combos=[], strategies=STRATS, previews=list(previews), policy=params)
    from roadsim import simulate
    from roadsim.sim import get_vehicle
    for cb in combos():
        veh = get_vehicle(cb['spec'])
        rec = dict(key=cb['key'], label=cb['label'], route=cb['route'], arch=veh.arch, cls=veh.cls, mass_kg=veh.mass,
                   soc_lo=veh.battery.soc_lo, soc_hi=veh.battery.soc_hi, E_kWh=veh.battery.E_nom / 3.6e6, Pe_kW=veh.engine.P_max / 1e3,
                   N=len(cb['trip']['v']), v=b64(np.round(np.asarray(cb['trip']['v']) * 10).astype(np.uint16)),
                   grade=b64(np.clip(np.round(np.asarray(cb['trip']['grade']) * 1000), -127, 127).astype(np.int8)),
                   T_route=float(np.mean(cb['trip']['T_amb'])), runs={})
        for v in variants(cb):
            spec, trip = apply(cb, v)
            pols = {'hybridpilot': dict(params, net=net, preview_s=params.get('preview_s', 30))}
            for pv in previews:
                if v['tag'] == 'base':
                    pols[f'hybridpilot_p{pv}'] = dict(params, net=net, preview_s=pv)
            from hp_score import dp_curve, dp_targets, metrics, grade, add_matched, LHV
            from hp_policy import HybridPilot
            import evaluate as EV
            vv = get_vehicle(spec)
            phev = vv.arch.startswith('phev')
            case = dict(spec=spec, trip=trip, soc0=cb['soc0'], temps=cb['temps'])
            pts, outs, stage = dp_curve(spec, trip, cb['soc0'], cb['temps'], dp_targets(vv, cb['soc0'], phev), veh=vv)
            nat = cb['soc0'] if not (phev and cb['soc0'] > vv.battery.soc_lo + 0.08) else vv.battery.soc_lo + 0.05
            jn = int(np.argmin([abs(p['target'] - nat) for p in pts]))
            runs = {'dp': (outs[pts[jn]['target']][0], dict(pts[jn]))}
            names = {'rule': EV.RULE[vv.arch], 'ecms': 'ecms', 'aecms': 'aecms'}
            for nm in list(names) + list(pols):
                stg = HybridPilot(**pols[nm]) if nm in pols else names[nm]
                o = simulate(spec, trip, strategy=stg, init_soc=cb['soc0'], init_temps=cb['temps'], record=True, options='roadsim-physics-v1')
                runs[nm] = (o, metrics(o, vv.arch, cb['soc0']))
                if phev:
                    add_matched(case, vv, stage, pts, runs[nm][1]['soc_end'])
            for nm, (o, m) in runs.items():
                grade(m, pts, LHV.get(vv.fuel, 42.9e3) / 1e3, cb['soc0'])
                on = (o['eng_on'] > 0.5).astype(np.uint8)
                soc = np.clip(np.round(o['soc'] * 255), 0, 255).astype(np.uint8)
                cf = np.cumsum(o['fuel_rate'])
                rec['runs'][f"{v['tag']}|{nm}"] = dict(on=b64(np.packbits(on)), soc=b64(soc), fuel=b64(np.round(cf * 10).astype(np.uint32)),
                                                       fuel_g=float(o['summary']['fuel_g']), E_kJ=float(m['E_kJ']), soc_end=float(o['soc'][-1]),
                                                       pct_dp=float(m['pct_dp']), starts_per_km=float(m['starts_per_km']),
                                                       chatter_per_km=float(m['chatter_per_km']), L100=float(o['summary'].get('L_per_100km', 0.0)),
                                                       dist_km=float(m['dist_km']))
            print(cb['key'], v['tag'], {k.split('|')[1]: round(x['pct_dp'] * 100, 2) for k, x in rec['runs'].items() if k.startswith(v['tag'] + '|')}, flush=True)
        out['combos'].append(rec)
    json.dump(out, open(f'{ROOT}/web/lab.json', 'w'))
    print('size', os.path.getsize(f'{ROOT}/web/lab.json'))


if __name__ == '__main__':
    run(sys.argv[1], json.loads(sys.argv[2]))
