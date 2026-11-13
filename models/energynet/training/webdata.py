"""Data for the web page: weights pack (float16), lab presets and routes, real-data explorer, exam tables,
ratchet, twin calibration, and JS parity cases. python webdata.py <member1,member2,...>"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.VEHICLE)
import numpy as np, pandas as pd
import features as F, routes, bank
from model_np import ENet, Ensemble
import realdata as rd

EN = _P.WORK
members = sys.argv[1].split(',')
MI = 1609.344


def pack(paths):
    """float16 weights of all members, base64, with a manifest."""
    man, arrs, off = [], [], 0
    for j, p in enumerate(paths):
        z = np.load(p)
        for k in z.files:
            if k == 'meta': continue
            a = z[k].astype(np.float16).ravel()
            man.append(dict(k=f'm{j}_{k}', off=off, shape=list(z[k].shape))); arrs.append(a); off += a.size
    allw = np.concatenate(arrs)
    meta = json.loads(str(np.load(paths[0])['meta'])); meta['n_members'] = len(paths)
    return dict(b64=base64.b64encode(allw.tobytes()).decode(), man=man, meta=meta)


def f16_ensemble(paths):
    """numpy ensemble with float16-rounded weights (what the browser runs), for parity checks."""
    E = Ensemble(paths)
    for m in E.m:
        m.mu = m.mu.astype(np.float16).astype(float); m.sd = m.sd.astype(np.float16).astype(float)
        m.L = [(W.astype(np.float16).astype(float), b.astype(np.float16).astype(float)) for W, b in m.L]
        m.H = (m.H[0].astype(np.float16).astype(float), m.H[1].astype(np.float16).astype(float))
    return E


def features_js(trip, veh):
    """mirror of enet.js features() (same inputs)."""
    P = F.prep(trip); cls = veh['cls']; hd = cls in F.HD
    mc, Cc, Rc, cvc, rot = F.CLS[cls]
    ml = veh['mLevel'] if veh.get('m') else 0; m = veh.get('m') or mc
    rl = bool(veh.get('cda')) and ml == 2
    cda = veh['cda'] if rl else Cc * (m / mc) ** 0.5; crr = veh['crr'] if rl else Rc
    cv = (veh.get('cv') if veh.get('cv') is not None else cvc) if rl else cvc
    ph = F.phys_ints(P, m, cda, crr, cv, rot, hd)
    dist = P['v'].sum(); dur = P['N']
    soc = veh.get('soc'); cap = veh.get('cap')
    E, Eh, Eb, f = F.e_phys(ph, veh['arch'], m, dur, soc, cap, hd)
    V = F.vehicle_vec(veh['arch'], cls, m, ml, cda, crr, rl, veh.get('disp'), veh.get('power'), veh.get('diesel'), veh.get('trans'),
                      veh.get('ngears'), cap, tuple(veh['labFuel']) if veh.get('labFuel') else None,
                      tuple(veh['labElec']) if veh.get('labElec') else None, soc, veh.get('warm'), veh.get('cold'), veh.get('maf'),
                      veh.get('dyno'), (E, dist / 1000), dist / max(dur, 1))
    return np.concatenate([F.route_vec(P), F.phys_vec(ph, E, Eh, Eb, f, m, dist), V]), E


def mpg2(mpg, d=745.0): return round(d * 3.785411784 * 0.042 / (mpg * MI / 1000), 4)
def kwh2(k): return round(k * 3.6 / (MI / 10), 4)


PRESETS = [
    dict(id='sedan', name='Compact sedan, 1.8 L, 5-speed automatic', note='Dev car VED_149: what VED knows (mass rounded to 500 lb, engine size, window-sticker mpg)',
         veh=dict(arch='conv', cls='car', m=1270, mLevel=1, disp=1.8, trans='AT', labFuel=[mpg2(25), mpg2(34)], warm=True)),
    dict(id='hybrid', name='Hybrid hatchback, power-split, 1.8 L', note='Dev car VED_241 spec',
         veh=dict(arch='hev_powersplit', cls='car', m=1497, mLevel=1, disp=1.8, trans='eCVT', labFuel=[mpg2(48), mpg2(43)], warm=True)),
    dict(id='erev', name='Plug-in hybrid, series (range extender), 1.5 L', note='Dev car VED_537 spec; set the start charge',
         veh=dict(arch='phev_series', cls='car', m=1724, mLevel=1, disp=1.5, trans='single', labFuel=[mpg2(43), mpg2(42)], labElec=[kwh2(29), kwh2(34)], soc=0.8, warm=True)),
    dict(id='ev', name='Battery-electric compact, 24 kWh', note='Dev car VED_455 spec',
         veh=dict(arch='bev', cls='car', m=1724, mLevel=1, trans='single', cap=24, labElec=[kwh2(26), kwh2(33)], soc=0.8, warm=True)),
    dict(id='pickup', name='Full-size pickup, 5.3 L V8', note='Dev car VED_250 spec',
         veh=dict(arch='conv', cls='pickup', m=2631, mLevel=1, disp=5.3, labFuel=[mpg2(14), mpg2(19)], warm=True)),
    dict(id='suv', name='Mid-size SUV with a full EPA-style spec', note='Example values: exact test mass, road-load A/B/C, engine, 8-speed automatic',
         veh=dict(arch='conv', cls='suv', m=2100, mLevel=2, cda=0.77, crr=0.0080, cv=0.012, disp=2.5, power=150, trans='AT', ngears=8, warm=True)),
    dict(id='bus', name='40-ft diesel city bus', note='Class and fuel only; mass and road load unknown',
         veh=dict(arch='conv', cls='bus', diesel=True, trans='AT', warm=True)),
    dict(id='ebus', name='Battery-electric city bus', note='Class only; mass unknown',
         veh=dict(arch='bev', cls='bus', trans='single', soc=0.8, warm=True)),
    dict(id='truck', name='Class 8 tractor-trailer, 34 t', note='Dev truck BIC-US spec: test mass, 14.5 L diesel, 12-speed AMT',
         veh=dict(arch='conv', cls='truck_hd', m=34408, mLevel=2, disp=14.5, power=315, diesel=True, trans='AMT', ngears=12, warm=True)),
]


def q(a, d=2): return [round(float(x), d) for x in a]


def route_pack(name, tr, kind):
    N = len(tr['v'])
    T = np.asarray(tr.get('T_amb', np.full(N, 20.0)), float)
    return dict(name=name, kind=kind, v=q(tr['v'], 2), grade=q(np.nan_to_num(tr.get('grade', np.zeros(N))), 4),
                heading=q(np.nan_to_num(tr.get('heading', np.zeros(N))), 0), T0=round(float(T.mean()), 1),
                dist_km=round(float(np.sum(tr['v'])) / 1000, 2), dur_s=N)


def lab_routes():
    cy = rd.cycles(); R = []
    for n, lab in [('UDDS', 'City test cycle (UDDS, 12 km)'), ('HWFET', 'Highway test cycle (HWFET, 16.5 km)'), ('US06', 'Aggressive test cycle (US06, 12.9 km)'), ('NYCC', 'Stop-and-go (NYCC, 1.9 km)')]:
        R.append(route_pack(lab, cy[n][1], 'cycle'))
    vr = list(rd.load_trips('train_noise', route_ok=True, min_km=8, limit=40, seed=7))
    best = sorted(vr, key=lambda x: -x[0]['elev_gain_m'] / x[0]['distance_m'])
    m1 = [x for x in vr if 600 < x[0]['duration_s'] < 1300][0]
    R.append(route_pack(f"Ann Arbor drive ({m1[0]['distance_m'] / 1000:.1f} km, real logged speed and lidar grade)", m1[1], 'real'))
    m2 = best[0]
    R.append(route_pack(f"Ann Arbor hilly drive ({m2[0]['distance_m'] / 1000:.1f} km, {m2[0]['elev_gain_m']:.0f} m of climb)", m2[1], 'real'))
    rng = np.random.default_rng(5)
    tr, _ = routes.synth_trip(rng, 'bus', 'bus'); R.append(route_pack(f"Bus route, synthetic ({tr['v'].sum() / 1000:.1f} km, many stops)", tr, 'synthetic'))
    lh = cy['FASTSim_long_haul'][1]; s0 = 20000; L = 2400
    tr = {k: v[s0:s0 + L] for k, v in lh.items() if isinstance(v, np.ndarray) and len(v) == len(lh['v'])}
    R.append(route_pack(f"Interstate with hills (40 min window of a long-haul day, {tr['v'].sum() / 1000:.0f} km)", tr, 'cycle'))
    return R


def trip_for_js(r, T, ws, wdir_rel, goff, rho=None):
    N = len(r['v'])
    head = np.array(r['heading'], float)
    return dict(v=np.array(r['v']), grade=np.array(r['grade']) + goff, heading=head, T_amb=np.full(N, T), wind_speed=np.full(N, ws),
                wind_dir=(head + wdir_rel) % 360, rho_air=np.full(N, 101325.0 / (287.05 * (T + 273.15))), surface=np.zeros(N))


def main():
    paths = [f'{EN}/ckpt/{m}.npz' for m in members]
    W = pack(paths)
    E16 = f16_ensemble(paths)
    LR = lab_routes()
    # parity cases (JS must reproduce these)
    par = []
    for (pi, ri, T, ws, wd, g) in [(0, 0, 24, 0, 0, 0), (1, 4, -10, 6, 0, 0), (2, 5, 30, 3, 180, 0.01), (3, 6, 5, 8, 90, -0.01), (8, 7, 15, 5, 0, 0), (6, 6, 28, 0, 0, 0)]:
        veh = PRESETS[pi]['veh']; tr = trip_for_js(LR[ri], T, ws, wd, g)
        x, E = features_js(tr, veh)
        p, s, c = E16.predict(x[None], np.array([E]))
        par.append(dict(preset=pi, route=ri, T=T, ws=ws, wd=wd, goff=g, E=float(p[0]), sig=float(s[0]), Ephys=float(E), x=q(x, 6)))
    # explorer: fixed random draw within strata (seed 2026)
    ex = pd.read_parquet(f'{EN}/data/exam_ved.parquet')
    ex['stratum'] = np.where(ex.arch == 'hev_parallel_p2', 'P2', ex.group)
    pick = pd.concat([d.sample(min(len(d), 9 if g != 'P2' else 3), random_state=2026) for g, d in ex.groupby('stratum')])
    trips = []
    for r in pick.itertuples():
        meta, arr = rd.load_trip(r.trip_id)
        trips.append(dict(id=r.trip_id, veh=r.vehicle_id, stratum=r.stratum, arch=r.arch, cls=r.cls, unit=r.unit, conv=r.conv,
                          T=round(r.T, 1), km=round(r.dist_km, 2), dur=int(r.dur_s), climb=round(r.climb, 0), y=r.y, pred=r.pred, sig=r.sig * r.k,
                          b_label=r.b_label, b_vsp=r.b_vsp, b_class=r.b_class, comps=[r.c_aero, r.c_rolling, r.c_brake, r.c_aux, r.c_losses],
                          v=q(arr['v'][::2], 1), elev=q(np.nan_to_num(arr['elev'][::2]), 1), date=meta['date'][:10], hour=meta.get('hour_local')))
    sc = pd.concat([d.sample(min(len(d), 500), random_state=1) for g, d in ex.groupby('group')])
    scatter = {g: dict(y=q(d.y, 4), p=q(d.pred, 4), vsp=q(d.b_vsp, 4)) for g, d in sc.groupby('group')}
    eo = pd.read_parquet(f'{EN}/data/exam_epa.parquet')
    eo = pd.concat([d.sample(min(len(d), 300), random_state=1) for g, d in eo.groupby(['arch', 'test'])])
    epa_sc = {f'{a}_{t}': dict(y=q(d.y, 5), p=q(d.pred, 5)) for (a, t), d in eo.groupby(['arch', 'test'])}
    hd = pd.read_parquet(f'{EN}/data/exam_hd.parquet')
    z = hd[hd.set == 'ZTBus_leg']
    zt = dict(y=q(z.y / 3.6, 2), p=q(z.pred / 3.6, 2), km=q(z.dist_km, 2), split=list(z.split), T=q(z['T'], 1))
    mt = pd.read_parquet(f'{EN}/data/exam_ztbus_missions.parquet')
    ms = hd[hd.set == 'Mersin'].groupby('trip_id').agg(p=('pred', 'mean'), y=('y', 'first'), fuel=('fuel', 'first'), split=('split', 'first'), dur=('dur_s', 'first'))
    ms['p'] = ms.p * np.where(ms.fuel == 'cng', 0.042 / 0.0471, 1.0)
    out = dict(weights=W, presets=PRESETS, routes=LR, parity=par, trips=trips, scatter=scatter, epa_scatter=epa_sc,
               ztbus=zt, ztbus_missions=dict(y=q(mt.y / 3.6, 1), p=q(mt.pred / 3.6, 1), b=q(mt.base / 3.6, 1)),
               mersin=dict(y=q(ms.y / 0.042 / 1000, 2), p=q(ms.p / 0.042 / 1000, 2), fuel=list(ms.fuel), split=list(ms.split)),
               exam=json.load(open(f'{EN}/exam.json')), ratchet=json.load(open(f'{EN}/ratchet.json')),
               twin=json.load(open(f'{EN}/data/twin_summary.json')), wrappers=dict(rw=__import__('wrappers').CAL['rw'], maf=__import__('wrappers').CAL['maf']))
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        return o
    out = clean(out)
    json.dump(out, open(f'{EN}/web/data.json', 'w'), separators=(',', ':'), allow_nan=False)
    print('data.json', len(json.dumps(out)) / 1e6, 'MB; weights', len(W['b64']) / 1e6, 'MB')


if __name__ == '__main__':
    main()
