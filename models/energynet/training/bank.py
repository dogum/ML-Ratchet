"""Simulated bank: fleet vehicles x routes (VED train_noise routes, synthetic drives, dynamometer cycles).
The network sees the OBSERVED trip (station temperature and wind); the simulator drives the TRUE conditions:
temperature + N(0, 1.5) C, wind x U(0.4, 0.9) (airport 10 m wind vs car height), so weather uncertainty is in the data.

python bank.py <fleet name> <examples per vehicle> <seed>   -> data/bank_<fleet>/chunk_XXX.npz (resumable)
"""
import sys, os, json, time, glob
from common import *
import numpy as np
import realdata as rd
from roadsim import make_trip
from roadsim.vehicle import fill_spec
import routes, features as F, wrappers

TR_KEYS = ('t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface')
LD_CYC = ['UDDS', 'FTP', 'HWFET', 'US06', 'SC03', 'NYCC', 'LA92', 'LA92_short', 'IM240', 'EUDC', 'JP_10_15', 'WLTC_3b', 'FTP_cold_20F']
HD_CYC = ['LA92_HD_class3', 'HD_transient_1037A', 'HD_cruise_55mph_1037D', 'HD_cruise_65mph_1037D', 'HD_cruise_55mph_flat',
          'FASTSim_long_haul', 'NYCC', 'UDDS', 'cbd14']
SRC = ['ved', 'synth', 'dyno', 'cyc_road']
_VED = None; _CYC = None


def ved_routes():
    global _VED
    if _VED is None:
        _VED = []
        for meta, arr in rd.load_trips('train_noise', route_ok=True):
            _VED.append({k: arr[k] for k in TR_KEYS})
    return _VED


def cycles():
    global _CYC
    if _CYC is None:
        _CYC = {}
        allc = rd.cycles()
        for n in set(LD_CYC + HD_CYC):
            if n == 'cbd14':
                tr = make_trip('cbd14'); _CYC[n] = {k: np.asarray(tr[k], np.float32) for k in tr if k in TR_KEYS}
            else:
                _CYC[n] = {k: allc[n][1][k] for k in TR_KEYS if k in allc[n][1]}
    return _CYC


def observed_and_true(trip, rng, dyno=False):
    """observed trip (what the network sees) and the true one (what the car drives)."""
    obs = {k: np.asarray(v).copy() for k, v in trip.items()}
    tru = {k: np.asarray(v).copy() for k, v in trip.items()}
    if not dyno:
        tru['T_amb'] = (obs['T_amb'] + rng.normal(0, 1.5)).astype(np.float32)
        tru['wind_speed'] = (obs['wind_speed'] * rng.uniform(0.4, 0.9)).astype(np.float32)
        tru['rho_air'] = (obs['rho_air'] * (273.15 + obs['T_amb']) / (273.15 + tru['T_amb'])).astype(np.float32)
    return obs, tru


def reweather(trip, rng):
    """new weather on a real route: temperature offset, wind redraw."""
    t = dict(trip)
    T, ws, wd, elev0, surf = routes.weather(rng)
    t['T_amb'] = (trip['T_amb'] - trip['T_amb'].mean() + T).astype(np.float32)
    t['wind_speed'] = np.full(len(trip['v']), ws, np.float32); t['wind_dir'] = np.full(len(trip['v']), wd, np.float32)
    t['rho_air'] = routes.rho_air(t['T_amb'], trip['elev']).astype(np.float32)
    t['surface'] = np.full(len(trip['v']), surf, np.int8)
    return t


def draw_route(rng, cls):
    hd = cls in HD
    u = rng.random()
    if hd:
        if u < 0.8:
            tr, kind = routes.synth_trip(rng, cls); return tr, 'synth', False
        cy = cycles()
        names = ['LA92_HD_class3', 'NYCC', 'UDDS', 'cbd14', 'HD_transient_1037A'] if cls == 'bus' else HD_CYC
        n = names[int(rng.integers(len(names)))]
        tr = cy[n]
        if n == 'FASTSim_long_haul':
            L = int(rng.integers(1200, 3600)); s0 = int(rng.integers(0, len(tr['v']) - L))
            tr = {k: v[s0:s0 + L] for k, v in tr.items()}; tr['t'] = np.arange(L, dtype=np.float32)
        tr = reweather(_fill(tr), rng) if rng.random() < 0.7 else _fill(tr)
        return tr, 'cyc_road', False
    if u < 0.45:
        R = ved_routes(); tr = R[int(rng.integers(len(R)))]
        tr = reweather(tr, rng) if rng.random() < 0.4 else dict(tr)
        return tr, 'ved', False
    if u < 0.80:
        tr, kind = routes.synth_trip(rng, cls); return tr, 'synth', False
    cy = cycles(); n = LD_CYC[int(rng.integers(len(LD_CYC)))]
    tr = _fill(cy[n])
    if u < 0.93:
        T = 35.0 if n == 'SC03' else (-6.7 if n == 'FTP_cold_20F' else 24.0)
        tr = dict(tr); N = len(tr['v'])
        tr.update(T_amb=np.full(N, T, np.float32), wind_speed=np.zeros(N, np.float32), grade=np.zeros(N, np.float32),
                  surface=np.zeros(N, np.int8), rho_air=routes.rho_air(np.full(N, T), np.zeros(N)).astype(np.float32))
        return tr, 'dyno', True
    tr = reweather(tr, rng)
    N = len(tr['v'])
    x = np.concatenate([[0], np.cumsum(tr['v'][1:])])
    tr['grade'] = routes.grade_profile(rng, x, float(rng.uniform(0.003, 0.03)), rng.uniform(200, 1000)).astype(np.float32)
    tr['heading'] = np.full(N, rng.uniform(0, 360), np.float32)
    return tr, 'cyc_road', False


def _fill(tr):
    N = len(tr['v']); t = dict(tr)
    d = dict(t=np.arange(N, dtype=np.float32), grade=0.0, elev=0.0, heading=0.0, T_amb=24.0, wind_speed=0.0, wind_dir=0.0, surface=0)
    for k, x in d.items():
        if k not in t or t[k] is None or (np.asarray(t[k]).dtype.kind == 'f' and np.isnan(t[k]).all()):
            t[k] = x if isinstance(x, np.ndarray) else np.full(N, x, np.int8 if k == 'surface' else np.float32)
    if 'rho_air' not in t or np.isnan(t['rho_air']).all():
        t['rho_air'] = routes.rho_air(t['T_amb'], t['elev']).astype(np.float32)
    return t


def init_state(rng, spec, full, trip, src, dyno, name=None):
    arch = spec['architecture']
    b = full.get('battery') or {}
    lo, hi = (b.get('soc_window') or [0.2, 0.9])
    moving = trip['v'][0] > 1.0
    if dyno:
        cold = (name in ('UDDS', 'FTP', 'FTP_cold_20F')) and rng.random() < 0.6
    elif moving:
        cold = False
    else:
        cold = rng.random() < (0.4 if src == 'ved' else 0.45)
    temps = None if cold else warm_temps(float(trip['T_amb'][0]))
    soc = disp = None
    if arch == 'conv':
        pass
    elif arch.startswith('phev'):
        disp = 0.0 if rng.random() < 0.3 else float(rng.uniform(0.05, 1.0))
        soc = lo + 0.02 if disp == 0 else lo + disp * (hi - lo)
    elif arch == 'bev':
        disp = float(rng.uniform(0.15, 1.0)); soc = lo + disp * (hi - lo)
    else:
        soc = float(rng.uniform(lo + 0.05, hi - 0.05)); disp = (soc - lo) / (hi - lo)
    return temps, cold, soc, disp


def veh_record(spec, full):
    cls, arch = spec['class'], spec['architecture']
    e = full.get('engine') or {}; tr = full.get('transmission') or {}; b = full.get('battery') or {}
    m_true = float(full['mass_kg']) + float(full.get('payload_kg') or 0)
    LB500 = 226.796
    m_rep = round((full['mass_kg'] - 136.0) / LB500) * LB500 + 136.0 if cls not in HD else full['mass_kg'] * float(np.random.default_rng(len(spec['id'])).uniform(0.95, 1.05))
    return dict(arch=ARCHS.index(arch), cls=CLASSES.index(cls), m_true=m_true, m_rep=float(m_rep), cda=float(full['CdA_m2']), crr=float(full['Crr']),
                cv=float(full.get('Crr_speed_coef') or 0.0), disp=float(e.get('displacement_L') or np.nan) if arch != 'bev' else np.nan,
                power=float(e.get('max_power_kW') or np.nan) if arch != 'bev' else float(sum(m.get('max_power_kW', 0) for m in full.get('motors') or [] if m.get('role') in ('traction', 'MG2', 'P2'))),
                diesel=float(e.get('fuel') == 'diesel'), trans=F.TRANS.index(tr.get('type')) if tr.get('type') in F.TRANS else -1,
                ngears=float(len(tr['gears'])) if tr.get('gears') else np.nan, cap=float(b.get('capacity_kWh') or np.nan),
                artic=float(bool(spec.get('articulated'))))


def vehicle_examples(spec, lab, K, rng, vid):
    rows = []
    full = fill_spec(spec); rec = veh_record(spec, full)
    cls, arch = spec['class'], spec['architecture']
    for k in range(K):
        try:
            tr, src, dyno = draw_route(rng, cls)
            obs, tru = observed_and_true(tr, rng, dyno)
            temps, cold, soc, disp = init_state(rng, spec, full, tru, src, dyno)
            out = wrappers.simulate(spec, tru, init_soc=soc, init_temps=temps, record=True, rng=int(rng.integers(1 << 30)))
            tg = wrappers.targets(out, spec, tru['T_amb'], dyno)
        except Exception as ex:
            print('fail', spec['id'], ex, flush=True); continue
        P = F.prep(obs)
        Rv = F.route_vec(P)
        PH = F.variants(P, cls, rec['m_true'], rec['cda'], rec['crr'], rec['cv'], rec['m_rep'])
        L = out['ledger']
        comp = [L['aero'], L['rolling'], L['brake'], L['aux'], L['eng_loss'] + L['mot_loss'] + L['trans_loss'] + L['batt_loss'],
                L['grade'], L['kinetic_change'], L['fuel'], L['batt_chem_out']]
        rows.append(dict(R=Rv.astype(np.float32), PH=PH.astype(np.float32), comp=np.array(comp, np.float32) / 1e3,
                         tg=np.array([tg['E_raw'], tg['E_rw'], tg['E_maf_raw'], tg['E_maf_rw']], np.float32),
                         soc=np.nan if disp is None else disp, cold=float(cold), warm=float(not cold), dyno=float(dyno),
                         src=SRC.index(src), dur=float(len(obs['v'])), dist=float(P['v'].sum()),
                         lab=np.array([lab.get('label_city_fuel', np.nan), lab.get('label_hwy_fuel', np.nan),
                                       lab.get('label_city_elec', np.nan), lab.get('label_hwy_elec', np.nan)], np.float32),
                         trace_miss=float(out['summary']['trace_rms_mps']), vid=vid, **rec))
    return rows


def main(fleet, K, seed):
    fdir = f'{EN}/data/fleet_{fleet}'; odir = f'{EN}/data/bank_{fleet}'; os.makedirs(odir, exist_ok=True)
    chunks = sorted(glob.glob(f'{fdir}/chunk_*.json'))
    for cp in chunks:
        ci = int(cp[-8:-5]); op = f'{odir}/chunk_{ci:03d}.npz'
        if os.path.exists(op): continue
        rng = np.random.default_rng([seed, ci])
        rows = []; t0 = time.time()
        for vi, V in enumerate(json.load(open(cp))):
            rows += vehicle_examples(V['spec'], V['labels'] or {}, K, rng, ci * 100 + vi)
        D = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(op + '.tmp.npz', **D); os.replace(op + '.tmp.npz', op)
        print('bank chunk', ci, len(rows), round(time.time() - t0, 1), flush=True)


ARCHS, CLASSES = F.ARCHS, F.CLASSES
if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
