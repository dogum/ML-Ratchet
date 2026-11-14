"""Check (dev only): RoadSim twins of the VED dev cars on their own logged trips, through the OBD measurement model.
Writes ved_twin.json (or ved_twin__<preset>.json).

Twin of each VED dev car (data/proc/splits.json; 8 ICE, 5 HEV, Volt, Ford Energi, Leaf):
  * spec: RoadData's partial spec (realdata.load_vehicle; mass = VED curb weight + 136 kg payload added once by
    RoadSim - the 13 Nov double count is fixed in RoadData) completed generically: rated power from displacement
    (55 kW/L NA, 75 kW/L boosted, 45 kW/L hybrid), class defaults for CdA/Crr, published plug-in hardware;
  * label matched (EnergyNet's vehfit method, now on the EPA procedures and the run's physics options): road-load
    scale k (CdA and Crr x k; sets the highway/city ratio) and engine efficiency scale (sets the level) so that the
    twin reproduces the fueleconomy.gov UNADJUSTED city / highway values RoadData matched to the car (FTP-75 and HWFET
    on a chassis dyno, HVAC off). PHEVs are matched in charge-sustaining operation (simulated as their HEV
    counterpart, which is what the CS label measures); the Leaf matches k only (MPGe level, no engine);
  * driven on every q_label_ok dev trip of the car: logged speed (from the logged initial speed), lidar grade,
    airport weather (temperature, wind, density, wet/snow surface), road (not dyno) rolling resistance, HVAC on
    (automatic), warm initial temperatures (VED trips are warm fragments: coolant 85 C, oil 90 C, transmission 70 C,
    warm tyres, cabin 21 C), logged initial SOC for cars with a battery.
Compared on the trip's valid seconds:
  fuel        conv / HEV / PHEV: simulated fuel through the OBD label model (roadsim.obd.label_fuel: MAF fuel-cut air
              and missed enrichment, for MAF / absolute-load sources) vs VED's estimated fuel; also the raw simulated
              fuel ('fuel_raw') for reference
  batt_kWh    PHEV / BEV: net DC battery energy (+ = discharge)
  energy      PHEV: fuel LHV + battery (the label RoadData uses)
  eng_on      share of valid seconds with the engine turning (logged rpm > 300; cars with an rpm channel)
  idle_fuel   mean fuel rate at standstill with the engine on (g/s), simulated (label) vs logged
Per trip error = sim/real - 1 (%); per car: median over trips and the ratio of totals; per architecture: median and IQR
of trip errors and the median over cars. Trips with a label below 2 g fuel / 0.05 kWh are skipped.

usage: python check_ved_twin.py [--options roadsim-physics-v2] [--workers 14] [--vehicles VED_546,...] [--no-match]
"""
import os, sys, json, time, argparse, copy, math
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(VEH, 'roadsim')); sys.path.insert(0, VEH)
import _opts
from epa_procedures import ftp, hot_cycle, test_spec

LHV = 0.042            # MJ/g (E10, RoadData's label definition)
WARM = dict(engine=85.0, oil=90.0, driveline=70.0, axle=60.0, tire_warm=1.0, cabin=21.0)
# Plug-in hardware (public spec sheets), used only for architectures RoadData identified (as EnergyNet's twins)
PLUGIN_HW = {
    # battery max_power_kW: min(traction machines / 0.9, 9 C), the plug-in rule of data/scripts/d3_specs.py (RoadSim's
    # 'energy' default of 3 C left the Energi twin at 23 kW and forced its engine on in charge depletion). Machines and
    # engine power from the same public spec sheets as data/d3_facts.json (added 13 Nov 2026, calibration agent 2).
    ('phev_series', 1.4): dict(battery=dict(capacity_kWh=16.5, nominal_V=360, soc_window=[0.2, 0.85], max_power_kW=123.0), P=111,
                               engine_kW=63, motors=[('traction', 111, 370), ('generator', 55, None)]),
    ('phev_series', 1.5): dict(battery=dict(capacity_kWh=18.4, nominal_V=360, soc_window=[0.15, 0.9], max_power_kW=123.0), P=111,
                               engine_kW=75, motors=[('traction', 111, 398), ('generator', 75, None)]),
    ('phev_powersplit', 2.0): dict(battery=dict(capacity_kWh=7.6, nominal_V=300, soc_window=[0.15, 0.95], max_power_kW=68.4), P=145,
                                   engine_kW=105, motors=[('MG1', 70, None), ('MG2', 88, 240)]),
    ('phev_powersplit', 1.8): dict(battery=dict(capacity_kWh=8.8, nominal_V=350, soc_window=[0.15, 0.95], max_power_kW=79.2), P=90,
                                   engine_kW=71, motors=[('MG1', 23, None), ('MG2', 53, 163)]),
}
# Mild (P1) hybrids that RoadData labels hev_powersplit (VED has no P1 type): Honda CR-Z IMA, run as a P2 machine that
# never drives electrically (strategy parameter engine_on_moving; the D3 Insight / S400h / Civic / CR-Z twins do the same)
MILD_HW = {
    'CR-Z': dict(engine_kW=84, transmission={'type': 'AMT'}, motors=[{'role': 'P2', 'max_power_kW': 10, 'max_torque_Nm': 78,
                 'eff_map': 'ipm_generic_150', 'max_rpm': 7000}], battery={'capacity_kWh': 0.58, 'nominal_V': 101, 'chemistry': 'NiMH',
                 'soc_window': [0.3, 0.8]}),
}
_TRANS = {'AT': 'AT', 'CVT': 'CVT', 'AMT': 'AMT', 'DCT': 'DCT', 'manual': 'AMT', 'single': 'single', 'eCVT': 'eCVT'}


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    if isinstance(d, np.floating):
        return float(d)
    if isinstance(d, np.integer):
        return int(d)
    return d


def ved_spec(vid):
    """RoadData partial spec -> runnable generic spec (no fitting)."""
    import realdata as rd
    p = rd.load_vehicle(vid)
    labels = p.get('labels_fueleconomy') or {}
    s = _clean(copy.deepcopy(p))
    for k in ('meta', 'labels_fueleconomy', 'split', 'source', 'mass_note', 'results', 'arch_source'):
        s.pop(k, None)
    arch, cls = s['architecture'], s['class']
    eng = s.get('engine') or {}
    if arch == 'bev':
        s.pop('engine', None)
    elif eng:
        d = eng.get('displacement_L')
        if not eng.get('max_power_kW') and d:
            eng['max_power_kW'] = float(d) * (75 if eng.get('boosted') else (45 if arch != 'conv' else 55))
        for k in ('cylinders', 'boosted'):
            eng.pop(k, None)
        if eng.get('fuel') not in ('gasoline', 'diesel'):
            eng['fuel'] = 'gasoline'
        s['engine'] = eng
    mild = next((MILD_HW[k] for k in MILD_HW if k in str(labels.get('example_models') or '')), None)
    if mild is not None and arch.startswith('hev'):
        arch = s['architecture'] = 'hev_parallel_p2'
        s['engine']['max_power_kW'] = float(mild['engine_kW'])
        s['motors'] = copy.deepcopy(mild['motors']); s['battery'] = dict(mild['battery'])
        s['control'] = {'strategy': 'aecms', 'params': {'engine_on_moving': True}}
    tr = {k: v for k, v in (s.get('transmission') or {}).items() if k in ('type',)}
    if mild is not None and arch == 'hev_parallel_p2':
        tr = dict(mild['transmission'])
    if tr.get('type') in _TRANS:
        tr['type'] = _TRANS[tr['type']]
    if arch in ('hev_powersplit', 'phev_powersplit'):
        tr = {'type': 'eCVT'}
    if arch in ('bev', 'phev_series', 'hev_series'):
        tr = {'type': 'single'}
    if tr.get('type'):
        s['transmission'] = tr
    else:
        s.pop('transmission', None)
    if arch.startswith('phev'):
        hw = PLUGIN_HW.get((arch, (eng or {}).get('displacement_L')))
        if hw:
            s['battery'] = dict(hw['battery'], chemistry='NMC')
            s['engine']['max_power_kW'] = float(hw['engine_kW'])
            s['motors'] = [dict(role=r_, max_power_kW=float(P_), eff_map='ipm_generic_150', max_rpm=13500 if r_ != 'MG2' else 14000,
                                **({'max_torque_Nm': float(T_)} if T_ else {})) for r_, P_, T_ in hw['motors']]
    elif arch == 'bev':
        cap = (s.get('battery') or {}).get('capacity_kWh', 24.0)
        s['battery'] = {'capacity_kWh': cap, 'nominal_V': 360, 'chemistry': 'NMC', 'soc_window': [0.02, 0.98]}
        s['motors'] = [{'role': 'traction', 'max_power_kW': 80.0 if cls == 'car' else 150.0, 'eff_map': 'ipm_generic_150'}]
    elif mild is None:
        s.pop('battery', None)
    for k in ('motors', 'aux', 'regen', 'control', 'CdA_m2', 'Crr', 'Crr_speed_coef', 'road_load_ABC', 'wheel_radius_m',
              'final_drive', 'driveline_eff'):
        if not s.get(k):
            s.pop(k, None)
    return s, labels


# ------------------------------------------------------------------ label matching
def _scaled(spec, eff, k):
    from roadsim.vehicle import fill_spec
    s = copy.deepcopy(spec)
    f = fill_spec(copy.deepcopy(spec))
    s['CdA_m2'] = f['CdA_m2'] * k; s['Crr'] = f['Crr'] * k
    if s.get('engine') is not None:
        s['engine']['eff_scale'] = eff
    return s


def _dyno(spec):
    s = test_spec(spec)
    s['tire'] = dict(s.get('tire') or {}, K_t=0.0, temp_effects=False)
    return s


def _lab_proxy(spec):
    """the spec the label is measured on: PHEVs in charge-sustaining operation = their HEV counterpart."""
    a = spec['architecture']
    if a.startswith('phev'):
        s = copy.deepcopy(spec); s['architecture'] = 'hev_' + a.split('_', 1)[1]
        return s
    return spec


def city_hwy(spec):
    s = _dyno(_lab_proxy(spec))
    if spec['architecture'] == 'bev':
        return hot_cycle(s, 'udds')['MPGe'], hot_cycle(s, 'hwfet')['MPGe']
    return ftp(s)['FE'], hot_cycle(s, 'hwfet')['FE']


def label_match_phev(spec, ucity, uhwy, ev_city_kWh=None, ev_hwy_kWh=None):
    """plug-ins: class-default road load (k = 1), engine level from the charge-sustaining label. One k scales CdA and
    Crr together; neither label ratio fixes it for the Ford Energi (charge-sustaining 51.4/55 needs k at the 1.8 bound
    with eff_scale 1.31; the electric label 42/34 kWh needs k 1.6 -> +31 % trip energy), so the ratio is not matched.
    The Volt's electric ratio gives k 0.84. Decided on VED dev, 13 Nov 2026 (CALIBRATION.md, rung H6)."""
    k = 1.0
    eff = 1.0
    for _ in range(3):
        c, h = city_hwy(_scaled(spec, eff, k))
        eff *= float(math.sqrt(ucity * uhwy / (c * h)))
    s = _scaled(spec, eff, k)
    c3, h3 = city_hwy(s)
    return s, dict(k=k, eff=eff, city=c3, hwy=h3, ucity=ucity, uhwy=uhwy, k_from='fixed 1 (plug-in)')


def label_match(spec, ucity, uhwy, ks=(0.75, 1.0, 1.35, 1.8)):
    tgt = math.log(uhwy / ucity)
    if spec['architecture'] == 'bev':        # level only, through k
        lv = []
        for k in ks:
            c, h = city_hwy(_scaled(spec, 1.0, k)); lv.append(math.log(c * h))
        o = np.argsort(lv)
        k = float(np.exp(np.interp(math.log(ucity * uhwy), np.array(lv)[o], np.log(ks)[o])))
        s = _scaled(spec, 1.0, k); c, h = city_hwy(s)
        return s, dict(k=k, eff=1.0, city=c, hwy=h, ucity=ucity, uhwy=uhwy)
    rs = []
    for k in ks:
        c, h = city_hwy(_scaled(spec, 1.0, k)); rs.append(math.log(h / c))
    r = np.array(rs); o = np.argsort(r)
    k = float(np.exp(np.interp(tgt, r[o], np.log(np.array(ks))[o])))
    c, h = city_hwy(_scaled(spec, 1.0, k))
    eff = float(math.sqrt(ucity * uhwy / (c * h)))
    s = _scaled(spec, eff, k)
    c2, h2 = city_hwy(s)
    eff2 = eff * float(math.sqrt(ucity * uhwy / (c2 * h2)))       # one more level step (the engine knob is not linear)
    s = _scaled(spec, eff2, k)
    c3, h3 = city_hwy(s)
    return s, dict(k=k, eff=eff2, city=c3, hwy=h3, ucity=ucity, uhwy=uhwy)


def build_twin(job):
    vid, opt, match = job
    _opts.set(opt)
    spec, lab = ved_spec(vid)
    info = dict(vehicle_id=vid, architecture=spec['architecture'], cls=spec['class'], mass_kg=spec.get('mass_kg'),
                payload_kg=spec.get('payload_kg'), labels={k: lab.get(k) for k in ('unadj_city_mpg', 'unadj_hwy_mpg', 'match_confidence', 'example_models')})
    uc, uh = lab.get('unadj_city_mpg'), lab.get('unadj_hwy_mpg')
    if match and uc and uh and np.isfinite(uc) and np.isfinite(uh):
        try:
            if spec['architecture'].startswith('phev'):
                spec, m = label_match_phev(spec, float(uc), float(uh))
            else:
                spec, m = label_match(spec, float(uc), float(uh))
            info['label_match'] = m
        except Exception as e:
            info['label_match_error'] = f'{type(e).__name__}: {e}'[:200]
    return vid, spec, info


# ------------------------------------------------------------------ trips
def run_trip(job):
    import roadsim
    from roadsim import obd
    meta, arr, spec, opt = job['meta'], job['arr'], job['spec'], job['opt']
    arch = spec['architecture']
    trip = {k: arr[k].astype(float) for k in ('t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air')}
    trip['surface'] = arr['surface'].astype(int)
    T0 = float(np.nanmean(arr['T_amb']))
    init = dict(WARM, battery=max(20.0, T0))
    soc0 = None
    if arch != 'conv':
        s = arr['soc'][np.isfinite(arr['soc'])]
        if len(s) and 0.02 < float(s[0]) <= 1.0:
            soc0 = float(s[0])
    try:
        o = roadsim.simulate(spec, trip, init_soc=soc0, init_temps=init, options=opt, check=False)
    except Exception as e:
        return dict(trip_id=meta['trip_id'], vehicle_id=meta['vehicle_id'], error=f'{type(e).__name__}: {e}'[:200])
    ok = arr['valid'] == 1
    v = arr['v'].astype(float)
    fs = meta.get('fuel_source') or ''
    r = dict(trip_id=meta['trip_id'], vehicle_id=meta['vehicle_id'], arch=arch, T=T0, km=float(v[ok].sum()) / 1000,
             dur=int(ok.sum()), v_mean=float(v.mean()), idle_frac=float((v < 0.5).mean()), month=str(meta.get('date', ''))[:7],
             wind=float(np.nanmean(arr['wind_speed'])), wet=meta.get('wet_frac', 0), snow=meta.get('snow_frac', 0),
             fuel_source=fs, trace_rms=float(o['summary']['trace_rms_mps']), closure=float(o['ledger']['closure_rel']))
    if spec.get('engine'):
        fl = obd.label_fuel(o, spec, fs)
        fr = np.nan_to_num(np.asarray(o['fuel_rate'], float))
        r['fuel_sim_g'] = float(fl[ok].sum()); r['fuel_raw_g'] = float(fr[ok].sum())
        r['fuel_real_g'] = float(np.nansum(arr['fuel_rate'][ok]))
        er = arr['engine_rpm'].astype(float)
        m_rpm = ok & np.isfinite(er)
        if m_rpm.sum() > 30:
            r['eng_on_real'] = float((er[m_rpm] > 300).mean())
            r['eng_on_sim'] = float((np.asarray(o['eng_on'], float)[m_rpm] > 0.5).mean())
            idr = m_rpm & (v < 0.1) & (er > 300)
            ids = ok & (v < 0.1) & (np.asarray(o['eng_on'], float) > 0.5)
            if idr.sum() >= 10:
                r['idle_fuel_real'] = float(np.nanmean(arr['fuel_rate'][idr]))
            if ids.sum() >= 10:
                r['idle_fuel_sim'] = float(fl[ids].mean())
    if spec.get('battery'):
        pb = np.nan_to_num(np.asarray(o['batt_current'], float) * np.asarray(o['batt_voltage'], float))
        pr = arr['batt_current'].astype(float) * arr['batt_voltage'].astype(float)
        if np.isfinite(pr[ok]).mean() > 0.5:
            r['batt_sim_kWh'] = float(pb[ok].sum() / 3.6e6)
            r['batt_real_kWh'] = float(np.nansum(pr[ok]) / 3.6e6)
    r['hvac_sim_kWh'] = float(np.sum(np.asarray(o['hvac_power'], float)[ok]) / 3.6e6)
    a = arr.get('aux_power')
    if a is not None and np.isfinite(a[ok]).mean() > 0.5:
        r['aux_real_kWh'] = float(np.nansum(a[ok]) / 3.6e6)
    g = 'ICE' if arch == 'conv' else 'EV' if arch == 'bev' else 'PHEV' if arch.startswith('phev') else 'HEV'
    pct = lambda s, m: 100 * (s / m - 1) if (s is not None and m is not None and abs(m) > 1e-9) else None
    if g in ('ICE', 'HEV') and r.get('fuel_real_g', 0) > 2:
        r['err_fuel'] = pct(r['fuel_sim_g'], r['fuel_real_g']); r['err_fuel_raw'] = pct(r['fuel_raw_g'], r['fuel_real_g'])
        r['err_primary'] = r['err_fuel']
    if g in ('EV', 'PHEV') and r.get('batt_real_kWh') is not None and abs(r['batt_real_kWh']) > 0.05:
        r['err_batt'] = pct(r['batt_sim_kWh'], r['batt_real_kWh'])
        if g == 'EV':
            r['err_primary'] = r['err_batt']
    if g == 'PHEV':
        Er = r.get('fuel_real_g', 0) * LHV + (r.get('batt_real_kWh') or 0) * 3.6
        Es = r.get('fuel_sim_g', 0) * LHV + (r.get('batt_sim_kWh') or 0) * 3.6
        r['energy_real_MJ'] = Er; r['energy_sim_MJ'] = Es
        if Er > 0.2:
            r['err_primary'] = pct(Es, Er)
        if r.get('fuel_real_g', 0) > 2:
            r['err_fuel'] = pct(r['fuel_sim_g'], r['fuel_real_g'])
    if 'eng_on_real' in r:
        r['eng_on_pts'] = 100 * (r['eng_on_sim'] - r['eng_on_real'])
    if 'idle_fuel_real' in r and 'idle_fuel_sim' in r:
        r['err_idle_fuel'] = pct(r['idle_fuel_sim'], r['idle_fuel_real'])
    return r


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    return dict(n=int(len(x)), median=round(float(np.median(x)), 2), p25=round(float(np.percentile(x, 25)), 2),
                p75=round(float(np.percentile(x, 75)), 2), iqr=round(float(np.subtract(*np.percentile(x, [75, 25]))), 2),
                mean_abs=round(float(np.mean(np.abs(x))), 2))


def summarize(rows):
    d = pd.DataFrame(rows)
    out = {'by_car': {}, 'by_arch': {}}
    keys = ['err_primary', 'err_fuel', 'err_fuel_raw', 'err_batt', 'eng_on_pts', 'err_idle_fuel']
    for vid, g in d.groupby('vehicle_id'):
        c = dict(arch=g.arch.iloc[0], n_trips=len(g), km=round(g.km.sum(), 1))
        for k in keys:
            if k in g and g[k].notna().any():
                c[k] = stats(g[k])
        for a, b in (('fuel_sim_g', 'fuel_real_g'), ('fuel_raw_g', 'fuel_real_g'), ('batt_sim_kWh', 'batt_real_kWh'),
                     ('energy_sim_MJ', 'energy_real_MJ'), ('idle_fuel_sim', 'idle_fuel_real'), ('eng_on_sim', 'eng_on_real')):
            if a in g and b in g:
                m = g[a].notna() & g[b].notna()
                if m.any() and abs(g.loc[m, b].sum()) > 1e-9:
                    c[f'ratio_{a}'] = round(float(g.loc[m, a].sum() / g.loc[m, b].sum()), 4)
        out['by_car'][vid] = c
    for a, g in d.groupby('arch'):
        o = dict(n_trips=len(g), n_cars=int(g.vehicle_id.nunique()))
        for k in keys:
            if k in g and g[k].notna().any():
                o[k] = stats(g[k])
                o[k + '_car_medians'] = stats([v[k]['median'] for v in out['by_car'].values() if v['arch'] == a and v.get(k)])
        # temperature bands (primary error): the on-road extra loads depend on them
        if 'err_primary' in g:
            o['err_primary_by_T'] = {b: stats(g.loc[(g['T'] >= lo) & (g['T'] < hi), 'err_primary'])
                                     for b, lo, hi in (('<0C', -40, 0), ('0-10C', 0, 10), ('10-20C', 10, 20), ('20-27C', 20, 27), ('>=27C', 27, 60))}
        out['by_arch'][a] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--vehicles', default=None)
    ap.add_argument('--no-match', action='store_true', help='generic spec, no label matching')
    ap.add_argument('--arch', default=None, help='only these architectures (comma list)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--twins', default=None, help='reuse label-matched twins from this ved_twin json')
    _opts.init()
    a = ap.parse_args()
    import realdata as rd
    t0 = time.time()
    opt = _opts.get()
    vv = rd.ved_vehicles()
    dev = sorted(vv.index[vv.split == 'dev'])
    if a.vehicles:
        dev = [v for v in dev if v in a.vehicles.split(',')]
    if a.arch:
        dev = [v for v in dev if vv.loc[v, 'architecture'] in a.arch.split(',')]
    if a.twins:
        T = json.load(open(a.twins))
        twins = {vid: (T['twins'][vid]['spec'], T['twins'][vid]['info']) for vid in dev}
    else:
        with Pool(min(a.workers, len(dev))) as pool:
            res = pool.map(build_twin, [(vid, opt, not a.no_match) for vid in dev], chunksize=1)
        twins = {vid: (s, i) for vid, s, i in res}
    print(f'{len(twins)} twins ({time.time() - t0:.0f} s)', flush=True)
    jobs = []
    for meta, arr in rd.load_trips('dev', label_ok=True, vehicles=dev):
        jobs.append(dict(meta=meta, arr=arr, spec=twins[meta['vehicle_id']][0], opt=opt))
    with Pool(a.workers) as pool:
        rows = pool.map(run_trip, jobs, chunksize=4)
    errors = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    summ = summarize(rows)
    out = dict(check='VED dev twins on their own logged trips (OBD label model applied)', method=__doc__.strip(),
               options=opt, options_name=_opts.name(), label_matched=not a.no_match, n_trips=len(rows), n_errors=len(errors),
               errors=errors[:20], summary=summ, twins={vid: dict(spec=s, info=i) for vid, (s, i) in twins.items()},
               trips=rows, wall_s=round(time.time() - t0, 1))
    fn = a.out or os.path.join(HERE, _opts.out('ved_twin.json'))
    json.dump(out, open(fn, 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    for arch, o in summ['by_arch'].items():
        p = o.get('err_primary') or {}; cm = o.get('err_primary_car_medians') or {}
        eo = o.get('eng_on_pts') or {}; idl = o.get('err_idle_fuel') or {}
        print(f"{arch:16s} trips {o['n_trips']:4d} cars {o['n_cars']}  primary median {p.get('median')} IQR [{p.get('p25')}, {p.get('p75')}]"
              f"  car medians {cm.get('median')} [{cm.get('p25')}, {cm.get('p75')}]  eng-on pts {eo.get('median')}  idle fuel {idl.get('median')}")
    for vid, c in summ['by_car'].items():
        p = c.get('err_primary') or {}
        print(f"  {vid:8s} {c['arch']:16s} n {c['n_trips']:3d}  median {p.get('median')}  ratio fuel {c.get('ratio_fuel_sim_g')} raw {c.get('ratio_fuel_raw_g')} batt {c.get('ratio_batt_sim_kWh')}"
              f" idle {c.get('ratio_idle_fuel_sim')} eng-on {c.get('ratio_eng_on_sim')}")
    print(f'{len(rows)} trips, {len(errors)} errors, {out["wall_s"]} s -> {fn}')


if __name__ == '__main__':
    main()
