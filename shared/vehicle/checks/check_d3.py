"""Check: RoadSim twins of the Argonne D3 vehicles on their own dynamometer tests. Writes d3_twin.json.

Each test is driven by its twin (data/proc/d3/specs.json) on the test's own measured 1 Hz dyno speed, cell
temperature, solar lamps and start condition (initial coolant, oil, cabin and battery temperatures when logged),
with the dyno's road load (target A/B/C at the reference air density, no tyre temperature effects). Only tests that
are standard schedules (UDDS, HWFET, US06, SC03, LA92, NYCC, NEDC, WLTC; >= 60 % of the driving matched (coast-downs after a HWFET are driven too), zero
grade) are simulated; steady states, WOTs, grades and coast-downs are not.

Compared per test (simulated vs measured):
  fuel_g          measured fuel flow (or the emissions modal fuel when that is all the file has)
  fuel_corr_g     hybrids: fuel + net battery energy / (0.33 x 0.90) / LHV on both sides (SAE J1711-style), the
                  headline for HEVs because the twin's SOC path differs from the car's
  batt_Wh         net DC battery energy (+ = discharge): the headline for BEVs
  energy_kWh      PHEVs: fuel chemical energy (LHV) + net DC battery energy: the PHEV headline
  engine-on share and engine starts (engine speed > 400 rpm, stops shorter than 1 s ignored)
  hvac_Wh         cold (< 5 C) and hot (>= 30 C) tests: simulated HVAC energy vs the logged heater / A/C power
                  where the file has it (electric heaters and compressors only)
Medians and IQRs per architecture, temperature class and cycle are reported separately for dev and exam vehicles
(data/proc/d3/splits.json). The calibration pass may look only at 'dev'; 'exam' is computed for the final report.

Procedure details (14 Nov 2026, v2 calibration; same for every physics version):
  * cold starts at cold ambient (< 5 C) begin with soaked tyres (tire_warm 0), as the 20 F FTP in epa_procedures (at
    test-cell temperature the dyno coefficients and v1's fitted cold-start friction assume warm tyres); the oil starts
    at the logged dipstick temperature where logged;
  * key-off soaks between segments cool every thermal state (epa_procedures.soak), then logged temperatures win;
  * multi-setting tests (SC03 x3-4 with "AUTO-AUTO-ECON-OFF" / "AUTO 72, AUTO 72, MAX, OFF" in the notes) run each
    segment with its own HVAC setting (OFF = HVAC disabled); per-segment fuel is reported in 'segments'.

usage: python check_d3.py [--options roadsim-physics-v1] [--workers 14] [--split dev|exam|all] [--limit N]
  Output: d3_twin.json for v1 and --split all; otherwise d3_twin__<preset>[__<split>].json unless --out is given.
"""
import os, sys, json, time, argparse, math, re
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(VEH, 'roadsim'))
PROC = os.path.join(VEH, 'data', 'proc', 'd3')
sys.path.insert(0, HERE)
STANDARD = {'UDDS', 'HWFET', 'US06', 'SC03', 'LA92', 'NYCC', 'NEDC', 'WLTC', 'JP10_15'}
ETA_BATT_EQ = 0.33 * 0.90
TWIN_CATS = {'conventional', 'hybrid', 'plugin', 'electric'}
_SPECS = None


def _specs():
    global _SPECS
    if _SPECS is None:
        S = json.load(open(os.path.join(PROC, 'specs.json')))
        _SPECS = {k: {kk: vv for kk, vv in v.items() if kk not in ('provenance', 'epa_match', 'category', 'missing', 'name',
                                                                  'epa_match_note', 'd3_settings')} for k, v in S.items()}
    return _SPECS


def eligible(r):
    if r['category'] not in TWIN_CATS or not r['ok']:
        return False, 'category/ok'
    cyc = str(r.get('cycle') or '')
    if not cyc or not all(c in STANDARD for c in cyc.split('+')):
        return False, 'not a standard schedule'
    if (r.get('cycle_coverage') or 0) < 0.6:
        return False, 'schedule coverage < 60 %'
    if (r.get('grade_max_pct') or 0) > 0.1:
        return False, 'grade'
    if (r.get('distance_km') or 0) < 1:
        return False, 'distance < 1 km'
    return True, ''


def _starts(on):
    """engine starts in a 1 Hz on/off trace (rising edges, plus one if on at t = 0)."""
    x = (np.asarray(on) > 0.5).astype(int)
    return int(np.sum(np.diff(x) == 1) + (1 if len(x) and x[0] else 0))


def hvac_sequence(r):
    """per-phase HVAC settings from the test notes ('AUTO-AUTO-ECON-OFF', 'AUTO 72, AUTO 72, MAX, OFF'), or None."""
    txt = ' '.join(str(r.get(k) or '') for k in ('notes_text', 'cycle_text')).upper()
    m = re.search(r'SEQUENCE\s+((?:AUTO|ECON|OFF|MAX)(?:\s*-\s*(?:AUTO|ECON|OFF|MAX))+)', txt)
    if m:
        return [x.strip() for x in m.group(1).split('-')]
    m = re.search(r'HVAC;\s*((?:(?:AUTO(?:\s*72)?|ECON|OFF|MAX)\s*,\s*)+(?:AUTO(?:\s*72)?|ECON|OFF|MAX))', txt)
    if m:
        return [x.strip().split()[0] for x in m.group(1).split(',')]
    return None


def run_one(job):
    import roadsim
    from roadsim.constants import FUELS
    from roadsim.sim import get_vehicle
    r, options = job['row'], job['options']
    spec = json.loads(json.dumps(_specs()[r['vehicle_id']]))          # deep copy: the spec is edited per test
    t0 = time.time()
    z = np.load(os.path.join(PROC, r['path']))
    A = {k: z[k].astype(float) for k in z.files if not k.endswith('_10hz')}
    n = len(A['v'])
    Tamb = A['T_amb']
    Ta = float(np.nanmedian(Tamb))
    cls = r.get('T_class')
    hv = r.get('hvac_text')
    hvac_on = (hv == 'on') or (hv != 'off' and cls in ('cold', 'hot'))
    spec.setdefault('aux', {}).setdefault('HVAC', {})['enabled'] = bool(hvac_on)
    spec['aux']['HVAC'] = dict(spec['aux']['HVAC'], T_heat_set=22.2, T_cool_set=22.2)      # auto 72 F
    veh = get_vehicle(spec)
    trip = dict(t=np.arange(n, dtype=float), v=A['v'], grade=np.zeros(n), elev=np.zeros(n), T_amb=Tamb,
                wind_speed=np.zeros(n), wind_dir=np.zeros(n), heading=np.zeros(n), rho_air=A['rho_air'],
                surface=np.zeros(n, int), solar=A['solar'])
    cold = r.get('start') == 'cold'
    def pick(k, dflt):
        x = r.get(k)
        return float(x) if x is not None and np.isfinite(x) else dflt
    Teng0 = pick('T_coolant0_C', None)
    if Teng0 is None or (cold and Teng0 > Ta + 15):
        Teng0 = Ta if cold else 88.0
    init = dict(engine=Teng0, cabin=pick('T_cabin0_C', Ta), battery=pick('T_batt0_C', Ta),
                driveline=pick('T_trans0_C', Ta if cold else 80.0), tire_warm=0.0 if (cold and Ta < 5.0) else 1.0)
    To0 = pick('T_oil0_C', None)
    if To0 is not None and (not cold or To0 < Ta + 15):
        init['oil'] = To0
    arch = veh.arch
    init_soc, soc_src = None, 'default'
    soc_m = A.get('soc')
    if soc_m is not None and np.isfinite(soc_m[:30]).any():
        s0 = float(np.nanmedian(soc_m[:30][np.isfinite(soc_m[:30])][:10]))
        if 0.02 < s0 <= 1.0:
            init_soc, soc_src = s0, 'measured'
    if init_soc is None and arch.startswith('phev'):
        cap_Wh = spec['battery']['capacity_kWh'] * 1000
        bw = r.get('batt_Wh') or 0.0
        eon = r.get('engine_on_share')
        cd = (eon is not None and eon < 0.05) or bw > 0.10 * cap_Wh
        init_soc = 0.9 if cd else veh.battery.soc_lo + 0.05
        soc_src = 'inferred: charge-depleting (full)' if cd else 'inferred: charge-sustaining (CS level)'
    if init_soc is None and arch == 'bev':
        init_soc = 0.9
    # key-off soaks: >= 300 s at standstill (with the engine stopped where engine speed is logged) are not driven;
    # the twin is run segment by segment, the soak counts as engine off / zero energy, temperatures restart from the
    # logged values at the segment start (else Newton cooling of the engine and driveline over the soak)
    still = A['v'] < 0.05
    er = A.get('engine_rpm')
    soaks = []
    i = 0
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]: j += 1
            if j - i >= 300 and i > 0:
                if er is None or not np.isfinite(er[i:j]).any() or np.nanmedian(er[i:j]) < 100:
                    soaks.append((i, j))
            i = j
        else:
            i += 1
    segs, a = [], 0
    for (i, j) in soaks:
        segs.append((a, i)); a = j
    segs.append((a, n))
    segs = [(a_, b_) for a_, b_ in segs if b_ - a_ > 10]
    seq = hvac_sequence(r) if hvac_on else None
    try:
        _meta = json.load(open(os.path.join(PROC, r['path'])[:-4] + '.json'))
    except Exception:
        _meta = {}
    phases = [p for p in (_meta.get('phases') or []) if p.get('cycle') in STANDARD]
    if seq and len(phases) == len(seq):
        # split segments between phases (state carried on, no soak) so each phase runs with its own HVAC setting
        cuts = [(phases[i]['end'] + phases[i + 1]['start']) // 2 for i in range(len(phases) - 1)]
        for c in cuts:
            segs = [x for (a_, b_) in segs for x in (((a_, c), (c, b_)) if a_ + 10 < c < b_ - 10 else ((a_, b_),))]
    # unmetered idle gaps: >= 60 s at standstill with the engine idling (> 400 rpm) but the fuel record exactly 0 (bag
    # sampling off between phases of multi-cycle tests). The twin idles through them with the HVAC off (the logged cabin
    # heats back up) and their fuel is excluded from the simulated total, as it is from the measured one.
    fm_ = A.get('fuel_rate')
    gaps = []
    if er is not None and fm_ is not None and np.isfinite(fm_).any():
        q = still & (np.nan_to_num(fm_, nan=0.0) == 0.0) & (np.nan_to_num(er) > 400)
        i = 0
        while i < n:
            if q[i]:
                j = i
                while j < n and q[j]: j += 1
                if j - i >= 60:
                    gaps.append((i, j))
                i = j
            else:
                i += 1
    for (gi, gj) in gaps:
        segs = [x for (a_, b_) in segs for x in
                ([y for y in ((a_, gi), (gi, gj), (gj, b_)) if y[1] - y[0] > 0] if a_ < gi and gj < b_ else
                 [y for y in ((a_, gj), (gj, b_)) if y[1] - y[0] > 0] if a_ >= gi and a_ < gj < b_ else
                 [y for y in ((a_, gi), (gi, b_)) if y[1] - y[0] > 0] if a_ < gi < b_ <= gj else [(a_, b_)])]
    gap_mask = np.zeros(n, bool)
    for (gi, gj) in gaps:
        gap_mask[gi:gj] = True
    seg_hvac = [True] * len(segs)
    if seq and len(phases) == len(seq):
        for si, (a_, b_) in enumerate(segs):
            ov = [max(0, min(b_, p['end']) - max(a_, p['start'])) for p in phases]
            if max(ov) > 0:
                seg_hvac[si] = seq[int(np.argmax(ov))]
    for si, (a_, b_) in enumerate(segs):
        if gap_mask[a_:b_].all():
            seg_hvac[si] = 'OFF'
    seg_rows = []
    K = ('fuel_rate', 'batt_current', 'batt_voltage', 'eng_on', 'hvac_power', 'soc')
    full = {k: np.zeros(n) for k in K}
    st, soc, closure, trace, dist = dict(init), init_soc, 0.0, 0.0, 0.0
    prev_end = None
    from epa_procedures import soak as _soak
    try:
        for si, (a_, b_) in enumerate(segs):
            if prev_end is not None and a_ > prev_end:
                dtk = a_ - prev_end
                st = _soak(spec, st, Ta, dtk, options=options)
                for key, sig in (('engine', 'T_coolant'), ('cabin', 'T_cabin'), ('driveline', 'T_trans_oil'), ('oil', 'T_oil')):
                    x = A.get(sig)
                    if x is not None and np.isfinite(x[a_]) and x[a_] != 0:
                        st[key] = float(x[a_])
                st['eng_on'] = False
            tr = {k: (v[a_:b_] if hasattr(v, '__len__') and len(v) == n else v) for k, v in trip.items()}
            tr['t'] = np.arange(b_ - a_, dtype=float)
            sh = seg_hvac[si]
            sp_seg = spec
            if sh == 'OFF':
                sp_seg = json.loads(json.dumps(spec)); sp_seg['aux']['HVAC']['enabled'] = False
            opt_seg = options
            if hvac_on and cls == 'hot':          # A/C auto 72 F / max: recirculated air (SC03 practice)
                opt_seg = (dict(options, hvac_recirc_cool=1.0) if isinstance(options, dict) else
                           {'hvac_recirc_cool': 1.0, **({'preset': options} if isinstance(options, str) else {})})
            elif hvac_on and cls == 'cold':       # heat auto 72 F, outside air (1066.710(c))
                opt_seg = (dict(options, hvac_recirc_heat=0.0) if isinstance(options, dict) else
                           {'hvac_recirc_heat': 0.0, **({'preset': options} if isinstance(options, str) else {})})
            if isinstance(opt_seg, dict):
                opt_seg['hvac_T_set'] = 22.2
            o = roadsim.simulate(sp_seg, tr, init_soc=soc, init_temps=st, options=opt_seg, check=False)
            fm = A.get('fuel_rate')
            seg_rows.append(dict(start=a_, end=b_, hvac=sh, sim_fuel_g=float(np.sum(o['fuel_rate'])),
                                 meas_fuel_g=float(np.nansum(fm[a_:b_])) if fm is not None and np.isfinite(fm[a_:b_]).any() else None,
                                 sim_hvac_Wh=float(np.sum(o['hvac_power']) / 3600),
                                 sim_cabin_end=float(o['final_state']['cabin']),
                                 meas_cabin_end=float(A['T_cabin'][b_ - 1]) if A.get('T_cabin') is not None and np.isfinite(A['T_cabin'][b_ - 1]) else None,
                                 distance_km=float(o['summary']['distance_m']) / 1000))
            for k in K:
                full[k][a_:b_] = o[k]
            full['fuel_rate'][a_:b_][gap_mask[a_:b_]] = 0.0
            st = dict(o['final_state']); soc = st.get('soc', soc)
            closure = max(closure, float(o['ledger']['closure_rel'])); trace = max(trace, float(o['summary']['trace_rms_mps']))
            dist += float(o['summary']['distance_m']); prev_end = b_
    except Exception as e:
        return dict(test_id=r['test_id'], error=f'{type(e).__name__}: {e}'[:300])
    out = full
    fuel = FUELS[veh.fuel] if veh.engine is not None else None
    lhv = fuel['lhv'] if fuel else 43e3                         # J/g
    sim = dict(fuel_g=float(np.sum(out['fuel_rate'])),
               batt_Wh=float(np.sum(out['batt_current'] * out['batt_voltage']) / 3600) if veh.battery is not None else None,
               eng_on_share=float(np.mean(out['eng_on'])) if veh.engine is not None else 0.0,
               eng_starts=_starts(out['eng_on']) if veh.engine is not None else 0,
               hvac_Wh=float(np.sum(out['hvac_power']) / 3600), distance_km=dist / 1000,
               trace_rms_mps=trace, ledger_closure=closure, n_segments=len(segs), soak_s=int(sum(j - i for i, j in soaks)),
               soc_end=float(soc) if veh.battery is not None else None)
    meas = dict(fuel_g=r.get('fuel_g'), batt_Wh=r.get('batt_Wh'), eng_on_share=r.get('engine_on_share'),
                eng_starts=r.get('engine_starts'), heater_Wh=r.get('heater_Wh'), ac_Wh=r.get('ac_Wh'),
                distance_km=r.get('distance_km'))
    pct = lambda s, m: 100 * (s / m - 1) if (s is not None and m is not None and np.isfinite(m) and abs(m) > 1e-9) else None
    res = dict(test_id=r['test_id'], vehicle=r['vehicle'], vehicle_id=r['vehicle_id'], split=job['split'], architecture=arch,
               category=r['category'], cycle=r['cycle'], cycle_seq=r.get('cycle_seq'), T_amb_C=Ta, T_class=cls,
               start=r.get('start'), solar=bool(r.get('solar_lamps')), hvac_on=bool(hvac_on), init_soc=init_soc,
               init_soc_source=soc_src, init_temps=init, fuel_source=r.get('fuel_source'), sim=sim, meas=meas,
               segments=seg_rows, hvac_sequence=seq, unmetered_gaps_s=int(gap_mask.sum()),
               wall_s=round(time.time() - t0, 2))
    err = {}
    if arch == 'conv':
        if meas['fuel_g'] and meas['fuel_g'] > 20:
            err['fuel_pct'] = pct(sim['fuel_g'], meas['fuel_g'])
        err['primary_pct'] = err.get('fuel_pct')
    elif arch.startswith('hev'):
        if meas['fuel_g'] and meas['fuel_g'] > 20:
            err['fuel_pct'] = pct(sim['fuel_g'], meas['fuel_g'])
            mb = meas['batt_Wh'] if meas['batt_Wh'] is not None and np.isfinite(meas['batt_Wh']) else None
            if mb is not None:
                fc_m = meas['fuel_g'] + mb * 3600 / ETA_BATT_EQ / lhv
                fc_s = sim['fuel_g'] + sim['batt_Wh'] * 3600 / ETA_BATT_EQ / lhv
                err['fuel_corr_pct'] = pct(fc_s, fc_m)
                res['meas']['fuel_corr_g'] = fc_m; res['sim']['fuel_corr_g'] = fc_s
        err['primary_pct'] = err.get('fuel_corr_pct', err.get('fuel_pct'))
        res['primary_metric'] = 'fuel_corr' if 'fuel_corr_pct' in err else 'fuel (no battery channel)'
    elif arch.startswith('phev'):
        mf = meas['fuel_g'] or 0.0
        mb = meas['batt_Wh'] if meas['batt_Wh'] is not None and np.isfinite(meas['batt_Wh']) else None
        if mf > 20: err['fuel_pct'] = pct(sim['fuel_g'], mf)
        if mb is not None and abs(mb) > 50: err['batt_pct'] = pct(sim['batt_Wh'], mb)
        if mb is not None:
            em = mf * lhv / 3.6e6 + mb / 1000; es = sim['fuel_g'] * lhv / 3.6e6 + sim['batt_Wh'] / 1000
            err['energy_pct'] = pct(es, em); res['meas']['energy_kWh'] = em; res['sim']['energy_kWh'] = es
        err['primary_pct'] = err.get('energy_pct')
    elif arch == 'bev':
        if meas['batt_Wh'] is not None and meas['batt_Wh'] > 50:
            err['batt_pct'] = pct(sim['batt_Wh'], meas['batt_Wh'])
        err['primary_pct'] = err.get('batt_pct')
    if veh.engine is not None and meas['eng_on_share'] is not None:
        err['eng_on_share_pts'] = 100 * (sim['eng_on_share'] - meas['eng_on_share'])
        if meas['eng_starts'] is not None:
            err['eng_starts_diff'] = sim['eng_starts'] - meas['eng_starts']
    parts = [x for x in (meas['heater_Wh'], meas['ac_Wh'], r.get('batt_heater_Wh')) if x is not None]
    mh = sum(parts) if parts else None          # logged electric heater + A/C compressor + battery heater
    if cls in ('cold', 'hot') and mh is not None and mh > 20:
        err['hvac_pct'] = pct(sim['hvac_Wh'], mh)
        res['meas']['hvac_logged_Wh'] = mh
    res['err'] = err
    return res


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    return dict(n=int(len(x)), median=round(float(np.median(x)), 2), p25=round(float(np.percentile(x, 25)), 2),
                p75=round(float(np.percentile(x, 75)), 2), iqr=round(float(np.percentile(x, 75) - np.percentile(x, 25)), 2),
                mean_abs=round(float(np.mean(np.abs(x))), 2), min=round(float(x.min()), 2), max=round(float(x.max()), 2))


def summarize(rows):
    out = {}
    def grp(name, f):
        rr = [r for r in rows if f(r)]
        if not rr:
            return None
        d = dict(n_tests=len(rr), n_vehicles=len({r['vehicle'] for r in rr}))
        for k in ('primary_pct', 'fuel_pct', 'fuel_corr_pct', 'batt_pct', 'energy_pct', 'eng_on_share_pts', 'eng_starts_diff', 'hvac_pct'):
            s = stats([r['err'].get(k) for r in rr])
            if s: d[k] = s
        eo_s = [r['sim']['eng_on_share'] for r in rr if r['meas'].get('eng_on_share') is not None and r['architecture'] != 'bev']
        eo_m = [r['meas']['eng_on_share'] for r in rr if r['meas'].get('eng_on_share') is not None and r['architecture'] != 'bev']
        if eo_m:
            d['eng_on_share_median'] = dict(sim=round(float(np.median(eo_s)), 3), meas=round(float(np.median(eo_m)), 3))
            st_s = [r['sim']['eng_starts'] for r in rr if r['meas'].get('eng_starts') is not None]
            st_m = [r['meas']['eng_starts'] for r in rr if r['meas'].get('eng_starts') is not None]
            d['eng_starts_median'] = dict(sim=float(np.median(st_s)), meas=float(np.median(st_m)))
        return d
    for split in ('dev', 'exam'):
        S = [r for r in rows if r['split'] == split]
        o = {}
        for a in sorted({r['architecture'] for r in S}):
            A = [r for r in S if r['architecture'] == a]
            o[a] = {'all': _g(A, grp)}
            for tc in ('std', 'cold', 'hot', 'mild'):
                g = _g([r for r in A if r['T_class'] == tc], grp)
                if g: o[a]['T_' + tc] = g
            for st in ('cold', 'hot'):
                g = _g([r for r in A if r['start'] == st and r['T_class'] == 'std'], grp)
                if g: o[a][st + '_start_std_T'] = g
            for cyc in sorted({r['cycle'] for r in A}):
                g = _g([r for r in A if r['cycle'] == cyc], grp)
                if g: o[a]['cycle_' + cyc] = g
            o[a]['vehicles'] = sorted({r['vehicle'] for r in A})
        o['hev_all'] = {'all': _g([r for r in S if r['architecture'].startswith('hev')], grp)}
        o['phev_all'] = {'all': _g([r for r in S if r['architecture'].startswith('phev')], grp)}
        out[split] = o
    return out


def _g(rr, grp):
    if not rr:
        return None
    return grp(None, lambda r, ids={id(x) for x in rr}: id(r) in ids)


def measurement_qa(idx):
    """Data check, no fitting: each conventional vehicle's measured hot HWFET fuel economy (median over tests at
    15-30 C) against the EPA test-car HWY value of the row its twin was matched to (data/proc/d3/specs.json)."""
    S = json.load(open(os.path.join(PROC, 'specs.json')))
    E = pd.read_parquet(os.path.join(VEH, 'data', 'proc', 'epa', 'testcars_vehicles.parquet'))
    x = idx[(idx.cycle == 'HWFET') & (idx.T_class == 'std') & (idx.start == 'hot') & (idx.category == 'conventional') & idx.fuel_g.notna()]
    out = {}
    for vid, g in x.groupby('vehicle_id'):
        em = (S.get(vid) or {}).get('epa_match')
        if not em or not np.isfinite(E.loc[em['epa_vehicle_id'], 'HWY_mpg']):
            continue
        dens = 0.85 if S[vid].get('engine', {}).get('fuel') == 'diesel' else 0.7433
        mpg = float((g.distance_km / 1.609344 / (g.fuel_g / (3785.41 * dens))).median())
        hw = float(E.loc[em['epa_vehicle_id'], 'HWY_mpg'])
        out[vid] = dict(meas_HWFET_mpg=round(mpg, 2), epa_HWY_mpg=hw, ratio=round(mpg / hw, 3), n=int(len(g)),
                        fuel_source=g.fuel_source.mode().iloc[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--options', default='roadsim-physics-v1', help='RoadSim physics preset name or a JSON dict of options')
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--split', default='all', choices=['dev', 'exam', 'all'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--vehicles', default=None, help='comma-separated vehicle ids or names (default: all)')
    a = ap.parse_args()
    options = json.loads(a.options) if a.options.strip().startswith('{') else a.options
    if options == 'roadsim-physics-v1':
        options = None
    if a.out is None:
        import _opts
        name = _opts.out('d3_twin.json', options)
        if a.split != 'all':
            name = name.replace('.json', f'__{a.split}.json')
        a.out = os.path.join(HERE, name)
    vsel = set(a.vehicles.split(',')) if a.vehicles else None
    import roadsim
    t0 = time.time()
    idx = pd.read_parquet(os.path.join(PROC, 'index.parquet'))
    sp = json.load(open(os.path.join(PROC, 'splits.json')))
    specs = _specs()
    jobs, skipped = [], {}
    for r in idx.to_dict('records'):
        r = {k: (None if (isinstance(v, float) and not np.isfinite(v)) or v is pd.NA else v) for k, v in r.items()}
        ok, why = eligible(r)
        if ok and r['vehicle_id'] not in specs:
            ok, why = False, 'no twin spec'
        if not ok:
            skipped[why] = skipped.get(why, 0) + 1; continue
        split = sp['vehicles'].get(r['vehicle_id'], {}).get('split')
        if split is None:
            skipped['not in split'] = skipped.get('not in split', 0) + 1; continue
        if a.split != 'all' and split != a.split:
            continue
        if vsel and r['vehicle_id'] not in vsel and r['vehicle'] not in vsel:
            continue
        jobs.append(dict(row=r, options=options, split=split))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f'{len(jobs)} tests to simulate; skipped {skipped}', flush=True)
    with Pool(a.workers) as pool:
        rows = pool.map(run_one, jobs, chunksize=1)
    errors = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    summ = summarize(rows)
    qa = measurement_qa(idx)
    out = dict(check='RoadSim twins of the Argonne D3 dynamometer tests', method=__doc__.strip(),
               measurement_qa=dict(note=measurement_qa.__doc__.strip(), vehicles=qa),
               options=options, physics=roadsim.PHYSICS_VERSION, roadsim_version=roadsim.__version__,
               splits_sha256=sp['sha256'], n_tests=len(rows), n_errors=len(errors), errors=errors, skipped=skipped,
               note='exam is computed for the final report only; the calibration pass looks only at dev',
               summary=summ, tests=rows, wall_s=round(time.time() - t0, 1),
               attribution='This data is from the Downloadable Dynamometer Database and was generated at the Advanced Mobility Technology Laboratory (AMTL) at Argonne National Laboratory under the funding and guidance of the U.S. Department of Energy (DOE).')
    json.dump(out, open(a.out, 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    for split in ('dev', 'exam'):
        print(f'==== {split}' + ('   (exam: final report only)' if split == 'exam' else ''))
        for arch, d in summ.get(split, {}).items():
            al = d.get('all') or {}
            p = al.get('primary_pct') or {}
            eo = al.get('eng_on_share_median') or {}
            print(f"  {arch:16s} tests {al.get('n_tests', 0):4d} veh {al.get('n_vehicles', 0):3d}  primary median {p.get('median')} "
                  f"IQR [{p.get('p25')}, {p.get('p75')}]  eng-on sim/meas {eo.get('sim')}/{eo.get('meas')}")
    print(f'{len(rows)} tests, {len(errors)} errors, {out["wall_s"]} s -> {a.out}')


if __name__ == '__main__':
    main()
