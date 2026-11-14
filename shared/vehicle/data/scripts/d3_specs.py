"""RoadSim vehicle specs ("twins") for the Argonne D3 vehicles -> data/proc/d3/specs.json.

Sources, in order of preference, and how each field is marked in spec['provenance']:
  reported: D3 test files   test weight and dyno target road-load A/B/C (median over the vehicle's tests)
  reported: EPA test-car    engine displacement and rated power, transmission type and gears, axle and N/V ratios
                            (data/proc/epa/testcars_vehicles.parquet, the row of that model year and model closest to
                            the D3 test weight and road load)
  reported: hand table      data/d3_facts.json: motors, battery, and engine/transmission where EPA has no row
                            (manufacturer figures entered by hand; anl.gov refuses scripts, so the D3 pages were not read)
  derived                   CdA/Crr from A/B/C (spec_from_abc, as checks/epa_vehicles.py), final drive from N/V and the
                            top gear, motor top speed from the final drive, engine map from fuel/size/turbo
  default                   everything else (RoadSim fill_spec class/architecture defaults: wheel radius, gear ratios,
                            maps, cabin, aux)
The twin runs on a chassis dynamometer: tyre temperature effects are off and air density is the reference 1.1767 kg/m3
used when CdA was derived (the dyno applies the target road load whatever the cell temperature).

usage: python d3_specs.py      (needs data/proc/d3/index.parquet from d3_index.py)
"""
import os, re, sys, json, math, copy, warnings
warnings.filterwarnings('ignore', 'This pattern is interpreted as a regular expression')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_VEH = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(_VEH, 'roadsim'))
sys.path.insert(0, HERE)
from roadsim.vehicle import GEAR_SETS, spec_from_abc, fill_spec
from roadsim.constants import LB, MPH
from d3_fetch import VEHICLES

PROC = os.path.join(_VEH, 'data', 'proc', 'd3')
FACTS = os.path.join(_VEH, 'data', 'd3_facts.json')
EPA = os.path.join(_VEH, 'data', 'proc', 'epa', 'testcars_vehicles.parquet')
LBF = 4.4482216
D3 = 'reported: D3 test files'
HAND = 'reported: hand table (data/d3_facts.json; manufacturer figure, not re-checked against the D3 page)'
DEF = 'default: RoadSim fill_spec'

# engine map overrides for hybrids whose engine is not an Atkinson-cycle unit
MAP_OVERRIDE = {'2019 Acura MDX Sport Hybrid': 'gas_v6_gm_43', '2010 Mercedes S400h BlueHybrid': 'gas_v6_gm_43',
                '2013 Chevrolet Malibu Eco': 'gas_na_toyota_a25a', '2013 Honda Civic Hybrid': 'gas_na_toyota_a25a',
                '2010 Honda CR-Z': 'gas_na_toyota_a25a', '2010 Honda Insight': 'gas_na_toyota_a25a'}


def _rl(A, B, C, mph):
    return A + B * mph + C * mph ** 2


def epa_match(name, f, w_lb, abc):
    """-> (row or None, note)."""
    if not f.get('epa'):
        return None, 'no EPA row (model not sold in the US or not in the list)'
    d = pd.read_parquet(EPA)
    y = int(name[:4])
    mk, md = f['epa']
    m = d.make.str.upper().str.contains(mk, regex=True, na=False) | d.mfr.str.upper().str.contains(mk, regex=True, na=False)
    m &= d.model.str.upper().str.contains(md, regex=True, na=False)
    archmap = {'conv': 'conv', 'bev': 'bev', 'hev_parallel_p2': 'hev', 'hev_series': 'hev', 'hev_powersplit': 'hev',
               'phev_powersplit': 'phev', 'phev_series': 'phev'}
    c = d[m]
    note = ''
    for dy in (0, 1, -1, 2, -2):
        cy = c[c.model_year == y + dy]
        if len(cy):
            c = cy
            if dy: note = f'nearest model year in the EPA list: {y + dy}'
            break
    else:
        return None, f'no EPA row for {mk}/{md}'
    arch_ok = True
    if f.get('arch') in archmap:
        ca = c[c.architecture == archmap[f['arch']]]
        if len(ca) or f['arch'] in ('conv', 'bev'):
            c = ca
        else:                                        # older hybrids are often labelled 'conv' in the EPA list
            arch_ok = False
    if not len(c):
        return None, f'no EPA row for {mk}/{md} with this architecture'
    if f.get('engine_L') and c.displacement_L.notna().any():
        cc = c[(c.displacement_L - f['engine_L']).abs() <= 0.15]
        if len(cc): c = cc
    if f.get('fuel') and 'fuel' in c:
        cc = c[c.fuel.astype(str).str.lower().str.contains('diesel') == (f['fuel'] == 'diesel')]
        if len(cc): c = cc
    if f.get('engine_kW') and f.get('arch') == 'conv' and c.rated_hp.notna().any():
        cc = c[(c.rated_hp * 0.7457 - f['engine_kW']).abs() <= 0.15 * f['engine_kW']]
        if len(cc): c = cc
    fam = {'AT': {'A', 'SA', 'L', 'S'}, 'DCT': {'AM', 'AMS', 'M'}, 'AMT': {'AM', 'AMS', 'M'}, 'CVT': {'AV', 'AVS', 'SCV', 'CVT'}}
    if f.get('trans') and f['trans'][0] in fam and f.get('arch') == 'conv':
        cc = c[c.trans_code.astype(str).isin(fam[f['trans'][0]])]
        if len(cc): c = cc
        if f['trans'][1] and f['trans'][0] != 'CVT':
            cc = c[c.gears == f['trans'][1]]
            if len(cc): c = cc
    if w_lb and abc and all(x is not None for x in abc):
        sc = (c.etw_lb - w_lb).abs() / w_lb
        for v in (20, 40, 60):
            ref = _rl(*abc, v)
            sc = sc + (_rl(c.A_lbf, c.B_lbf_mph, c.C_lbf_mph2, v) - ref).abs() / ref
        r = c.loc[sc.idxmin()]
        note = (note + '; ' if note else '') + f'closest of {len(c)} rows to the D3 test weight and road load (score {sc.min():.3f})'
    else:
        r = c.iloc[0]
        note = (note + '; ' if note else '') + f'first of {len(c)} rows (no D3 road load to match)'
    if not arch_ok:
        r = r.copy()
        r['axle_ratio'] = np.nan; r['nv_ratio'] = np.nan; r['rated_hp'] = np.nan
        note += '; no EPA row with this architecture: only the displacement is taken from the conventional sibling'
    return r, note


def trans_of(r):
    code = str(r.trans_code)
    n = int(r.gears) if pd.notna(r.gears) else None
    if code in ('AV', 'AVS', 'SCV', 'CVT'):
        return 'CVT', 1
    if code in ('AM', 'AMS', 'M'):
        return 'DCT', n
    return 'AT', n


def build_one(name, cat, f, d3):
    prov = {}
    arch = f['arch']; cls = f['class']
    w_lb = d3.get('test_weight_lb'); abc = d3.get('abc')
    r, note = epa_match(name, f, w_lb, abc)
    spec = {'id': 'D3_' + re.sub(r'[^\w.-]+', '_', name), 'name': name, 'class': cls, 'architecture': arch,
            'rot_mass_frac': 0.015, 'tire': {'K_t': 0.0, 'temp_effects': False}}
    prov['class'] = HAND; prov['architecture'] = HAND + ('; ' + f['arch_note'] if f.get('arch_note') else '')
    prov['rot_mass_frac'] = 'derived: as checks/epa_vehicles.py (dyno emulation)'
    prov['tire'] = 'derived: chassis dynamometer, road load fixed by the targets (no tyre temperature effects)'
    if w_lb:
        mass = w_lb * LB; prov['mass_kg'] = f'{D3}: test weight {w_lb:.0f} lb (median of {d3["n_w"]} tests)'
    elif r is not None:
        mass = float(r.etw_lb) * LB; prov['mass_kg'] = f'reported: EPA test-car {r.name}: ETW'
    else:
        mass = None; prov['mass_kg'] = DEF
    if mass: spec['mass_kg'] = mass
    rr = 0.33 if cls == 'car' else 0.36 if cls in ('suv', 'van') else 0.39
    spec['wheel_radius_m'] = rr; prov['wheel_radius_m'] = 'default: class value as checks/epa_vehicles.py'
    k_nv = 2 * math.pi * rr / (60 * MPH)
    nv = float(r.nv_ratio) if r is not None and pd.notna(r.nv_ratio) else None
    axle = float(r.axle_ratio) if r is not None and pd.notna(r.axle_ratio) else None
    EP = f'reported: EPA test-car {r.name}' if r is not None else None
    # ---- engine
    if arch != 'bev':
        disp = f.get('engine_L'); P = f.get('engine_kW')
        if r is not None and pd.notna(r.displacement_L):
            disp = float(r.displacement_L); prov['engine.displacement_L'] = EP
        else:
            prov['engine.displacement_L'] = HAND if disp else DEF
        if arch == 'conv' and r is not None and pd.notna(r.rated_hp):
            P = float(r.rated_hp) * 0.7457; prov['engine.max_power_kW'] = EP + ' (rated hp)'
        else:
            prov['engine.max_power_kW'] = HAND if P else DEF      # hybrids: EPA lists system power; use the engine's
        fuel = f.get('fuel', 'gasoline')
        turbo = f.get('turbo', False) or (fuel == 'gasoline' and P and disp and P / disp > 85)
        if name in MAP_OVERRIDE: m = MAP_OVERRIDE[name]
        elif fuel == 'diesel': m = 'diesel_gm_30'
        elif turbo: m = 'gas_turbo_honda_15' if (disp or 2) < 2.2 else 'gas_turbo_ford_27'
        elif arch != 'conv': m = 'gas_atk_toyota_tnga_hev_paper'
        elif (disp or 2) >= 4.5: m = 'gas_v8_ford_73'
        elif (disp or 2) >= 3.0: m = 'gas_v6_gm_43'
        else: m = 'gas_na_toyota_a25a'
        spec['engine'] = {k: v for k, v in dict(fuel=fuel, displacement_L=disp, max_power_kW=P, bsfc_map=m).items() if v is not None}
        if f.get('idle_stop'): spec['engine']['idle_stop'] = True; prov['engine.idle_stop'] = HAND
        prov['engine.fuel'] = HAND
        prov['engine.bsfc_map'] = 'derived: rule on fuel, size, turbo and architecture (as checks/epa_vehicles.py)'
    # ---- transmission and final drive
    if arch in ('conv', 'hev_parallel_p2'):
        t, n = f['trans'][0], f['trans'][1]
        if r is not None and arch == 'conv':
            t2, n2 = trans_of(r)
            if not (t == 'DCT' and t2 == 'AT'):      # keep the hand table's clutch gearbox for manuals/DCTs
                t = t2
            n = n2 or n
            prov['transmission'] = EP + ' (type, gears); ratios ' + DEF
        else:
            prov['transmission'] = HAND + ' (type, gears); ratios ' + DEF
        if arch == 'hev_parallel_p2' and f.get('tc'):
            t = 'AT'
        if t == 'CVT':
            top = 0.40
            fd = axle if (axle and axle > 2) else (nv * k_nv / top if nv else None)
            spec['transmission'] = dict(type='CVT', ratio_range=[2.6, (nv * k_nv / fd) if (nv and fd) else 0.40])
            if fd: spec['final_drive'] = fd; prov['final_drive'] = EP + ' (axle ratio)'
        else:
            n = n or 6
            gears = GEAR_SETS.get((t, n)) or GEAR_SETS.get(('AT', n)) or list(np.round(np.geomspace(4.6, 0.65, n), 3))
            spec['transmission'] = dict(type=t, gears=gears)
            if nv:
                spec['final_drive'] = nv * k_nv / gears[-1]; prov['final_drive'] = 'derived: EPA N/V ratio / top gear'
    elif arch in ('hev_powersplit', 'phev_powersplit'):
        spec['transmission'] = dict(type='eCVT', ring_teeth=78, sun_teeth=30, mg2_ratio=f.get('mg2_ratio', 3.0))
        prov['transmission'] = ('reported: hand table (MG2 reduction)' if f.get('mg2_ratio') else 'default: as checks/epa_vehicles.py (mg2_ratio 3.0)') + '; planetary 78/30 ' + DEF
        if axle: spec['final_drive'] = axle; prov['final_drive'] = EP + ' (axle ratio)'
    elif arch in ('hev_series', 'phev_series', 'bev'):
        spec['transmission'] = dict(type='single'); prov['transmission'] = HAND
    # ---- motors and battery
    if arch in ('hev_series', 'phev_series'):
        fd = nv * k_nv if (nv and nv > 30) else None
        # a series hybrid's EPA N/V is the engine's (lock-up / coupled mode), not the traction motor's: keep it only if
        # it puts the motor's 14,000 rpm between 140 and 260 km/h (2015 Accord Hybrid: 2.84 -> 614 km/h, the motor ran
        # torque-limited at a few hundred rpm and regenerated 0.46 of the braking energy vs 0.64 measured)
        if fd and not (140 <= 14000 * math.pi / 30 / fd * rr * 3.6 <= 260):
            fd = None
        mt = {m[0]: m[1] for m in f['motors']}
        spec['motors'] = [dict(role='traction', max_power_kW=mt['traction'], eff_map='ipm_generic_150', max_rpm=14000),
                          dict(role='generator', max_power_kW=mt['generator'], eff_map='ipm_generic_150', max_rpm=14000)]
        tq = {m[0]: m[2] for m in f['motors']}
        if tq.get('traction'): spec['motors'][0]['max_torque_Nm'] = tq['traction']
        spec['final_drive'] = fd or 14000 * math.pi / 30 * rr / (180 / 3.6)
        prov['final_drive'] = 'derived: EPA N/V ratio' if fd else 'derived: motor 14,000 rpm at 180 km/h (as checks/epa_vehicles.py)'
        prov['motors'] = HAND + '; maps ' + DEF
    elif arch in ('hev_powersplit', 'phev_powersplit'):
        mt = {m[0]: m for m in f['motors']}
        spec['motors'] = [dict(role='MG1', max_power_kW=mt['MG1'][1], eff_map='ipm_generic_150', max_rpm=13500),
                          dict(role='MG2', max_power_kW=mt['MG2'][1], eff_map='ipm_generic_150', max_rpm=17000 if 'Toyota' in name else 14000)]
        if mt['MG2'][2]: spec['motors'][1]['max_torque_Nm'] = mt['MG2'][2]
        prov['motors'] = HAND + '; maps and top speeds ' + DEF
    elif arch == 'hev_parallel_p2':
        k, P_, T_ = f['motors'][0]
        spec['motors'] = [dict(role='P2', max_power_kW=P_, max_torque_Nm=T_, eff_map='ipm_generic_150', max_rpm=7000)]
        prov['motors'] = HAND + '; map ' + DEF
    elif arch == 'bev':
        P_ = f.get('motor_kW')
        if r is not None and pd.notna(r.rated_hp) and (P_ is None):
            P_ = float(r.rated_hp) * 0.7457; prov['motors'] = EP + ' (rated hp as motor power)'
        else:
            prov['motors'] = HAND
        fd = nv * k_nv if (nv and nv > 30) else 9.0
        spec['final_drive'] = fd; prov['final_drive'] = 'derived: EPA N/V ratio' if (nv and nv > 30) else 'default: 9.0 (as checks/epa_vehicles.py)'
        mot = dict(role='traction', max_power_kW=P_, eff_map=f.get('motor_map', 'ipm_generic_150'), max_rpm=fd * (200 / 3.6) / rr * 30 / math.pi)
        if f.get('motor_Nm'): mot['max_torque_Nm'] = f['motor_Nm']
        spec['motors'] = [mot]
        prov['motors'] += '; top speed derived (200 km/h at the final drive)'
    if arch != 'conv':
        b = f.get('battery')
        if f.get('variants_by_epa_hp') and r is not None:
            hp = str(int(round(float(r.rated_hp))))
            b = f['variants_by_epa_hp'].get(hp, b)
        c, V, ch = b
        win = [0.02, 0.98] if arch == 'bev' else [0.15, 0.95] if arch.startswith('phev') else [0.4, 0.8] if arch == 'hev_powersplit' else [0.3, 0.8]
        spec['battery'] = dict(capacity_kWh=c, nominal_V=V, chemistry=ch if ch in ('NMC', 'LFP', 'NiMH', 'NCA') else 'NMC', soc_window=win)
        if spec['battery']['chemistry'] == 'NCA': spec['battery']['chemistry'] = 'NMC'
        prov['battery'] = HAND + '; SOC window ' + DEF
        if arch.startswith('phev'):
            # plug-in packs are sized to drive the car electrically: discharge power = traction machines / 0.9, at most
            # 9 C (RoadSim's 'energy' default of 3 C gave 23-55 kW; D3 dev plug-ins reach 67 (Energi), 69 (Prius Prime) and
            # 127 kW (Volt) at 1 Hz, rule: 68 / 79 / 123 kW). Charge power stays RoadSim's 0.6 x discharge.
            trac = sum(m['max_power_kW'] for m in spec['motors'] if m['role'] in ('traction', 'MG2', 'MG1'))
            spec['battery']['max_power_kW'] = round(min(trac / 0.9, 9.0 * c), 1)
            prov['battery'] += '; max_power_kW derived: min(traction machines / 0.9, 9 C) (plug-in rule, 13 Nov 2026)'
    if arch.startswith('hev') and re.search(r'\bP[01]\b', f.get('arch_note') or ''):
        # mild hybrids (belt starter-generator P0, crankshaft motor P1): the engine cannot be decoupled, so no electric-only
        # driving; assist, regen, stop/start and fuel cut only (strategy parameter engine_on_moving, 13 Nov 2026)
        spec['control'] = {'strategy': 'aecms', 'params': {'engine_on_moving': True}}
        prov['control'] = 'derived: P0/P1 mild hybrid (arch note) -> A-ECMS with engine_on_moving'
    if f.get('hvac_source'):
        spec.setdefault('aux', {}).setdefault('HVAC', {})['source'] = f['hvac_source']; prov['aux.HVAC.source'] = HAND
        if f.get('hvac_ptc_W') is not None:
            spec['aux']['HVAC']['ptc_max_W'] = f['hvac_ptc_W']; prov['aux.HVAC.ptc_max_W'] = HAND
    else:
        prov['aux.HVAC.source'] = DEF + ' (architecture default)'
    # ---- road load
    if abc and all(x is not None for x in abc):
        A, B, C = abc; prov['road_load_ABC'] = f'{D3}: dyno target A/B/C (median of {d3["n_abc"]} tests)'
    elif r is not None:
        A, B, C = float(r.A_lbf), float(r.B_lbf_mph), float(r.C_lbf_mph2); prov['road_load_ABC'] = EP + ' target A/B/C'
    else:
        A = B = C = None; prov['road_load_ABC'] = DEF
    if A is not None and mass:
        spec = spec_from_abc(spec, A, B, C, mass_kg=mass)
        prov['CdA_m2'] = prov['Crr'] = prov['Crr_speed_coef'] = 'derived: from A/B/C at 1.1767 kg/m3 (spec_from_abc)'
    spec['provenance'] = prov
    spec['epa_match'] = None if r is None else dict(epa_vehicle_id=str(r.name), model_year=int(r.model_year), model=str(r.model),
                                                   displacement_L=None if pd.isna(r.displacement_L) else float(r.displacement_L),
                                                   rated_hp=None if pd.isna(r.rated_hp) else float(r.rated_hp), trans_code=str(r.trans_code),
                                                   gears=None if pd.isna(r.gears) else int(r.gears), etw_lb=float(r.etw_lb),
                                                   A_lbf=float(r.A_lbf), B_lbf_mph=float(r.B_lbf_mph), C_lbf_mph2=float(r.C_lbf_mph2), note=note)
    if r is None:
        spec['epa_match_note'] = note
    spec['category'] = cat
    spec['missing'] = [k for k in ('engine.displacement_L', 'engine.max_power_kW') if prov.get(k) == DEF and arch != 'bev']
    fill_spec({k: v for k, v in spec.items() if k not in ('provenance', 'epa_match', 'category', 'missing', 'name', 'epa_match_note')})
    return spec


def d3_vehicle_settings(idx):
    """median test weight and dyno targets per vehicle from the index."""
    out = {}
    for vid, g in idx.groupby('vehicle'):
        w = g.test_weight_lb.dropna()
        a = g[['target_A_lbf', 'target_B_lbf_mph', 'target_C_lbf_mph2']].dropna()
        out[vid] = dict(test_weight_lb=float(w.median()) if len(w) else None, n_w=int(len(w)),
                        abc=[float(x) for x in a.median()] if len(a) else None, n_abc=int(len(a)),
                        abc_spread_pct_at_50mph=float(100 * np.ptp(_rl(a.target_A_lbf, a.target_B_lbf_mph, a.target_C_lbf_mph2, 50)) /
                                                      _rl(*a.median(), 50)) if len(a) > 1 else None,
                        weights_seen=sorted({float(x) for x in w}))
    return out


def main():
    facts = json.load(open(FACTS, encoding='utf-8'))['vehicles']
    idx = pd.read_parquet(os.path.join(PROC, 'index.parquet'))
    st = d3_vehicle_settings(idx)
    out = {}
    for cat, name, *_ in VEHICLES:
        f = facts.get(name)
        if f is None or not f.get('arch'):
            continue
        s = build_one(name, cat, f, st.get(name, {}))
        s['d3_settings'] = {k: v for k, v in st.get(name, {}).items()}
        out[s['id']] = s
        em = s.get('epa_match') or {}
        print(f"{s['architecture']:16s} {name:42s} {s.get('mass_kg', 0):6.0f} kg  EPA {em.get('epa_vehicle_id', '-'):32s} "
              f"{em.get('displacement_L', '')} L {em.get('rated_hp', '')} hp {em.get('trans_code', '')}{em.get('gears', '')}  "
              f"{'missing ' + ','.join(s['missing']) if s['missing'] else ''}")
    os.makedirs(PROC, exist_ok=True)
    json.dump(out, open(os.path.join(PROC, 'specs.json'), 'w'), indent=1, default=float)
    print(len(out), 'twin specs ->', os.path.join(PROC, 'specs.json'))


if __name__ == '__main__':
    main()
