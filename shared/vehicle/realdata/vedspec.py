"""VED static data -> vehicle table and vehicle-spec dicts (shared contract format).
VED anonymised make/model/year; we keep what is published and infer the rest with a stated confidence."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import re, numpy as np, pandas as pd

DATA = _VEH + '/data'
LB = 0.45359237


def _arch(vtype, eng, displ, cyl):
    """Architecture guess + confidence + likely models (reasoning recorded for DATA.md)."""
    if vtype == 'ICE':
        return 'conv', 'high', ''
    if vtype == 'EV':
        return 'bev', 'high', '2013 Nissan Leaf (VED README)'
    if vtype == 'PHEV':
        if displ == 1.4: return 'phev_series', 'high', 'Chevrolet Volt gen 1 (only 1.4 L PHEV of the era; GM-style fuel-rate PID)'
        if displ == 1.5: return 'phev_series', 'medium', 'Chevrolet Volt gen 2 (1.5 L; same GM signal set as the 1.4 L cars)'
        if displ == 2.0: return 'phev_powersplit', 'high', 'Ford C-Max / Fusion Energi (2.0 L, eCVT)'
        if displ == 1.8: return 'phev_powersplit', 'high', 'Toyota Prius Plug-in / Prius Prime (1.8 L)'
    if vtype == 'HEV':
        if displ in (1.8, 2.5, 3.3, 3.5, 2.3) or (displ == 1.5 and cyl == 4 and 'GAS/ELECTRIC' in eng):
            return 'hev_powersplit', 'medium', {1.8: 'Toyota Prius / Lexus CT', 2.5: 'Toyota Camry/Avalon/RAV4 / Lexus ES hybrid',
                                                3.3: 'Toyota Highlander / Lexus RX 400h', 3.5: 'Toyota Highlander / Lexus RX 450h',
                                                2.3: 'Ford Escape / Mercury Mariner hybrid',
                                                1.5: 'Toyota Prius c (or Honda Civic hybrid, a parallel IMA car)'}[displ]
        if displ == 2.0: return 'hev_powersplit', 'medium', 'Ford Fusion / C-Max hybrid (or Hyundai/Kia 2.0 parallel)'
        if displ == 2.4: return 'hev_powersplit', 'low', 'Toyota Camry hybrid 2007-11 (power split) or Hyundai Sonata hybrid (parallel P2)'
        if displ in (1.0, 1.3): return 'hev_parallel_p2', 'medium', 'Honda Insight / Civic hybrid (IMA: parallel P1, closest contract type P2)'
        if displ == 1.6: return 'hev_parallel_p2', 'medium', 'Hyundai Ioniq / Kia Niro hybrid (parallel P2, DCT)'
    return 'hev_powersplit', 'low', ''


def ved_vehicle_table():
    a = pd.read_excel(f'{DATA}/raw/ved/VED_Static_Data_ICE&HEV.xlsx').rename(columns={'Vehicle Type': 'ved_type'})
    b = pd.read_excel(f'{DATA}/raw/ved/VED_Static_Data_PHEV&EV.xlsx').rename(columns={'EngineType': 'ved_type'})
    st = pd.concat([a, b], ignore_index=True)
    st.columns = ['VehId', 'ved_type', 'ved_class', 'engine_str', 'trans_str', 'drive_str', 'gen_weight_lb']
    rows = []
    for r in st.itertuples():
        eng = str(r.engine_str)
        m = re.search(r'(\d\.\d)\s*L', eng) or re.search(r'(\d\.\d)', eng)
        displ = float(m.group(1)) if m else np.nan
        c = re.match(r'^(\d+)-', eng) or re.search(r'(\d)cyl', eng) or re.search(r'[IVH]-?(\d+)', eng)
        cyl = int(c.group(1)) if c else np.nan
        if cyl == 242: cyl = 6
        fuel = 'none' if r.ved_type == 'EV' else ('diesel' if 'DSL' in eng else 'gasoline')
        arch, conf, models = _arch(r.ved_type, eng, displ, cyl)
        t = str(r.trans_str).upper()
        trans = ('CVT' if 'CVT' in t else 'manual' if 'MANUAL' in t else 'AT' if 'AUTO' in t else None)
        if r.ved_type in ('HEV', 'PHEV') and arch.endswith('powersplit'): trans = 'eCVT'
        if r.ved_type == 'EV' or arch == 'phev_series': trans = 'single'
        gw = pd.to_numeric(r.gen_weight_lb, errors='coerce')
        cls = {'Car': 'car', 'SUV': 'suv'}.get(r.ved_class)
        if cls is None and np.isfinite(gw):
            cls = 'pickup' if (fuel == 'diesel' or (np.isfinite(displ) and displ >= 4.6)) else ('suv' if gw >= 4500 else 'car')
        rows.append(dict(vehicle_id=f'VED_{r.VehId}', VehId=r.VehId, ved_type=r.ved_type, architecture=arch,
                         arch_confidence=conf, likely_models=models, cls=cls or 'car', cls_source='VED' if r.ved_class in ('Car', 'SUV') else 'inferred',
                         fuel=fuel, displacement_L=displ, cylinders=cyl, turbo=('T/C' in eng or 'S/C' in eng),
                         transmission=trans, trans_str=r.trans_str, drive=None if r.drive_str == 'NO DATA' else r.drive_str,
                         gen_weight_lb=gw, engine_str=eng))
    return pd.DataFrame(rows).set_index('vehicle_id')


def spec_from_row(r):
    """Vehicle spec dict in the shared contract format. Unknowns are None (Phase 2 infers them)."""
    curb = r.gen_weight_lb * LB if np.isfinite(r.gen_weight_lb) else None
    return {
        'id': r.name, 'class': r.cls, 'architecture': r.architecture, 'source': 'VED',
        'mass_kg': curb if curb else None, 'payload_kg': 136.0,   # fixed 13 Nov: curb only; roadsim adds payload (was counted twice)
        'mass_note': 'mass_kg = VED Generalized_Weight (curb weight rounded to 500 lb, +-115 kg); payload_kg = 136 kg driver+load (EPA ETW convention), added once by roadsim',
        'CdA_m2': None, 'Crr': None, 'Crr_speed_coef': None, 'road_load_ABC': None,
        'wheel_radius_m': None, 'final_drive': None, 'driveline_eff': None,
        'engine': None if r.fuel == 'none' else {'fuel': r.fuel, 'displacement_L': None if not np.isfinite(r.displacement_L) else r.displacement_L,
                                                'cylinders': None if not np.isfinite(r.cylinders) else int(r.cylinders),
                                                'boosted': bool(r.turbo), 'max_power_kW': None, 'bsfc_map': None,
                                                'idle_rpm': None, 'max_rpm': None, 'warmup_model': None},
        'transmission': {'type': r.transmission, 'raw': r.trans_str, 'gear_ratios': None},
        'motors': None, 'battery': ({'chemistry': 'Li-ion', 'capacity_kWh': 24.0} if r.architecture == 'bev' else None),
        'aux': None, 'regen': None, 'control': None,
        'meta': {'arch_confidence': r.arch_confidence, 'likely_models': r.likely_models, 'drive': r.drive,
                 'engine_str': r.engine_str, 'ved_type': r.ved_type, 'class_source': r.cls_source},
    }
