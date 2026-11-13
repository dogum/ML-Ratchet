"""Build the EPA validation vehicle set: roadsim specs for real MY2022-24 test vehicles, from the EPA test-car list
(equivalent test weight, target road-load A/B/C, engine displacement and rated power, transmission type and gears,
N/V ratio, axle ratio) plus published motor/battery figures for hybrids and EVs (manufacturer spec sheets; approximate).

A fixed-seed dev subset (about one third per architecture) is the only part used for any calibration; errors are
reported on the held-out rest.  Writes road/checks/epa_vehicles.json.
"""
import os, sys, json, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
from roadsim.vehicle import GEAR_SETS, spec_from_abc, fill_spec
from roadsim.constants import LB, MPH

REF = os.path.join(HERE, '..', 'roadsim', 'refdata', 'epa_candidates.csv')

# row index in epa_candidates.csv -> (class, architecture, extra spec)
# HEV/BEV motor & battery figures: manufacturer press/spec sheets (rounded). Rear motors of AWD hybrids are lumped
# into the main traction motor.
TABLE = {
    # ---- conventional
    1312: ('suv', 'conv', {}),             # Toyota RAV4 2.5 8AT
    2311: ('car', 'conv', {}),             # Mazda3 2.0 6AT
    644: ('car', 'conv', {}),              # Hyundai Sonata 2.5 8AT
    584: ('car', 'conv', {}),              # Hyundai Elantra 2.0 IVT
    563: ('suv', 'conv', {'turbo': True}),  # Honda CR-V 1.5T CVT
    438: ('suv', 'conv', {'turbo': True}),  # Ford Escape 1.5T 8AT
    449: ('pickup', 'conv', {'turbo': True}),  # Ford F-150 2.7T 10AT
    454: ('pickup', 'conv', {'turbo': True}),  # Ford F-150 3.5T 10AT
    1901: ('pickup', 'conv', {}),          # Ford F-150 5.0 V8 10AT
    364: ('pickup', 'conv', {}),           # Chevrolet Silverado 5.3 V8 10AT
    363: ('pickup', 'conv', {'diesel': True}),  # Chevrolet Silverado 3.0 Duramax 10AT
    688: ('suv', 'conv', {}),              # Jeep Grand Cherokee L 3.6 8AT
    392: ('van', 'conv', {}),              # Chrysler Pacifica 3.6 9AT
    3372: ('van', 'conv', {}),             # Ford Transit wagon 3.5 10AT
    474: ('pickup', 'conv', {'turbo': True}),  # Ford Maverick 2.0T 8AT
    1258: ('car', 'conv', {}),             # Toyota Camry 2.5 8AT
    1265: ('car', 'conv', {}),             # Toyota Corolla 2.0 CVT
    549: ('car', 'conv', {'turbo': True}),  # Honda Civic 1.5T CVT
    583: ('car', 'conv', {'turbo': True}),  # Hyundai Elantra N-Line 1.6T 7DCT
    647: ('car', 'conv', {'turbo': True}),  # Hyundai Sonata N-Line 2.5T 8DCT
    464: ('suv', 'conv', {'turbo': True}),  # Ford Explorer 2.3T 10AT
    1266: ('suv', 'conv', {}),             # Toyota Corolla Cross 2.0 CVT
    # ---- HEV
    1254: ('car', 'hev_powersplit', dict(mg2=88, mg1=65, batt=(1.1, 259, 'NMC'), mg2_rpm=17000)),   # Camry Hybrid LE
    1319: ('suv', 'hev_powersplit', dict(mg2=128, mg1=65, batt=(1.6, 245, 'NiMH'), mg2_rpm=17000)),  # RAV4 Hybrid AWD
    1920: ('pickup', 'hev_powersplit', dict(mg2=94, mg1=70, batt=(1.1, 280, 'NMC'), mg2_rpm=14000)),  # Maverick HEV
    439: ('suv', 'hev_powersplit', dict(mg2=88, mg1=70, batt=(1.1, 280, 'NMC'), mg2_rpm=14000)),    # Escape FWD HEV
    1269: ('suv', 'hev_powersplit', dict(mg2=110, mg1=55, batt=(0.9, 222, 'NMC'), mg2_rpm=17000)),  # Corolla Cross Hybrid AWD
    1304: ('car', 'hev_powersplit', dict(mg2=83, mg1=55, batt=(0.9, 222, 'NMC'), mg2_rpm=17000)),   # Prius 2024
    587: ('car', 'hev_parallel_p2', dict(mot=32, mot_T=170, batt=(1.32, 240, 'NMC'))),               # Elantra Hybrid 6DCT
    717: ('suv', 'hev_parallel_p2', dict(mot=32, mot_T=170, batt=(1.32, 240, 'NMC'))),               # Niro Hybrid 6DCT
    597: ('car', 'hev_parallel_p2', dict(mot=39, mot_T=205, batt=(1.76, 270, 'NMC'))),               # Sonata Hybrid 6AT(no TC)
    602: ('suv', 'hev_parallel_p2', dict(mot=44, mot_T=264, batt=(1.49, 270, 'NMC'), turbo=True)),  # Tucson Hybrid 1.6T
    452: ('pickup', 'hev_parallel_p2', dict(mot=35, mot_T=240, batt=(1.5, 300, 'NMC'), turbo=True, tc=True)),  # F-150 PowerBoost
    1343: ('pickup', 'hev_parallel_p2', dict(mot=36, mot_T=250, batt=(1.87, 288, 'NiMH'), turbo=True, tc=True, eng_kW=170)),  # Tacoma i-FORCE MAX
    544: ('car', 'hev_series', dict(mot=135, mot_T=335, gen=105, batt=(1.05, 260, 'NMC'))),           # Accord Hybrid (e:HEV)
    565: ('suv', 'hev_series', dict(mot=135, mot_T=335, gen=105, batt=(1.06, 260, 'NMC'))),           # CR-V Hybrid FWD (e:HEV)
    # ---- BEV
    1373: ('car', 'bev', dict(mot=208, batt=(57.5, 350, 'LFP'))),     # Tesla Model 3 RWD
    1380: ('suv', 'bev', dict(mot=291, batt=(75.0, 350, 'NMC'))),     # Tesla Model Y LR AWD
    1919: ('suv', 'bev', dict(mot=198, batt=(70.0, 360, 'NMC'))),     # Mustang Mach-E RWD
    705: ('suv', 'bev', dict(mot=168, batt=(77.4, 697, 'NMC'))),      # Kia EV6 LR RWD
    1047: ('car', 'bev', dict(mot=110, batt=(40.0, 350, 'NMC'))),     # Nissan Leaf 40 kWh
    1403: ('suv', 'bev', dict(mot=157, batt=(77.0, 400, 'NMC'))),     # VW ID.4 Pro
    446: ('pickup', 'bev', dict(mot=318, batt=(98.0, 400, 'NMC'))),   # Ford F-150 Lightning SR
    719: ('suv', 'bev', dict(mot=150, batt=(64.8, 358, 'NMC'))),      # Kia Niro EV
    1731: ('suv', 'bev', dict(mot=150, batt=(65.0, 350, 'NMC'), map='bolt_edu_150')),  # Chevrolet Bolt EUV
    617: ('suv', 'bev', dict(mot=125, batt=(58.0, 522, 'NMC'))),      # Hyundai Ioniq 5 SR RWD
    625: ('car', 'bev', dict(mot=111, batt=(53.0, 522, 'NMC'))),      # Hyundai Ioniq 6 SR RWD
}


def trans_of(row):
    t = str(row['Tested Transmission Type']); n = int(row['# of Gears'])
    if 'Continuously' in t:
        return 'CVT', n
    if 'Automated Manual' in t or t == 'Manual':
        return 'DCT', n
    return 'AT', n


def build():
    d = pd.read_csv(REF)
    out = []
    for idx, (cls, arch, x) in TABLE.items():
        r = d.loc[idx]
        name = f"{int(r['Model Year'])} {r['Represented Test Veh Make']} {r['Represented Test Veh Model']}"
        mass = float(r['Equivalent Test Weight (lbs.)']) * LB
        disp = float(r['Test Veh Displacement (L)'])
        P = float(r['Rated Horsepower']) * 0.7457
        nv = float(r['N/V Ratio']); axle = float(r['Axle Ratio'])
        spec = {'id': 'epa_%d' % idx, 'name': name, 'class': cls, 'architecture': arch, 'mass_kg': mass,
                'rot_mass_frac': 0.015, 'tire': {'K_t': 0.0, 'temp_effects': False}}
        rr = 0.33 if cls == 'car' else 0.36 if cls in ('suv', 'van') else 0.39
        spec['wheel_radius_m'] = rr
        k_nv = 2 * math.pi * rr / (60 * MPH)          # N/V [rpm/mph] -> overall ratio
        if arch == 'conv':
            diesel = x.get('diesel', False); turbo = x.get('turbo', False) or (P / disp > 85 and not diesel)
            if diesel: m = 'diesel_gm_30'
            elif turbo: m = 'gas_turbo_honda_15' if disp < 2.2 else 'gas_turbo_ford_27'
            elif disp >= 4.5: m = 'gas_v8_ford_73'
            elif disp >= 3.0: m = 'gas_v6_gm_43'
            else: m = 'gas_na_toyota_a25a'
            spec['engine'] = dict(fuel='diesel' if diesel else 'gasoline', displacement_L=disp, max_power_kW=P, bsfc_map=m)
            t, n = trans_of(r)
            if t == 'CVT':
                top = 0.40
                fd = axle if axle > 2 else nv * k_nv / top
                spec['transmission'] = dict(type='CVT', ratio_range=[2.6, nv * k_nv / fd])
                spec['final_drive'] = fd
            else:
                gears = GEAR_SETS.get((t, n)) or GEAR_SETS.get(('AT', n)) or list(np.round(np.geomspace(4.6, 0.65, n), 3))
                spec['transmission'] = dict(type=t, gears=gears)
                spec['final_drive'] = nv * k_nv / gears[-1]
        elif arch in ('hev_powersplit',):
            spec['engine'] = dict(fuel='gasoline', displacement_L=disp, max_power_kW=x.get('eng_kW', P), bsfc_map='gas_atk_toyota_tnga_hev_paper')
            spec['transmission'] = dict(type='eCVT', ring_teeth=78, sun_teeth=30, mg2_ratio=3.0)
            spec['final_drive'] = axle
            spec['motors'] = [dict(role='MG1', max_power_kW=x['mg1'], eff_map='ipm_generic_150', max_rpm=13500),
                              dict(role='MG2', max_power_kW=x['mg2'], eff_map='ipm_generic_150', max_rpm=x['mg2_rpm'])]
            c, V, ch = x['batt']
            spec['battery'] = dict(capacity_kWh=c, nominal_V=V, chemistry=ch, soc_window=[0.4, 0.8])
        elif arch == 'hev_parallel_p2':
            turbo = x.get('turbo', False)
            Pe = x.get('eng_kW', P)
            m = ('gas_turbo_honda_15' if disp < 2.2 else 'gas_turbo_ford_27') if turbo else 'gas_atk_toyota_tnga_hev_paper'
            spec['engine'] = dict(fuel='gasoline', displacement_L=disp, max_power_kW=Pe, bsfc_map=m)
            t, n = trans_of(r)
            t = 'AT' if x.get('tc') else 'DCT'
            gears = GEAR_SETS.get((t, n)) or GEAR_SETS.get(('AT', n))
            spec['transmission'] = dict(type=t, gears=gears)
            spec['final_drive'] = nv * k_nv / gears[-1]
            spec['motors'] = [dict(role='P2', max_power_kW=x['mot'], max_torque_Nm=x['mot_T'], eff_map='ipm_generic_150', max_rpm=7000)]
            c, V, ch = x['batt']
            spec['battery'] = dict(capacity_kWh=c, nominal_V=V, chemistry=ch, soc_window=[0.3, 0.8])
        elif arch == 'hev_series':
            spec['engine'] = dict(fuel='gasoline', displacement_L=disp, max_power_kW=P, bsfc_map='gas_atk_toyota_tnga_hev_paper')
            spec['transmission'] = dict(type='single')
            spec['final_drive'] = nv * k_nv if nv > 30 else 7.0
            spec['motors'] = [dict(role='traction', max_power_kW=x['mot'], max_torque_Nm=x['mot_T'], eff_map='ipm_generic_150', max_rpm=14000),
                              dict(role='generator', max_power_kW=x['gen'], eff_map='ipm_generic_150', max_rpm=14000)]
            c, V, ch = x['batt']
            spec['battery'] = dict(capacity_kWh=c, nominal_V=V, chemistry=ch, soc_window=[0.3, 0.8])
            spec['final_drive'] = 14000 * math.pi / 30 * rr / (180 / 3.6)
        elif arch == 'bev':
            spec['transmission'] = dict(type='single')
            fd = nv * k_nv if nv > 30 else 9.0
            spec['final_drive'] = fd
            vtop = 200 / 3.6
            spec['motors'] = [dict(role='traction', max_power_kW=x['mot'], eff_map=x.get('map', 'ipm_generic_150'),
                                   max_rpm=fd * vtop / rr * 30 / math.pi)]
            c, V, ch = x['batt']
            spec['battery'] = dict(capacity_kWh=c, nominal_V=V, chemistry=ch, soc_window=[0.02, 0.98])
        spec = spec_from_abc(spec, float(r['Target Coef A (lbf)']), float(r['Target Coef B (lbf/mph)']),
                             float(r['Target Coef C (lbf/mph**2)']), mass_kg=mass)
        bev_unit = None
        epa = {}
        for k in ['FTP', 'HWY', 'US06', 'SC03', 'COLD', 'CD_UDDS', 'CD_HWY']:
            v = r.get('FE_' + k)
            if pd.notna(v):
                epa[k] = float(v)
        for b in ['FE Bag 1', 'FE Bag 2', 'FE Bag 3', 'COLD_bag1', 'COLD_bag2', 'COLD_bag3']:
            if pd.notna(r.get(b)):
                epa[b.replace(' ', '_')] = float(r[b])
        if arch == 'bev':
            u = epa.get('CD_UDDS', 0)
            bev_unit = 'kWh/100mi' if u < 60 else 'MPGe'
        out.append(dict(row=int(idx), name=name, spec=spec, epa=epa, bev_unit=bev_unit,
                        abc=[float(r['Target Coef A (lbf)']), float(r['Target Coef B (lbf/mph)']), float(r['Target Coef C (lbf/mph**2)'])],
                        etw_lb=float(r['Equivalent Test Weight (lbs.)'])))
    # fixed-seed dev subset, stratified by architecture group
    rng = np.random.default_rng(20261112)
    for grp in ['conv', 'hev', 'bev']:
        ids = [i for i, o in enumerate(out) if (o['spec']['architecture'] == grp if grp != 'hev' else o['spec']['architecture'].startswith('hev'))]
        dev = rng.choice(ids, size=max(2, len(ids) // 3), replace=False)
        for i in ids:
            out[i]['split'] = 'dev' if i in dev else 'heldout'
    json.dump(out, open(os.path.join(HERE, 'epa_vehicles.json'), 'w'), indent=1)
    for o in out:
        fill_spec(o['spec'])
    print(len(out), 'vehicles;', sum(o['split'] == 'dev' for o in out), 'dev')
    return out


if __name__ == '__main__':
    build()
