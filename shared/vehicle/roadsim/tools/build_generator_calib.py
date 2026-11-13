"""Calibration data for the vehicle generator, from real light-duty data:

- EPA test-car lists MY2018, 2022-2024 (epa.gov): per record equivalent test weight, target road-load A/B/C, rated power,
  displacement, transmission, test-car class (car vs truck + model keywords -> car/suv/pickup/van), fuel type.
  Saved as a compact table of real (ETW, CdA, Crr, Crr_v, P_kW, disp, n_gears, trans) rows per class x powertrain group,
  which the generator bootstraps (with jitter) so cross-correlations are kept.
- fueleconomy.gov vehicles.csv (MY2018-2026): EV usable battery energy ~ range x combined consumption (combE, kWh/100 mi),
  by class; hybrid share and EV motor power where listed.
Writes roadsim/data/generator_calib.json.
"""
import os, re, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, '..', 'refdata')
OUT = os.path.join(HERE, '..', 'roadsim', 'data', 'generator_calib.json')
LBF, MPH, LB = 4.4482216, 0.44704, 0.45359237

PICKUP = r'F-?150|F-?250|SILVERADO|SIERRA|RAM ?1500|RAM ?2500|TACOMA|TUNDRA|RANGER|COLORADO|CANYON|MAVERICK|FRONTIER|TITAN|GLADIATOR|RIDGELINE|SANTA CRUZ|LIGHTNING|CYBERTRUCK|R1T|HUMMER EV PICKUP|PICKUP'
VAN = r'TRANSIT|SPRINTER|PROMASTER|EXPRESS|SAVANA|PACIFICA|SIENNA|ODYSSEY|CARNIVAL|METRIS|VOYAGER|E-TRANSIT|ID\. ?BUZZ|VAN'


def cls_of(vtype, model):
    m = str(model).upper()
    if re.search(PICKUP, m): return 'pickup'
    if re.search(VAN, m): return 'van'
    if str(vtype).strip().lower() == 'car': return 'car'
    return 'suv'


def testcar_rows():
    files = ['18tstcar-2018-10-24.xlsx', '22-testcar-2023-06-13.xlsx', '23-testcar-2024-05-17_0.xlsx', '24-testcar-2025-05.xlsx']
    rows = []
    for f in files:
        d = pd.read_excel(os.path.join(REF, 'epa_testcar', f))
        d = d.dropna(subset=['Target Coef A (lbf)', 'Target Coef C (lbf/mph**2)', 'Equivalent Test Weight (lbs.)'])
        d = d.drop_duplicates(subset=['Represented Test Veh Make', 'Represented Test Veh Model', 'Equivalent Test Weight (lbs.)',
                                      'Target Coef A (lbf)', 'Target Coef B (lbf/mph)', 'Target Coef C (lbf/mph**2)'])
        for _, r in d.iterrows():
            fuel = str(r['Test Fuel Type Description'])
            grp = 'bev' if 'Electric' in fuel else ('diesel' if 'Diesel' in fuel else 'gasoline')
            if 'Hydrogen' in fuel or 'E85' in fuel: continue
            m = float(r['Equivalent Test Weight (lbs.)']) * LB
            A, B, C = r['Target Coef A (lbf)'] * LBF, r['Target Coef B (lbf/mph)'] * LBF / MPH, r['Target Coef C (lbf/mph**2)'] * LBF / MPH ** 2
            if not (0 < A < 1500 and C > 0 and m > 800): continue
            Crr = A / (m * 9.80665); CdA = C / (0.5 * 1.1767)
            if not (0.003 < Crr < 0.03 and 0.3 < CdA < 2.5): continue
            P = float(r['Rated Horsepower']) * 0.7457 if pd.notna(r['Rated Horsepower']) else np.nan
            t = str(r['Tested Transmission Type'])
            tr = 'CVT' if 'Continuously' in t else ('DCT' if ('Automated Manual' in t or t == 'Manual') else 'AT')
            rows.append(dict(cls=cls_of(r['Vehicle Type'], r['Represented Test Veh Model']), grp=grp, mass_kg=round(m, 1),
                             CdA=round(CdA, 4), Crr=round(Crr, 5), Crr_v=round(float(np.clip(B / max(A, 1e-6), -0.03, 0.03)), 5),
                             P_kW=round(P, 1), disp_L=float(r['Test Veh Displacement (L)']) if grp != 'bev' else 0.0,
                             n_gears=int(r['# of Gears']) if pd.notna(r['# of Gears']) else 1, trans=tr,
                             year=int(r['Model Year'])))
    return pd.DataFrame(rows)


def fueleconomy_ev():
    d = pd.read_csv(os.path.join(REF, 'fueleconomy', 'vehicles.csv'), low_memory=False)
    d = d[(d.year >= 2018) & (d.atvType == 'EV') & (d['range'] > 50) & (d.combE > 15)]
    vc = d.VClass.str.lower()
    cls = np.where(vc.str.contains('pickup'), 'pickup', np.where(vc.str.contains('van'), 'van',
                   np.where(vc.str.contains('sport utility'), 'suv', 'car')))
    # label combE (kWh/100 mi, adjusted, wall energy) x range -> usable DC energy ~ x 0.9
    E = d['range'] * d['combE'] / 100.0 * 0.9
    mot = d['evMotor'].astype(str).str.extract(r'(\d{2,3})\s*[- ]?kW', expand=False).astype(float)
    return pd.DataFrame(dict(cls=cls, batt_kWh=E.round(1), motor_kW=mot, year=d.year))


def main():
    tc = testcar_rows()
    ev = fueleconomy_ev()
    calib = {'source': __doc__.strip(), 'testcar': {}, 'ev_battery': {}, 'counts': {}}
    for (c, g), sub in tc.groupby(['cls', 'grp']):
        calib['testcar'][f'{c}|{g}'] = sub.drop(columns=['cls', 'grp']).to_dict(orient='list')
        calib['counts'][f'{c}|{g}'] = int(len(sub))
    for c, sub in ev.groupby('cls'):
        calib['ev_battery'][c] = dict(batt_kWh=sub.batt_kWh.tolist(), motor_kW=sub.motor_kW.dropna().tolist())
    json.dump(calib, open(OUT, 'w'))
    print(calib['counts'], {k: len(v['batt_kWh']) for k, v in calib['ev_battery'].items()})
    print(tc.groupby(['cls', 'grp'])[['mass_kg', 'CdA', 'Crr', 'P_kW']].median())


if __name__ == '__main__':
    main()
