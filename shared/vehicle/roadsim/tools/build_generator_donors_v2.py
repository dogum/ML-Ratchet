"""Donor subsets for generate_vehicles(donor_model='v2') (RoadSim physics v2, work package B1).

v1 draws every light-duty body from all gasoline records of its class, so a generated hybrid can get the body of a
performance car or a full-size SUV, and a generated van the body of a 15-passenger Transit. v2 draws:
- hybrid bodies (hev_*) from the real full-hybrid records of the class (model names with HYBRID / HEV / Prius / Lexus
  'h' models ...; mild 48 V hybrids excluded), PHEV bodies from the plug-in records (falling back to the hybrids);
- van bodies from minivans (Pacifica, Sienna, Odyssey, Carnival, ...) with probability VAN_MINIVAN_SHARE, else from all
  vans (full-size passenger vans).
Only record identity (model name) is used, never a test result. The subsets are stored as row indices into the
existing arrays of data/generator_calib.json (built by build_generator_calib.py from the same four EPA test-car lists),
so the v1 generator is untouched. Rows are matched on their (mass, CdA, Crr, Crr_v, P, displacement, year) values.

usage: python build_generator_donors_v2.py [dir with the EPA test-car .xlsx files]
"""
import os, re, sys, json
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_generator_calib import cls_of, LBF, MPH, LB

CAL = os.path.join(HERE, '..', 'roadsim', 'data', 'generator_calib.json')
OUT = os.path.join(HERE, '..', 'roadsim', 'data', 'generator_donors_v2.json')
FILES = ['18tstcar-2018-10-24.xlsx', '22-testcar-2023-06-13.xlsx', '23-testcar-2024-05-17_0.xlsx', '24-testcar-2025-05.xlsx']

PHEV = r'PHEV|PLUG-?IN|PRIME|E-HYBRID|\bPHV\b|ENERGI|4XE|RECHARGE|\bT8\b|PACIFICA HYBRID|\d{3}H\+'
HEV = (r'HYBRID|\bHEV\b|FHEV|PRIUS|INSIGHT|IONIQ|\b[A-Z]{2}\s?\d{3}H\b|\b\d{3}H\b|VENZA|CROWN|I-?FORCE MAX|POWERBOOST')
HEV_YEAR = {r'SIENNA': 2021, r'SEQUOIA': 2023, r'NIRO': 2017}       # hybrid-only nameplates from that model year
MINIVAN = r'PACIFICA|SIENNA|ODYSSEY|CARNIVAL|SEDONA|VOYAGER|GRAND CARAVAN|QUEST'
VAN_MINIVAN_SHARE = 0.6     # minivans ~0.4 M/yr vs ~0.3 M/yr full-size vans in the US (2023 sales), many of them > 8,500 lb


def kind(model, year):
    m = str(model).upper()
    if 'MHEV' in m or 'MILD' in m:
        return None
    if re.search(PHEV, m):
        return 'phev'
    if re.search(HEV, m):
        return 'hev'
    for pat, y0 in HEV_YEAR.items():
        if re.search(pat, m) and year >= y0:
            return 'hev'
    return None


def rows(src):
    """the same rows as build_generator_calib.testcar_rows (same filters and order), plus the model name."""
    out = []
    for f in FILES:
        d = pd.read_excel(os.path.join(src, f))
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
            cls = cls_of(r['Vehicle Type'], r['Represented Test Veh Model'])
            key = (round(m, 1), round(CdA, 4), round(Crr, 5), round(float(np.clip(B / max(A, 1e-6), -0.03, 0.03)), 5),
                   round(P, 1), float(r['Test Veh Displacement (L)']) if grp != 'bev' else 0.0, int(r['Model Year']))
            out.append((f'{cls}|{grp}', key, str(r['Represented Test Veh Make']).upper() + ' ' + str(r['Represented Test Veh Model']).upper()))
    return out


def main(src):
    cal = json.load(open(CAL))['testcar']
    index = {}
    for k, T in cal.items():
        pos = defaultdict(list)
        for i in range(len(T['mass_kg'])):
            P = T['P_kW'][i]
            pos[(T['mass_kg'][i], T['CdA'][i], T['Crr'][i], T['Crr_v'][i], None if P != P else P, T['disp_L'][i], T['year'][i])].append(i)
        index[k] = pos
    sub = defaultdict(lambda: defaultdict(set))
    names = defaultdict(lambda: defaultdict(set))
    miss = 0
    for k, key, name in rows(src):
        if k not in index:
            continue
        kk = tuple(None if (isinstance(x, float) and x != x) else x for x in key)
        idx = index[k].get(kk)
        if not idx:
            miss += 1
            continue
        yr = key[-1]
        kd = kind(name.split(' ', 1)[1] if ' ' in name else name, yr)
        if kd is None and re.search(r'LEXUS', name) and re.search(r'\d{3}H\b', name):
            kd = 'hev'
        if kd:
            sub[k][kd].update(idx); names[k][kd].add(name)
        if k.startswith('van|') and re.search(MINIVAN, name):
            sub[k]['minivan'].update(idx); names[k]['minivan'].add(name)
    out = {'source': __doc__.strip(), 'van_minivan_share': VAN_MINIVAN_SHARE, 'unmatched_rows': miss,
           'subsets': {k: {kd: sorted(v) for kd, v in d.items()} for k, d in sub.items()},
           'names': {k: {kd: sorted(v) for kd, v in d.items()} for k, d in names.items()}}
    json.dump(out, open(OUT, 'w'), indent=0)
    for k, d in out['subsets'].items():
        print(k, {kd: len(v) for kd, v in d.items()})
    print('unmatched rows:', miss)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'refdata', 'epa_testcar'))
