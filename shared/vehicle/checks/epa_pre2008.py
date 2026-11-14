"""EPA 'Data on Cars Used for Testing Fuel Economy', model years 2000-2007, in the wide format of RoadData's
data/proc/epa/testcars_vehicles.parquet (one row per test vehicle configuration).

RoadData's table starts at MY2008. The 2000-2007 files have the same coded columns as the 2008-2016 CSVs and the same
target road-load ABC. They were downloaded on 14 Nov 2026 for the RoadSim v3 pass (exam-3) from
https://www.epa.gov/sites/default/files/2016-07/<yy>tstcar.csv into data/raw/epa_pre2008/ (git-ignored, not
tracked). The transformation repeats data/scripts/epa_testcars.py (same cycle codes, fuel and architecture rules,
same epa_vehicle_id scheme); the family split is that of data/scripts/make_splits_all.py (make + base model; a
family is dev only if it is in the frozen dev list of data/proc/splits.json).

    import epa_pre2008; w = epa_pre2008.vehicles()     # pandas DataFrame, index epa_vehicle_id
"""
import os, sys, glob, json, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
RAW = os.path.join(VEH, 'data', 'raw', 'epa_pre2008')
URL = 'https://www.epa.gov/sites/default/files/2016-07/{yy}tstcar.csv'
YEARS = ['00', '01', '02', '03', '04', '05', '06', '07']


def _modules():
    sys.path.insert(0, os.path.join(VEH, 'data', 'scripts'))
    import epa_testcars as ET
    src = open(os.path.join(VEH, 'data', 'scripts', 'make_splits_all.py'), encoding='utf-8').read()
    ns = {}
    # the family() rule and its constants only (the script itself writes the frozen splits when run)
    head = 'import re\n' + src[src.index('GENERIC = '):src.index('RANK = ')]
    exec(compile(head, 'make_splits_all.py(family)', 'exec'), ns)
    return ET, ns['family']


def raw_files():
    return [os.path.join(RAW, f'{y}tstcar.csv') for y in YEARS]


def tests():
    """long table: one row per test."""
    ET, family = _modules()
    fr = []
    for f in raw_files():
        if not os.path.exists(f):
            raise FileNotFoundError(f'{f} missing: download {URL.format(yy=os.path.basename(f)[:2])}')
        d = pd.read_csv(f, encoding='latin1', low_memory=False)
        d['source_file'] = os.path.basename(f)
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d = d.rename(columns=ET.OLD)
    d['Test Veh Displacement (L)'] = d['GBE_CID_MSR'] * 0.016387064
    d['Represented Test Veh Make'] = d['Vehicle Manufacturer Name']
    d['Vehicle Type'] = d['Vehicle Type'].map({'C': 'Car', 'T': 'Truck'})
    d['# of Gears'] = pd.to_numeric(d.trans_code.astype(str).str[1:], errors='coerce')
    d['Tested Transmission Type Code'] = d.trans_code.astype(str).str[0]
    d['Drive System Description'] = d['Drive System Code'].map({'F': '2-Wheel Drive, Front', 'R': '2-Wheel Drive, Rear', '4': '4-Wheel Drive'})
    t = pd.DataFrame({
        'model_year': pd.to_numeric(d['Model Year'], errors='coerce').astype('Int64'),
        'mfr': d['Vehicle Manufacturer Name'].astype(str).str.strip(),
        'mfr_code': d['Veh Mfr Code'].astype(str),
        'make': d['Represented Test Veh Make'].astype(str).str.strip().str.upper(),
        'model': d['Represented Test Veh Model'].astype(str).str.strip(),
        'test_vehicle_id': d['Test Vehicle ID'].astype(str).str.strip(),
        'config': pd.to_numeric(d['Test Veh Configuration #'], errors='coerce'),
        'displacement_L': pd.to_numeric(d['Test Veh Displacement (L)'], errors='coerce').round(2),
        'cylinders': pd.to_numeric(d['# of Cylinders and Rotors'], errors='coerce'),
        'rated_hp': pd.to_numeric(d['Rated Horsepower'], errors='coerce'),
        'veh_type': d['Vehicle Type'], 'test_group': d['Actual Tested Testgroup'], 'engine_code': d['Engine Code'],
        'trans_code': d['Tested Transmission Type Code'], 'trans_type': None,
        'gears': pd.to_numeric(d['# of Gears'], errors='coerce'), 'drive': d['Drive System Description'],
        'etw_lb': pd.to_numeric(d['Equivalent Test Weight (lbs.)'], errors='coerce'),
        'axle_ratio': pd.to_numeric(d['Axle Ratio'], errors='coerce'), 'nv_ratio': pd.to_numeric(d['N/V Ratio'], errors='coerce'),
        'test_proc_cd': pd.to_numeric(d['Test Procedure Cd'], errors='coerce'),
        'fuel_cd': pd.to_numeric(d['Test Fuel Type Cd'], errors='coerce'),
        'fe_mpg': pd.to_numeric(d['RND_ADJ_FE'], errors='coerce'),
        'A_lbf': pd.to_numeric(d['Target Coef A (lbf)'], errors='coerce'),
        'B_lbf_mph': pd.to_numeric(d['Target Coef B (lbf/mph)'], errors='coerce'),
        'C_lbf_mph2': pd.to_numeric(d['Target Coef C (lbf/mph**2)'], errors='coerce'),
        'source_file': d['source_file']})
    t['cycle'] = t.test_proc_cd.map(ET.CYCLE).fillna('other')
    # test fuel codes of these files: 61 Tier 2 cert gasoline, 6 / 23 / 38 / 9 gasoline grades and E-blends seen on
    # gasoline cars; 22 / 19 / 44 diesel / CNG / other (not used below: only gasoline-group hybrids are taken)
    t['fuel'] = np.where(t.fuel_cd.isin([61, 6, 23, 38, 9, 41]), 'gasoline', 'other')
    bad = (t.fe_mpg < 3) | (t.fe_mpg > 150)
    t.loc[bad, 'fe_mpg'] = np.nan
    t['epa_vehicle_id'] = ('EPA' + t.model_year.astype(str) + '_' + t.mfr_code + '_' + t.test_vehicle_id.str.replace(r'\s+', '', regex=True)
                           + '_c' + t.config.fillna(-1).astype(int).astype(str))
    name = t.make + ' ' + t.model
    t['architecture'] = np.select([name.str.contains(ET.PLUG), name.str.contains(ET.HYB)], ['phev', 'hev'], 'conv')
    t['family'] = [family(a, b) for a, b in zip(t.make, t.model)]
    return t


def vehicles():
    """wide table, one row per vehicle configuration (median over repeated tests of a cycle), with family and split."""
    t = tests()
    keep = ['model_year', 'mfr', 'make', 'model', 'test_vehicle_id', 'config', 'displacement_L', 'cylinders', 'rated_hp',
            'veh_type', 'trans_code', 'trans_type', 'gears', 'drive', 'etw_lb', 'axle_ratio', 'nv_ratio',
            'A_lbf', 'B_lbf_mph', 'C_lbf_mph2', 'architecture', 'engine_code', 'test_group', 'family', 'source_file']
    base = t.sort_values('cycle').groupby('epa_vehicle_id')[keep].first()
    base['fuel'] = t.groupby('epa_vehicle_id').fuel.agg(lambda s: s.mode().iloc[0])
    res = t[t.cycle.isin(['FTP', 'HWY', 'US06', 'SC03', 'FTP_cold'])].groupby(['epa_vehicle_id', 'cycle']).fe_mpg.median().unstack()
    res.columns = [f'{c}_mpg' for c in res.columns]
    w = base.join(res)
    for c in ('FTP_mpg', 'HWY_mpg', 'US06_mpg', 'SC03_mpg', 'FTP_cold_mpg'):
        if c not in w:
            w[c] = np.nan
    dev = set(json.load(open(os.path.join(VEH, 'data', 'proc', 'splits.json')))['EPA_testcars_and_fueleconomy']['dev_families'])
    w['split'] = np.where(w.family.isin(dev), 'dev', 'exam')
    return w


if __name__ == '__main__':
    w = vehicles()
    print(w.shape, w.architecture.value_counts().to_dict())
    h = w[w.architecture != 'conv']
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    print(h[['model_year', 'make', 'model', 'displacement_L', 'FTP_mpg', 'HWY_mpg', 'family', 'split']].sort_values(['family', 'model_year']).to_string())
