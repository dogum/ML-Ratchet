"""EPA 'Data on Cars Used for Testing Fuel Economy' 2008-2026 -> one clean long table (one row per test)
and one wide table (one row per test vehicle configuration), with road-load ABC in lbf units and SI."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import glob, os, re, numpy as np, pandas as pd
ROOT = _VEH + '/data'
OUT = f'{ROOT}/proc/epa'; os.makedirs(OUT, exist_ok=True)
LBF = 4.4482216; MPH = 0.44704

OLD = {'MDLYR_DT': 'Model Year', 'VI_MFR_NM': 'Vehicle Manufacturer Name', 'VI_MFR_CD': 'Veh Mfr Code',
       'CL_NM': 'Represented Test Veh Model', 'OV_ID': 'Test Vehicle ID', 'VC_CNFG_NUM': 'Test Veh Configuration #',
       'CLS_TYP_CD': 'Vehicle Type', 'VC_RTD_HP_MSR': 'Rated Horsepower', 'TRNS': 'trans_code', 'DRV_SYS_CD': 'Drive System Code',
       'VC_DSN_ETW_MSR': 'Equivalent Test Weight (lbs.)', 'VC_AXLE_RAT_MSR': 'Axle Ratio', 'VC_NV_RAT_MSR': 'N/V Ratio',
       'TST_PRC_CD': 'Test Procedure Cd', 'TST_NUM_ID': 'Test Number', 'VEH_FL_TYP_CD': 'Test Fuel Type Cd',
       'CMYT_CO2_FE_MSR': 'CO2 (g/mi)', 'GT_RND_ADJ_QTY': 'RND_ADJ_FE', 'TPF_DYN_TRK_A_MSR': 'Target Coef A (lbf)',
       'TPF_DYN_TRK_B_MSR': 'Target Coef B (lbf/mph)', 'TPF_DYN_TRK_C_MSR': 'Target Coef C (lbf/mph**2)',
       'TPF_EDYN_TRK_A_MSR': 'Set Coef A (lbf)', 'TPF_EDYN_TRK_B_MSR': 'Set Coef B (lbf/mph)', 'TPF_EDYN_TRK_C_MSR': 'Set Coef C (lbf/mph**2)',
       'ENG_CD': 'Engine Code', 'EF_ID': 'Actual Tested Testgroup', 'VC_CYL_CNT': '# of Cylinders and Rotors'}

CYCLE = {3: 'HWY', 2: 'FTP', 31: 'FTP', 25: 'FTP', 35: 'FTP', 21: 'FTP',   # fixed 13 Nov: code 3 is HWFE, 21 is the federal 2-day FTP (was swapped)
         90: 'US06', 95: 'SC03', 11: 'FTP_cold',
         81: 'CD_UDDS', 84: 'CD_HWY', 83: 'CD_US06', 85: 'CD_SC03', 86: 'CD_FTP_cold', 80: 'CD', 82: 'CD'}
HYB = re.compile(r'HYBRID|PRIUS|INSIGHT|\bHEV\b|IONIQ(?!.*(?:ELECTRIC|EV|PLUG))|NIRO(?!.*(?:EV|PLUG|PHEV))|C-MAX(?!.*ENERGI)|CT 200H|\b\d{3}H\b|RX ?4[05]0H|NX ?\d{3}H|ES ?300H|UX ?250H', re.I)
PLUG = re.compile(r'PLUG|PHEV|ENERGI|VOLT\b|PRIME|\b4XE\b|RECHARGE|E-HYBRID|CLARITY PLUG|\bT8\b|XDRIVE\d+E|\d{3}E\b', re.I)


def load_all():
    frames = []
    for f in sorted(glob.glob(f'{ROOT}/raw/epa/*')):
        b = os.path.basename(f)
        if not re.match(r'\d\d-?(tst|test)car', b): continue
        d = pd.read_csv(f, encoding='latin1', low_memory=False) if f.endswith('.csv') else pd.read_excel(f)
        if 'MDLYR_DT' in d.columns:
            d = d.rename(columns=OLD)
            d['Test Veh Displacement (L)'] = d['GBE_CID_MSR'] * 0.016387064
            d['Represented Test Veh Make'] = d['Vehicle Manufacturer Name']
            d['Vehicle Type'] = d['Vehicle Type'].map({'C': 'Car', 'T': 'Truck'})
            d['# of Gears'] = pd.to_numeric(d.trans_code.str[1:], errors='coerce')
            d['Tested Transmission Type Code'] = d.trans_code.str[0]
            d['Drive System Description'] = d['Drive System Code'].map({'F': '2-Wheel Drive, Front', 'R': '2-Wheel Drive, Rear', '4': '4-Wheel Drive'})
        d['source_file'] = b
        frames.append(d)
    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    d = load_all()
    # fuel-code -> description from the newer files (codes are shared)
    fmap = d.dropna(subset=['Test Fuel Type Description']).groupby('Test Fuel Type Cd')['Test Fuel Type Description'].agg(lambda s: s.mode().iloc[0])
    d['Test Fuel Type Description'] = d['Test Fuel Type Description'].fillna(d['Test Fuel Type Cd'].map(fmap))
    pmap = d.dropna(subset=['Test Procedure Description']).groupby('Test Procedure Cd')['Test Procedure Description'].agg(lambda s: s.mode().iloc[0])
    d['Test Procedure Description'] = d['Test Procedure Description'].fillna(d['Test Procedure Cd'].map(pmap))
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
        'trans_code': d['Tested Transmission Type Code'], 'trans_type': d.get('Tested Transmission Type'),
        'gears': pd.to_numeric(d['# of Gears'], errors='coerce'), 'drive': d['Drive System Description'],
        'etw_lb': pd.to_numeric(d['Equivalent Test Weight (lbs.)'], errors='coerce'),
        'axle_ratio': pd.to_numeric(d['Axle Ratio'], errors='coerce'), 'nv_ratio': pd.to_numeric(d['N/V Ratio'], errors='coerce'),
        'test_number': d['Test Number'].astype(str), 'test_proc_cd': pd.to_numeric(d['Test Procedure Cd'], errors='coerce'),
        'test_proc': d['Test Procedure Description'], 'test_fuel': d['Test Fuel Type Description'],
        'analytically_derived': d.get('Analytically Derived FE?'),
        'fe_mpg': pd.to_numeric(d['RND_ADJ_FE'], errors='coerce'),
        'co2_g_mi': pd.to_numeric(d['CO2 (g/mi)'], errors='coerce'),
        'fe_bag1': pd.to_numeric(d.get('FE Bag 1'), errors='coerce'), 'fe_bag2': pd.to_numeric(d.get('FE Bag 2'), errors='coerce'),
        'fe_bag3': pd.to_numeric(d.get('FE Bag 3'), errors='coerce'), 'fe_bag4': pd.to_numeric(d.get('FE Bag 4'), errors='coerce'),
        'A_lbf': pd.to_numeric(d['Target Coef A (lbf)'], errors='coerce'),
        'B_lbf_mph': pd.to_numeric(d['Target Coef B (lbf/mph)'], errors='coerce'),
        'C_lbf_mph2': pd.to_numeric(d['Target Coef C (lbf/mph**2)'], errors='coerce'),
        'setA_lbf': pd.to_numeric(d['Set Coef A (lbf)'], errors='coerce'),
        'setB_lbf_mph': pd.to_numeric(d['Set Coef B (lbf/mph)'], errors='coerce'),
        'setC_lbf_mph2': pd.to_numeric(d['Set Coef C (lbf/mph**2)'], errors='coerce'),
        'police': d.get('Police - Emergency Vehicle?'), 'source_file': d['source_file']})
    t['cycle'] = t.test_proc_cd.map(CYCLE).fillna('other')
    fuel = t.test_fuel.fillna('').str.lower()
    t['fuel'] = np.select([fuel.str.contains('electric'), fuel.str.contains('diesel'), fuel.str.contains('e85'),
                           fuel.str.contains('hydrogen'), fuel.str.contains('natural gas|cng'), fuel.str.contains('gasoline|premium|regular|cert')],
                          ['electricity', 'diesel', 'e85', 'hydrogen', 'cng', 'gasoline'], 'other')
    # Electric tests: FE_UNIT always says 'MPG' but values mix MPGe (~60-200), what look like kWh/100 mi or blended-PHEV
    # mpg (10-50) and placeholders (0, 999, 1000). Only 50-600 is taken as MPGe -> kWh/mi = 33.705 / MPGe.
    el = t.fuel == 'electricity'
    t['fe_unit_flag'] = np.where(~el, 'mpg', np.where(t.fe_mpg.between(50, 600), 'mpge',
                                 np.where(t.fe_mpg.between(10, 50, inclusive='left'), 'ambiguous_10_50', 'placeholder_or_zero')))
    t['kWh_per_mi'] = np.where(t.fe_unit_flag == 'mpge', 33.705 / t.fe_mpg.where(t.fe_mpg > 0), np.nan)
    t.loc[el & (t.fe_unit_flag != 'mpge'), 'fe_mpg'] = np.nan
    bad = (~el) & ((t.fe_mpg < 3) | (t.fe_mpg > 150))      # placeholders (10000) and typos (0.9)
    t.loc[bad, 'fe_unit_flag'] = 'implausible_dropped'; t.loc[bad, 'fe_mpg'] = np.nan
    t['A_N'] = t.A_lbf * LBF; t['B_N_per_mps'] = t.B_lbf_mph * LBF / MPH; t['C_N_per_mps2'] = t.C_lbf_mph2 * LBF / MPH ** 2
    t['test_mass_kg'] = t.etw_lb * 0.45359237
    t['epa_vehicle_id'] = ('EPA' + t.model_year.astype(str) + '_' + t.mfr_code + '_' + t.test_vehicle_id.str.replace(r'\s+', '', regex=True)
                           + '_c' + t.config.fillna(-1).astype(int).astype(str))
    # architecture guess per vehicle
    g = t.groupby('epa_vehicle_id')
    has_elec = g.fuel.transform(lambda s: (s == 'electricity').any())
    has_fuel = g.fuel.transform(lambda s: s.isin(['gasoline', 'diesel', 'e85']).any())
    nodispl = (t.displacement_L.fillna(0) <= 0.15) | (t.displacement_L >= 20)  # EVs carry a 100 L placeholder
    t.loc[(t.displacement_L >= 20) | (t.displacement_L <= 0.15), 'displacement_L'] = np.nan
    name = t.make + ' ' + t.model
    t['architecture'] = np.select(
        [has_elec & ~has_fuel & nodispl, t.fuel == 'hydrogen', has_elec & (has_fuel | ~nodispl), name.str.contains(PLUG) & ~nodispl,
         name.str.contains(HYB)],
        ['bev', 'fcev', 'phev', 'phev', 'hev'], 'conv')
    t['arch_source'] = np.where(t.architecture.isin(['bev', 'phev']), 'test fuels (electric CD tests)',
                                np.where(t.architecture == 'hev', 'model-name pattern (e.g. Hybrid/Prius)', 'default'))
    t.to_parquet(f'{OUT}/testcars_tests.parquet', index=False)
    # wide table: one row per vehicle; median over repeated tests of the same cycle
    keep = ['model_year', 'mfr', 'make', 'model', 'test_vehicle_id', 'config', 'displacement_L', 'cylinders', 'rated_hp',
            'veh_type', 'trans_code', 'trans_type', 'gears', 'drive', 'etw_lb', 'test_mass_kg', 'axle_ratio', 'nv_ratio',
            'A_lbf', 'B_lbf_mph', 'C_lbf_mph2', 'A_N', 'B_N_per_mps', 'C_N_per_mps2', 'setA_lbf', 'setB_lbf_mph', 'setC_lbf_mph2',
            'architecture', 'arch_source', 'engine_code', 'test_group']
    base = t.sort_values('cycle').groupby('epa_vehicle_id')[keep].first()
    base['fuel'] = t[t.fuel != 'electricity'].groupby('epa_vehicle_id').fuel.agg(lambda s: s.mode().iloc[0])
    base['fuel'] = base.fuel.fillna('electricity')
    res = t[t.cycle != 'other'].groupby(['epa_vehicle_id', 'cycle']).agg(mpg=('fe_mpg', 'median'), co2=('co2_g_mi', 'median'),
                                                                        kwh=('kWh_per_mi', 'median'), n=('fe_mpg', 'size'))
    w = res.unstack('cycle')
    w.columns = [f'{c}_{a}' for a, c in w.columns]
    w = w.rename(columns=lambda c: c.replace('_mpg', '_mpg').replace('_co2', '_co2_g_mi').replace('_kwh', '_kWh_per_mi'))
    w = w.dropna(axis=1, how='all')
    veh = base.join(w)
    veh.to_parquet(f'{OUT}/testcars_vehicles.parquet')
    print(t.shape, veh.shape)
    print(t.cycle.value_counts().to_string()); print(t.fuel.value_counts().to_string())
    print(veh.groupby('architecture').size()); print(veh.model_year.value_counts().sort_index().to_string())
    print('ABC present', veh[['A_lbf', 'B_lbf_mph', 'C_lbf_mph2']].notna().all(axis=1).mean())


if __name__ == '__main__':
    main()
