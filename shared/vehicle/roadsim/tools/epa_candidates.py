"""Collect EPA test-car records into one row per tested vehicle configuration (FTP / HWFE / US06 / SC03 / ColdCO / CD tests)."""
import pandas as pd, sys, os
REF = os.path.join(os.path.dirname(__file__), '..', 'refdata')
files = {2024: '24-testcar-2025-05.xlsx', 2023: '23-testcar-2024-05-17_0.xlsx', 2022: '22-testcar-2023-06-13.xlsx'}
proc = {'Federal fuel 2-day exhaust (w/can load)': 'FTP', 'Federal fuel 3-day exhaust': 'FTP', 'CVS 75 and later (w/o can. load)': 'FTP',
        'HWFE': 'HWY', 'US06': 'US06', 'SC03': 'SC03', 'Cold CO': 'COLD', 'Charge Depleting UDDS': 'CD_UDDS',
        'Charge Depleting Highway': 'CD_HWY', 'Charge Depleting US06': 'CD_US06'}
rows = []
for yr, f in files.items():
    df = pd.read_excel(os.path.join(REF, 'epa_testcar', f))
    df['proc'] = df['Test Procedure Description'].map(proc)
    df = df[df['proc'].notna()]
    key = ['Model Year', 'Represented Test Veh Make', 'Represented Test Veh Model', 'Test Vehicle ID', 'Test Veh Configuration #']
    for k, g in df.groupby(key):
        r = g.iloc[0]
        d = dict(zip(key, k))
        for c in ['Test Veh Displacement (L)', 'Rated Horsepower', '# of Cylinders and Rotors', 'Tested Transmission Type', '# of Gears',
                  'Drive System Description', 'Equivalent Test Weight (lbs.)', 'Axle Ratio', 'N/V Ratio', 'Target Coef A (lbf)',
                  'Target Coef B (lbf/mph)', 'Target Coef C (lbf/mph**2)', 'Vehicle Type', 'Test Fuel Type Description']:
            d[c] = r[c]
        for p, gg in g.groupby('proc'):
            d['FE_' + p] = gg['RND_ADJ_FE'].mean()
            if p == 'FTP':
                for b in ['FE Bag 1', 'FE Bag 2', 'FE Bag 3']:
                    d[b] = gg[b].mean()
            if p == 'COLD':
                d['COLD_bag1'] = gg['FE Bag 1'].mean(); d['COLD_bag2'] = gg['FE Bag 2'].mean(); d['COLD_bag3'] = gg['FE Bag 3'].mean()
        fuels = set(g['Test Fuel Type Description'])
        d['fuels'] = ';'.join(sorted(map(str, fuels)))
        rows.append(d)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(REF, 'epa_candidates.csv'), index=False)
print(out.shape)
