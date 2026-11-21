"""ResOpsUS v2 daily operations + GRanD attributes (from ResOpsUS+CARS) -> proc/reservoirs.

Out: proc/reservoirs/reservoirs.parquet (one row per dam), proc/reservoirs/daily/grand_<id>.parquet
Cleaning: negative storage/outflow -> NaN (flag counted); negative inflow kept (back-calculated inflows can be
negative from mass balance) but counted; storage > 1.5 x max(GRanD CAP, ResOpsUS CAP) -> NaN (counted).
Mass-balance check: daily dS vs (I - O) * 0.0864 hm3 over days with all three, reported as slope/corr/bias.
"""
import glob, os, json, io, zipfile
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
ZIP = f'{ROOT}/raw/resopsus/ResOpsUS2.zip'   # Zenodo 6612040 'ResOpsUS 2.zip' (read in place)
CARS = f'{ROOT}/raw/resopsus_cars'
OUT = f'{ROOT}/proc/reservoirs'


def main():
    os.makedirs(f'{OUT}/daily', exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    rd = lambda n: io.BytesIO(z.read('ResOpsUS/' + n))
    ra = pd.read_csv(rd('attributes/reservoir_attributes.csv'), encoding='utf-8-sig')
    gr = pd.read_csv(f'{CARS}/v1.0__attributes__grand.csv')
    rc = pd.read_csv(f'{CARS}/v1.0__attributes__resops.csv')[['GRAND_ID', 'CAP_RESOPS']]
    gr = gr.merge(rc, on='GRAND_ID', how='left')
    rows = []
    names = sorted(n for n in z.namelist() if n.startswith('ResOpsUS/time_series_all/ResOpsUS_') and n.endswith('.csv'))
    for p in names:
        did = int(os.path.basename(p)[9:-4])
        d = pd.read_csv(io.BytesIO(z.read(p)), na_values=['NA'])
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d = d.dropna(subset=['date']).set_index('date').sort_index()
        d = d.apply(pd.to_numeric, errors='coerce')
        d = d[~d.index.duplicated()]
        g = gr[gr.GRAND_ID == did]
        g = g.iloc[0] if len(g) else None
        cap = float(g.CAP_MCM) if g is not None and pd.notna(g.CAP_MCM) else np.nan
        cap_r = float(g.CAP_RESOPS) if g is not None and pd.notna(g.CAP_RESOPS) else np.nan
        capmax = np.nanmax([cap, cap_r, 0])
        st = {'n_neg_storage': int((d.storage < 0).sum()), 'n_neg_outflow': int((d.outflow < 0).sum()),
              'n_neg_inflow': int((d.inflow < 0).sum())}
        d.loc[d.storage < 0, 'storage'] = np.nan
        d.loc[d.outflow < 0, 'outflow'] = np.nan
        if capmax > 0:
            st['n_storage_gt_1p5cap'] = int((d.storage > 1.5 * capmax).sum())
            d.loc[d.storage > 1.5 * capmax, 'storage'] = np.nan
        out = pd.DataFrame({'storage_hm3': d.storage, 'inflow_cms': d.inflow, 'release_cms': d.outflow,
                            'elevation_m': d.elevation, 'evaporation': d.evaporation}).astype('float32')
        out.index.name = 'date'
        rid = f'grand_{did}'
        out.to_parquet(f'{OUT}/daily/{rid}.parquet', compression='zstd')
        # mass balance: try storage stamped at end of day (dS = S(t) - S(t-1)) and start of day (S(t+1) - S(t))
        net = (out.inflow_cms - out.release_cms) * 0.0864
        mb = {}
        best = None
        for lab, ds in [('end_of_day', out.storage_hm3.diff()), ('start_of_day', out.storage_hm3.diff().shift(-1))]:
            m = pd.concat([ds, net], axis=1, keys=['ds', 'net']).dropna()
            if len(m) > 365:
                c = float(m.ds.corr(m.net))
                if best is None or c > best[1]:
                    best = (lab, c, m)
        if best is not None:
            lab, c, m = best
            mb = {'mb_days': len(m), 'mb_corr': c, 'storage_stamp': lab,
                  'mb_bias_hm3_per_day': float((m.net - m.ds).mean()),
                  'mb_rel_resid': float((m.net - m.ds).abs().mean() / max(1e-9, m.net.abs().mean()))}
        full = out[['storage_hm3', 'inflow_cms', 'release_cms']].dropna()
        rec = {'res_id': rid, 'grand_id': did,
               'start': str(out.dropna(how='all').index.min().date()), 'end': str(out.dropna(how='all').index.max().date()),
               'n_storage': int(out.storage_hm3.notna().sum()), 'n_inflow': int(out.inflow_cms.notna().sum()),
               'n_release': int(out.release_cms.notna().sum()), 'n_elev': int(out.elevation_m.notna().sum()),
               'n_evap': int(out.evaporation.notna().sum()), 'n_all3': len(full),
               'all3_start': str(full.index.min().date()) if len(full) else None,
               'all3_end': str(full.index.max().date()) if len(full) else None,
               'storage_max_obs_hm3': float(out.storage_hm3.max()) if out.storage_hm3.notna().any() else np.nan,
               'inflow_mean_cms': float(out.inflow_cms.mean()) if out.inflow_cms.notna().any() else np.nan,
               'cap_mcm_grand': cap, 'cap_mcm_resops': cap_r, **st, **mb}
        rows.append(rec)
    r = pd.DataFrame(rows)
    a = ra.rename(columns={'DAM_ID': 'grand_id'})[['grand_id', 'DAM_NAME', 'STATE', 'AGENCY_CODE', 'LONG', 'LAT', 'INCONSISTENCIES_NOTED']]
    gcols = ['GRAND_ID', 'RES_NAME', 'RIVER', 'YEAR', 'DAM_HGT_M', 'AREA_SKM', 'DEPTH_M', 'CATCH_SKM', 'DIS_AVG_LS', 'DOR_PC',
             'ELEV_MASL', 'MAIN_USE' if 'MAIN_USE' in gr else 'MAIN_ELEC', 'USE_IRRI', 'USE_ELEC', 'USE_SUPP', 'USE_FCON',
             'USE_RECR', 'USE_NAVI', 'MAIN_IRRI', 'MAIN_ELEC', 'MAIN_SUPP', 'MAIN_FCON', 'MAIN_RECR', 'MAIN_NAVI', 'QUALITY']
    gcols = list(dict.fromkeys(c for c in gcols if c in gr))
    r = r.merge(a, on='grand_id', how='left').merge(gr[gcols].rename(columns={'GRAND_ID': 'grand_id'}), on='grand_id', how='left')
    mains = ['MAIN_IRRI', 'MAIN_ELEC', 'MAIN_SUPP', 'MAIN_FCON', 'MAIN_RECR', 'MAIN_NAVI']
    r[mains] = r[mains].fillna(0)
    r['main_use'] = r[mains].idxmax(axis=1).str[5:].str.lower().where(r[mains].max(axis=1) > 0, 'other')
    r['cap_hm3'] = r.cap_mcm_grand.fillna(r.cap_mcm_resops)
    r['dor'] = r.cap_hm3 / (r.DIS_AVG_LS / 1000 * 31557600 / 1e6)     # capacity / mean annual flow volume (GRanD)
    r.columns = [c.lower() for c in r.columns]
    r.to_parquet(f'{OUT}/reservoirs.parquet')
    print(len(r), 'reservoirs;', (r.n_all3 > 3650).sum(), 'with >=10 y of storage+inflow+release')
    print(r.main_use.value_counts().to_dict())
    print(r[['mb_corr', 'mb_rel_resid']].describe().T)


if __name__ == '__main__':
    main()
