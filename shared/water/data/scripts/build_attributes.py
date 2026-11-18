"""Merge Caravan attributes (+ NWIS site info for CAMELS-US) and map them to the shared basin spec.

Out: proc/attributes/<region>.parquet (all published columns + derived), proc/specs/<region>_specs.json
Region keys: camels (US), camelsgb, camelsaus, camelscl, camelsbr.
"""
import json, os, io
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
RAWA = f'{ROOT}/raw/caravan/attributes'
REGIONS = ['camels', 'camelsgb', 'camelsaus', 'camelscl', 'camelsbr']
SEC_PER_YR = 365.25 * 86400


def nwis_sites():
    txt = open(f'{ROOT}/raw/nwis/site_expanded.rdb').read().splitlines()
    rows, hdr = [], None
    for ln in txt:
        if ln.startswith('#') or not ln.strip():
            continue
        p = ln.split('\t')
        if p[0] == 'agency_cd':
            hdr = p; continue
        if p[0] == 'USGS' and hdr:
            rows.append(dict(zip(hdr, p)))
    d = pd.DataFrame(rows).drop_duplicates('site_no')
    out = pd.DataFrame({'gauge_id': 'camels_' + d.site_no,
                        'nwis_name': d.station_nm, 'nwis_lat': pd.to_numeric(d.dec_lat_va, errors='coerce'),
                        'nwis_lon': pd.to_numeric(d.dec_long_va, errors='coerce'),
                        'nwis_drain_area_km2': pd.to_numeric(d.drain_area_va, errors='coerce') * 2.58998811,
                        'nwis_contrib_area_km2': pd.to_numeric(d.contrib_drain_area_va, errors='coerce') * 2.58998811,
                        'nwis_gauge_alt_m': pd.to_numeric(d.alt_va, errors='coerce') * 0.3048,
                        'huc8': d.huc_cd.str[:8], 'huc2': d.huc_cd.str[:2], 'state_cd': d.state_cd, 'nwis_tz': d.tz_cd})
    return out


def load(region):
    a = None
    for kind in ['other', 'caravan', 'hydroatlas']:
        x = pd.read_csv(f'{RAWA}/{region}/attributes_{kind}_{region}.csv')
        a = x if a is None else a.merge(x, on='gauge_id', how='outer')
    a = a.copy()
    a['region'] = region
    if region == 'camels':
        a = a.merge(nwis_sites(), on='gauge_id', how='left').copy()
        miss = a.huc2.isna() | (a.huc2 == '')
        a.loc[miss, 'huc2'] = a.loc[miss, 'gauge_id'].str[7:9]   # USGS station-number prefix = HUC2 region (fallback)
        b531 = set(open(f'{ROOT}/raw/lists/531_basin_list.txt').read().split())
        a['in_531'] = a.gauge_id.str[7:].isin(b531)
        ia = f'{ROOT}/proc/attributes/camels_implied_area.csv'
        if os.path.exists(ia):                        # written by derive_area.py (needs NWIS); see its docstring
            a = a.merge(pd.read_csv(ia)[['gauge_id', 'implied_area_km2']], on='gauge_id', how='left')
            a['camels_area_km2'] = a.implied_area_km2.fillna(a.nwis_contrib_area_km2).fillna(a.nwis_drain_area_km2)
        a['area_discrepancy'] = (a['area'] - a['nwis_drain_area_km2']) / a['nwis_drain_area_km2']
    import warnings; warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
    # derived (documented in DATA.md)
    S, C, Si = a.snd_pc_sav, a.cly_pc_sav, a.slt_pc_sav
    a['porosity_cosby'] = (50.5 - 0.142 * S - 0.037 * C) / 100.0                    # Cosby et al. 1984, table 4
    a['ksat_cosby_mm_h'] = 25.4 * 10 ** (-0.60 + 0.0126 * S - 0.0064 * C)            # in/h -> mm/h
    a['b_cosby'] = 3.10 + 0.157 * C - 0.003 * S
    a['slope_deg'] = a.slp_dg_sav / 10.0                                              # HydroATLAS: degrees x10
    a['tmp_mean_c'] = a.tmp_dc_syr / 10.0                                             # degrees C x10
    flow_hm3 = a.dis_m3_pyr * SEC_PER_YR / 1e6                                        # WaterGAP natural discharge
    a['dor_frac'] = np.where(flow_hm3 > 0, a.rev_mc_usu / flow_hm3, np.nan)           # upstream reservoir vol / annual flow
    a['reg_class'] = pd.cut(a['dor_frac'].fillna(0), [-1, 0.0001, 0.1, 0.5, 1e9], labels=['none', 'low', 'mid', 'high']).astype(str)
    return a


def spec(r):
    def f(x, k=1.0):
        return None if pd.isna(x) else float(x) * k
    emin, emean, emax = r.ele_mt_smn, r.ele_mt_sav, r.ele_mt_smx
    return {
        'id': r.gauge_id, 'region': r.region, 'name': r.gauge_name, 'country': r.country,
        'area_km2': f(getattr(r, 'camels_area_km2', r.area)) if not pd.isna(getattr(r, 'camels_area_km2', np.nan)) else f(r.area),
        'area_polygon_km2': f(r.area),
        'area_note': ('CAMELS-US: the area CAMELS/Caravan used to convert discharge to mm/day, recovered from the data '
                      '(= NWIS contributing drainage area where published, else NWIS drainage area; derive_area.py). '
                      'Forcing is averaged over the Caravan polygon (area_polygon_km2), which can differ: see '
                      'attributes column area_discrepancy') if r.region == 'camels' else 'Caravan polygon area',
        'lat': f(r.gauge_lat), 'lon': f(r.gauge_lon), 'latlon_is': 'gauge location (not centroid)',
        'elev_mean_m': f(emean), 'elev_min_m': f(emin), 'elev_max_m': f(emax),
        'elev_bands': None,
        'slope_mean': f(r.slp_dg_sav, 0.1), 'slope_unit': 'degrees',
        'aridity': f(r.aridity_FAO_PM), 'p_mean': f(r.p_mean), 'pet_mean': f(r.pet_mean_FAO_PM), 'frac_snow': f(r.frac_snow),
        'forest_frac': f(r.for_pc_sse, 0.01), 'lai': None,
        'soil_depth_m': None, 'soil_porosity': f(r.porosity_cosby), 'soil_conductivity': f(r.ksat_cosby_mm_h),
        'soil_conductivity_unit': 'mm/h', 'max_water_content_mm': None,
        'sand_frac': f(r.snd_pc_sav, 0.01), 'clay_frac': f(r.cly_pc_sav, 0.01), 'silt_frac': f(r.slt_pc_sav, 0.01),
        'geology_permeability': None, 'carbonate_frac': f(r.kar_pc_sse, 0.01), 'lithology_class_glim': f(r.lit_cl_smj),
        'groundwater_table_depth_m': f(r.gwt_cm_sav, 0.01),
        'dam_count': None, 'reservoir_storage_frac': f(r.dor_frac), 'irrigation_frac': f(r.ire_pc_sse, 0.01),
        'urban_frac': f(r.urb_pc_sse, 0.01), 'glacier_frac': f(r.gla_pc_sse, 0.01),
        'params': None,
        'defaults_source': {
            'elev_bands': 'unknown (no DEM in Caravan); approximate with waterdata.approx_elev_bands(spec) (triangular hypsometry from min/mean/max)',
            'lai': 'unknown; use simulator default (e.g. 1 + 3*forest_frac seasonal max)',
            'soil_depth_m': 'unknown; simulator default 1.5 m',
            'max_water_content_mm': 'unknown; soil_depth_m * soil_porosity * 1000 * 0.5 (plant-available fraction) is a common prior',
            'geology_permeability': 'unknown (GLHYMPS not in Caravan); lithology_class_glim and carbonate_frac (HydroATLAS karst extent) are proxies',
            'dam_count': 'unknown; reservoir_storage_frac (HydroATLAS upstream reservoir volume / WaterGAP natural annual flow) is given',
            'soil_porosity': 'Cosby et al. (1984) pedotransfer from SoilGrids sand/clay in HydroATLAS',
            'soil_conductivity': 'Cosby et al. (1984) pedotransfer, mm/h',
        },
    }


def main():
    os.makedirs(f'{ROOT}/proc/attributes', exist_ok=True); os.makedirs(f'{ROOT}/proc/specs', exist_ok=True)
    for r in REGIONS:
        a = load(r)
        a.to_parquet(f'{ROOT}/proc/attributes/{r}.parquet')
        specs = {row.gauge_id: spec(row) for row in a.itertuples()}
        json.dump(specs, open(f'{ROOT}/proc/specs/{r}_specs.json', 'w'), indent=0)
        print(r, len(a), a.shape[1])
        if r == 'camels':
            print(a[['area', 'nwis_drain_area_km2', 'area_discrepancy', 'dor_frac']].describe().T)
            print(a.reg_class.value_counts(), a.huc2.value_counts().sort_index().to_dict(), a.in_531.sum())


if __name__ == '__main__':
    main()
