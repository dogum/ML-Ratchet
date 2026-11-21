"""WaterData: real watershed data for the Watershed project (ML Ratchet, Phase 1 agent B).

pandas / numpy / pyarrow only. All daily frames use a date index (days; see DATA.md 'time' for what a day means).

    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import waterdata as wd
    sp = wd.splits()                                   # frozen splits (data/proc/splits.json)
    ids = wd.basins('camels', split='dev')             # basin ids ('camels_01022500', ...)
    a = wd.attributes('camels', split='dev')           # Caravan + HydroATLAS + NWIS attributes (one row per basin)
    s = wd.spec('camels_01022500')                     # contract basin spec (unknowns None + defaults_source)
    f = wd.forcing('camels_01022500', '1989-10-01', '1999-09-30')   # P,T_mean,T_min,T_max,SRAD,vp,wind,PET,(+extras)
    m = wd.forcing_meta('camels_01022500')
    q = wd.streamflow('camels_01022500', source='best')  # Q_mm, Q_cms, source, flag
    n = wd.nwm('camels_01022500')                       # NWM retrospective v3.0 daily mean m3/s (+ mm/day)
    pk = wd.peaks('camels_01022500')                   # USGS annual peaks
    st = wd.snotel_meta(); sw = wd.snotel('1000_CO_SNTL')
    rv = wd.reservoir_meta(); r = wd.reservoir('<id>')
    from waterdata import metrics                      # nse, kge, fhv, flv, peak errors, timing
    bl = wd.baselines()                                # exam baseline summary
"""
import json, os, functools
import numpy as np, pandas as pd

ROOT = os.path.join(os.environ.get('ML_RATCHET_WATER') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data').replace(os.sep, '/')
PROC = f'{ROOT}/proc'
REGIONS = ['camels', 'camelsgb', 'camelsaus', 'camelscl', 'camelsbr']
CONTRACT_FORCING = ['P', 'T_mean', 'T_min', 'T_max', 'SRAD', 'vp', 'wind', 'PET']
CFS_TO_CMS = 0.028316846592


def region_of(basin_id):
    return basin_id.split('_')[0]


@functools.lru_cache(None)
def splits():
    return json.load(open(f'{PROC}/splits.json'))


def _split_ids(region, split):
    s = splits()
    if region == 'camels':
        cu = s['camels_us']
        return {'dev': cu['dev'], 'exam': cu['exam'], 'exam_531': cu['exam_531']}[split]
    return s['other_regions'][region][split]


@functools.lru_cache(None)
def _attrs(region):
    return pd.read_parquet(f'{PROC}/attributes/{region}.parquet').set_index('gauge_id')


def attributes(region='camels', split=None):
    a = _attrs(region).copy()
    if split is not None:
        a = a.loc[_split_ids(region, split)]
    return a


def basins(region='camels', split=None):
    return list(attributes(region, split).index)


@functools.lru_cache(None)
def _specs(region):
    return json.load(open(f'{PROC}/specs/{region}_specs.json'))


def spec(basin_id, approx_bands=False):
    s = dict(_specs(region_of(basin_id))[basin_id])
    if approx_bands and s.get('elev_bands') is None:
        s['elev_bands'] = approx_elev_bands(s)
        s['elev_bands_source'] = 'approx_elev_bands (triangular hypsometry from HydroATLAS min/mean/max)'
    return s


def specs(region='camels', split=None):
    ids = basins(region, split)
    return [spec(i) for i in ids]


def approx_elev_bands(s, n=5):
    """Approximate elevation bands from min/mean/max with a triangular hypsometry (mode chosen so the mean matches).
    Returns {'frac': [...], 'elev_m': [...]} with n equal-height bands. A crude stand-in for a DEM."""
    lo, mu, hi = s.get('elev_min_m'), s.get('elev_mean_m'), s.get('elev_max_m')
    if lo is None or hi is None or mu is None or hi - lo < 1:
        return {'frac': [1.0], 'elev_m': [mu]}
    c = np.clip(3 * mu - lo - hi, lo, hi)                     # mode of the triangular pdf
    edges = np.linspace(lo, hi, n + 1)

    def cdf(x):
        x = np.clip(x, lo, hi)
        return np.where(x <= c, (x - lo) ** 2 / ((hi - lo) * max(c - lo, 1e-9)),
                        1 - (hi - x) ** 2 / ((hi - lo) * max(hi - c, 1e-9)))
    fr = np.diff(cdf(edges)); fr = fr / fr.sum()
    return {'frac': [float(x) for x in fr], 'elev_m': [float(x) for x in 0.5 * (edges[1:] + edges[:-1])]}


def forcing(basin_id, start=None, end=None, extras=True):
    """Basin-average daily forcing (ERA5-Land via Caravan v1.6). Contract columns first:
    P mm/d, T_mean/T_min/T_max degC, SRAD W/m2 (NET shortwave), vp Pa, wind m/s, PET mm/d (FAO-56 Penman-Monteith).
    Extras: LRAD_net W/m2, pressure kPa, SWE_era5 mm (model), sm1..sm4 m3/m3 (model)."""
    r = region_of(basin_id)
    df = pd.read_parquet(f'{PROC}/caravan/{r}/{basin_id}.parquet')
    df = df.drop(columns=['Q_mm'])
    df = df[CONTRACT_FORCING + [c for c in df.columns if c not in CONTRACT_FORCING]]
    if not extras:
        df = df[CONTRACT_FORCING]
    return df.loc[start:end]


def forcing_meta(basin_id):
    s = spec(basin_id)
    return {'basin_id': basin_id, 'source': 'Caravan v1.6 (ERA5-Land basin averages; doi:10.5281/zenodo.15529786)',
            'area_km2': s['area_km2'], 'area_polygon_km2': s.get('area_polygon_km2'),
            'lat': s['lat'], 'lon': s['lon'], 'latlon_is': s.get('latlon_is'),
            'elev_mean_m': s['elev_mean_m'], 'elev_min_m': s['elev_min_m'], 'elev_max_m': s['elev_max_m'],
            'elev_bands': s['elev_bands'], 'PET_method': 'FAO-56 Penman-Monteith (Caravan potential_evaporation_sum_FAO_PENMAN_MONTEITH)',
            'SRAD_note': 'ERA5-Land surface NET solar radiation (incoming * (1 - albedo)), daily mean W/m2',
            'vp_note': 'from daily mean 2 m dewpoint, Magnus formula (Alduchov & Eskridge 1996)',
            'wind_note': 'magnitude of the daily mean 10 m wind vector (a lower bound of mean speed)',
            'day': 'UTC calendar day (ERA5-Land hourly aggregated by Caravan in UTC)'}


def _q_from_mm(q_mm, basin_id):
    return q_mm * spec(basin_id)['area_km2'] / 86.4


def streamflow(basin_id, source='best', start=None, end=None):
    """Daily streamflow. Columns Q_mm (mm/day over spec area_km2), Q_cms (m3/s), src, flag.
    source='caravan': CAMELS record in Caravan (1980/81-2014 for US; region records elsewhere);
    'nwis': USGS daily values 1980-2026 (CAMELS-US only; flags: A approved, P provisional, e estimated, Ice, ...);
    'best' (US): NWIS where available, else Caravan."""
    r = region_of(basin_id)
    if source == 'caravan' or r != 'camels':
        q = pd.read_parquet(f'{PROC}/caravan/{r}/{basin_id}.parquet', columns=['Q_mm']).Q_mm
        out = pd.DataFrame({'Q_mm': q, 'Q_cms': _q_from_mm(q, basin_id), 'src': 'caravan', 'flag': ''})
        return out.loc[start:end]
    p = f'{PROC}/nwis/daily/{basin_id}.parquet'
    nw = pd.read_parquet(p) if os.path.exists(p) else None
    if source == 'nwis':
        if nw is None:
            raise FileNotFoundError(p)
        return nw.loc[start:end]
    cv = streamflow(basin_id, 'caravan')
    if nw is None:
        return cv.loc[start:end]
    out = nw.combine_first(cv)
    return out.loc[start:end]


def nwm(basin_id, start=None, end=None, day='local'):
    """NOAA NWM retrospective v3.0 daily mean streamflow at the gauge reach (m3/s and mm/day).
    day='local': mean of hourly values over the local-standard-time day of the gauge (matches USGS daily values);
    day='utc': UTC calendar day."""
    p = f'{PROC}/nwm/daily_{day}.parquet'
    g = basin_id.split('_')[1]
    d = pd.read_parquet(p, columns=[g])[g]
    out = pd.DataFrame({'Q_cms': d, 'Q_mm': d * 86.4 / spec(basin_id)['area_km2']})
    return out.loc[start:end]


def nwm_hourly(basin_id, start=None, end=None):
    """NWM v3.0 retrospective HOURLY streamflow (m3/s, UTC valid time) at the gauge reach, kept for 2015-10..2023-01 only
    (older months were aggregated to daily and deleted to save disk; scripts/fetch_nwm.py re-fetches them)."""
    import glob
    g = basin_id.split('_')[1]
    fs = sorted(glob.glob(f'{ROOT}/raw/nwm/chanobs_hourly/*.parquet'))
    d = pd.concat([pd.read_parquet(f, columns=[g])[g] for f in fs])
    return d.loc[start:end]


def peaks(basin_id):
    return pd.read_parquet(f'{PROC}/nwis/peaks.parquet').query('gauge_id == @basin_id')


def snotel_meta(split=None):
    m = pd.read_parquet(f'{PROC}/snotel/stations.parquet')
    if split is not None:
        ids = splits()['snotel'][split]
        m = m[m.station_id.isin(ids)]
    return m


def snotel(station_id, start=None, end=None):
    """Daily SNOTEL: SWE mm (start of day), P_accum mm, P mm/day (from accumulation), T_mean/T_min/T_max degC; local day."""
    return pd.read_parquet(f'{PROC}/snotel/daily/{station_id}.parquet').loc[start:end]


def snotel_near(basin_id, max_km=50.0):
    """SNOTEL stations near a CAMELS-US basin: same HUC8 as the gauge or within max_km of the gauge, sorted by distance.
    Columns add dist_km, same_huc8, d_elev_m (station minus basin mean elevation). A proximity rule, not point-in-polygon."""
    m = snotel_meta().copy()
    a = attributes('camels').loc[basin_id]
    la1, lo1 = np.radians(a.gauge_lat), np.radians(a.gauge_lon)
    la2, lo2 = np.radians(m.lat.values), np.radians(m.lon.values)
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    m['dist_km'] = 2 * 6371.0 * np.arcsin(np.sqrt(h))
    m['same_huc8'] = m.huc12.astype(str).str[:8] == str(a.huc8)
    m['d_elev_m'] = m.elev_m - a.ele_mt_sav
    return m[(m.dist_km <= max_km) | m.same_huc8].sort_values('dist_km')


def reservoir_cdec(cdec_id, start=None, end=None):
    """CDEC daily storage_hm3 / inflow_cms / release_cms 1990-2025 for 18 California reservoirs (crosswalk to ResOpsUS
    res_id in data/proc/reservoirs/cdec/crosswalk.csv). Local (Pacific) calendar day."""
    return pd.read_parquet(f'{PROC}/reservoirs/cdec/{cdec_id}.parquet').loc[start:end]


def reservoir_meta(split=None):
    m = pd.read_parquet(f'{PROC}/reservoirs/reservoirs.parquet')
    if split is not None:
        ids = splits()['reservoirs'][split]
        m = m[m.res_id.isin(ids)]
    return m


def reservoir(res_id, start=None, end=None):
    """Daily reservoir operations: storage_hm3, inflow_cms, release_cms (+ elevation_m, evap where published)."""
    return pd.read_parquet(f'{PROC}/reservoirs/daily/{res_id}.parquet').loc[start:end]


def baselines(name=None):
    if name is None:
        return json.load(open(f'{PROC}/baselines/summary.json'))
    return pd.read_parquet(f'{PROC}/baselines/{name}.parquet')
