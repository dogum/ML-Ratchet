"""Other real datasets -> proc/other/ + catalogue.json (DATA.md section 5).

  hue   : HUE (Makonin 2019, Harvard Dataverse doi:10.7910/DVN/N3HGRN, CC BY 4.0) - 28 BC homes, hourly kWh, house
          metadata incl. heating/cooling system codes, YVR/WYJ airport weather. Times: local (America/Vancouver),
          'hour' 1..24 = hour ENDING -> converted to UTC hour-beginning.
  wpuq  : WPuQ (Schlemminger et al. 2022, Zenodo 10.5281/zenodo.5642902, CC BY 4.0) - 38 single-family houses with heat
          pumps in Lower Saxony, Germany: household and heat-pump active power (60-min files), local weather 2018-2020.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import os, re, json, glob
import numpy as np, pandas as pd

D = f'{BLDG_ROOT}/data'; R = f'{D}/raw/other'; O = f'{D}/proc/other'
os.makedirs(O, exist_ok=True)
HVAC_CODES = {'FAGF': 'forced-air gas furnace', 'FPG': 'gas fireplace', 'FPE': 'electric fireplace', 'HP': 'heat pump',
              'IFRHG': 'in-floor radiant heat, gas boiler', 'IFRHE': 'in-floor radiant heat, electric', 'BHE': 'baseboard electric heaters',
              'WRHIR': 'water radiator heat, oil? (IR)', 'GEOTH': 'geothermal (ground-source) heat pump', 'NAC': 'no air conditioning',
              'PAC': 'portable air conditioner', 'FAC': 'forced-air (central) air conditioning'}


def hue():
    txt = open(f'{R}/hue/All_Residential.txt').read().splitlines()
    rows = []
    for l in txt[2:]:
        if not l.strip() or l.startswith('*'): break
        p = l.split()
        if len(p) < 4: continue
        rows.append(dict(house=int(p[0]), first=p[1], last=p[2], cover=float(p[3]), raw=l))
    meta = pd.DataFrame(rows)
    # fixed-width columns after 'Cover'
    hdr = txt[0]
    cols = ['HouseType', 'Facing', 'Region', 'RUs', 'EVs', 'SN', 'HVAC']
    starts = [hdr.index(c) for c in cols]
    for c, a, b in zip(cols, starts, starts[1:] + [None]):
        meta[c] = meta.raw.str.slice(a, b).str.strip().replace('', None)
    meta = meta.drop(columns='raw')
    meta['heating_codes'] = meta.HVAC.fillna('').str.split(r'[,+]').apply(lambda L: [x.strip().split(' ')[-1] for x in L if x.strip()])
    def system(codes):
        s = set(codes)
        if 'GEOTH' in s: return 'heat_pump_ground'
        if 'HP' in s: return 'heat_pump_air'
        if 'FAGF' in s: return 'split_ac_furnace'          # contract name; has_ac says whether the AC half exists
        if 'IFRHG' in s or 'IFRHE' in s or 'WRHIR' in s: return 'radiant'
        if 'BHE' in s: return 'baseboard_resistance'
        return None
    meta['system'] = meta.heating_codes.apply(system)
    meta['electric_heat_main'] = meta.system.isin(['heat_pump_air', 'heat_pump_ground', 'baseboard_resistance']) | meta.HVAC.fillna('').str.match(r'^IFRHE')
    meta['has_ac'] = ~meta.HVAC.fillna('').str.contains('NAC') & meta.HVAC.notna()
    meta['heating_codes'] = meta.heating_codes.apply(','.join)
    meta['primary_use'] = np.where(meta.HouseType.isin(['apartment']), 'residential_multi', 'residential_single')
    meta.to_parquet(f'{O}/hue_houses.parquet', index=False)
    L = []
    for f in sorted(glob.glob(f'{R}/hue/Residential_*.csv')):
        h = int(re.findall(r'_(\d+)\.csv', f)[0])
        x = pd.read_csv(f)
        t = pd.to_datetime(x.date) + pd.to_timedelta(x.hour.astype(int) - 1, unit='h')       # hour-ending -> beginning
        t = t.dt.tz_localize('America/Vancouver', ambiguous='NaT', nonexistent='NaT')
        y = pd.DataFrame({'house': np.int16(h), 'time_utc': t.dt.tz_convert('UTC'), 'kWh': x.energy_kWh.astype('float32')}).dropna(subset=['time_utc'])
        L.append(y)
    m = pd.concat(L).sort_values(['house', 'time_utc']).drop_duplicates(['house', 'time_utc'])
    m.to_parquet(f'{O}/hue_meters.parquet', index=False)
    W = []
    for st in ['YVR', 'WYJ']:
        x = pd.read_csv(f'{R}/hue/Weather_{st}.csv')
        t = pd.to_datetime(x.date) + pd.to_timedelta(x.hour.astype(int) - 1, unit='h')
        t = t.dt.tz_localize('America/Vancouver', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
        W.append(pd.DataFrame({'station': st, 'time_utc': t, 'T_out': x.temperature.astype('float32'), 'RH': x.humidity.astype('float32'),
                               'pressure': (x.pressure * 1000).astype('float32'), 'weather': x.weather}).dropna(subset=['time_utc']))
    pd.concat(W).to_parquet(f'{O}/hue_weather.parquet', index=False)
    print('hue', meta.shape, m.shape, m.house.nunique(), meta.system.value_counts(dropna=False).to_dict())
    return dict(name='hue', file='hue_meters.parquet', kind='real residential smart meters (hourly) + weather',
                files=['hue_houses.parquet', 'hue_meters.parquet', 'hue_weather.parquet'], n_units=int(m.house.nunique()),
                period=f'{m.time_utc.min()} .. {m.time_utc.max()}', licence='CC BY 4.0',
                citation='S. Makonin, "HUE: The Hourly Usage of Energy Dataset for Buildings in British Columbia", Data in Brief 23 (2019) 103744; Harvard Dataverse doi:10.7910/DVN/N3HGRN',
                hvac_codes=HVAC_CODES,
                notes='Donated BC Hydro hourly data; electricity only (most homes heat with FortisBC gas, not metered here). '
                      'House 7 has no metadata. Weather: Environment Canada airport stations YVR (Vancouver) and WYJ (Victoria). '
                      'Gradable for BuildingDetective (electric vs gas heat, heat pump) and LoadCast residential.')


def _t(ix):
    ix = np.asarray(ix, 'int64'); unit = 'ns' if np.nanmax(np.abs(ix)) > 1e15 else 's'
    return pd.to_datetime(ix, unit=unit, utc=True)


def _h5_tables(path):
    import h5py
    out = {}
    with h5py.File(path, 'r') as f:
        def vis(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith('/table'):
                out[name[:-6]] = obj[()]
        f.visititems(vis)
    return out


def wpuq():
    files = sorted(glob.glob(f'{R}/wpuq/*_data_60min.hdf5'))
    if not files: return None
    L = []
    for f in files:
        yr = int(os.path.basename(f)[:4])
        for name, arr in _h5_tables(f).items():
            parts = name.split('/')
            fields = [c for c in arr.dtype.names if c != 'index']
            keep = [c for c in fields if c.startswith(('P_TOT', 'P_1', 'P_2', 'P_3', 'Q_TOT', 'S_TOT', 'PF_TOT'))] or fields
            t = _t(arr['index'])
            d = pd.DataFrame({c: arr[c].astype('float32') for c in keep}); d.insert(0, 'time_utc', t)
            d.insert(0, 'path', name); d.insert(1, 'year_file', yr)
            L.append(d)
    x = pd.concat(L, ignore_index=True)
    x['path'] = x.path.astype('category')
    x.to_parquet(f'{O}/wpuq_60min.parquet', index=False, compression='zstd')
    W = []
    for f in sorted(glob.glob(f'{R}/wpuq/*_weather.hdf5')):
        for name, arr in _h5_tables(f).items():
            c = [n for n in arr.dtype.names if n != 'index'][0]
            W.append(pd.DataFrame({'time_utc': _t(arr['index']), 'var': c, 'value': arr[c].astype('float32'), 'source': name}))
    w = pd.concat(W, ignore_index=True)
    wh = w.assign(time_utc=w.time_utc.dt.floor('h')).pivot_table(index='time_utc', columns='var', values='value', aggfunc='mean')
    ren = {'TEMPERATURE:TOTAL': 'T_out', 'RELATIVE_HUMIDITY:TOTAL': 'RH', 'ATMOSPHERIC_PRESSURE:TOTAL': 'pressure_raw', 'SOLAR_IRRADIANCE:GLOBAL': 'GHI',
           'WIND_SPEED:TOTAL': 'wind_speed', 'WIND_DIRECTION:TOTAL': 'wind_dir', 'PRECIPITATION_RATE:TOTAL': 'precip_rate', 'APPARENT_TEMPERATURE:TOTAL': 'T_apparent'}
    wh = wh.rename(columns=ren); wh.columns.name = None
    wh.astype('float32').to_parquet(f'{O}/wpuq_weather_hourly.parquet')
    print('wpuq', x.shape, x.path.nunique(), 'series; weather', wh.shape)
    return dict(name='wpuq', file='wpuq_60min.parquet', kind='real heat-pump households (household + heat-pump electric power, hourly) + local weather',
                files=['wpuq_60min.parquet', 'wpuq_weather_hourly.parquet'], n_series=int(x.path.nunique()),
                period=f'{x.time_utc.min()} .. {x.time_utc.max()}', licence='CC BY 4.0',
                citation='M. Schlemminger, T. Ohrdes, E. Schneider, M. Knoop, "Dataset on electrical single-family house and heat pump load profiles in Germany", Scientific Data 9, 56 (2022); Zenodo doi:10.5281/zenodo.5642902',
                notes='Series `path` names follow the HDF5 groups (e.g. NO_PV/SFH<k>/HEATPUMP or HOUSEHOLD). Power in W (P_TOT = active power, '
                      'Q = reactive). 10-s and 1-min files were not downloaded (2-9 GB each). Heat output is not metered: heat pump ELECTRIC '
                      'demand only. Weather from a local weather service, irregular -> hourly means.')


GATED = {
    'ecobee_donate_your_data': 'smart-thermostat logs (runtime, setpoints, indoor/outdoor T) of ~100k homes; needs a research application to ecobee (gated).',
    'pecan_street_dataport': 'circuit-level residential data incl. HVAC, EV, PV (Austin); paid / university licence (gated).',
    'uk_rhpp_heat_pump_trial': 'Renewable Heat Premium Payment heat-pump field trial (~700 heat pumps, 2-min data); UK Data Service SN 8151, registration + safeguarded licence (gated).',
    'uk_electrification_of_heat': 'Electrification of Heat demonstration (742 heat pumps, 2020-23); UK Data Service SN 9050, registration (gated).',
    'irish_cer_smart_meter': 'CER smart metering trial (5,000 homes/SMEs, 30 min); Irish Social Science Data Archive, application (gated).',
    'ideal_household_dataset': 'IDEAL (Edinburgh DataShare, CC BY 4.0) 255 UK homes with room temperatures, gas and electricity: open but very large (sensor zip >10 GB); not taken this phase.',
    'low_carbon_london': 'UKPN Low Carbon London smart meters (5,567 homes, 30 min, 2011-14), London Datastore: open but ~10 GB CSV and no per-home weather; not taken this phase.',
    'lbnl_fdd_full_sets': 'faultdetection.lbl.gov full LBNL FDD sets (RTU lab/field, SD-AHU, chiller, boiler, FCU, VAV sims): downloads sit behind a name/email form; OEDI submission 5763 page shows no data to this client. The CC0 figshare synthesis subset was used instead.',
}


def main():
    cat = {}
    for fn in (hue, wpuq):
        r = fn()
        if r: cat[r['name']] = r
    cat['_gated_or_not_taken'] = GATED
    json.dump(cat, open(f'{O}/catalogue.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
