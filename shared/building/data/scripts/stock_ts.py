"""Stratified sample of ResStock 2025 R1 / ComStock 2025 R3 baseline buildings: hourly end-use profiles + county weather.

MODELLED DATA (EnergyPlus via OpenStudio), not measurements. For generator priors and the simulator-vs-simulator check.

  python3 -I stock_ts.py sample            # draw the sample (seed 20261116) -> proc/stock/<name>_sample.parquet
  python3 -I stock_ts.py fetch resstock    # hourly time series -> proc/stock/<name>_hourly.parquet (resumable parts)
  python3 -I stock_ts.py weather           # county AMY2018 weather of sampled buildings -> proc/stock/weather/<county>.parquet

Time: NREL files are 15-min, period-ENDING, local standard time (first stamp 00:15). Hourly = sum (energy) / mean (states)
over (h, h+1], labelled h (hour-beginning), column time_lst. std_utc_offset_h per building from its state (approximate).
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import sys, os, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import s3http as s

D = f'{BLDG_ROOT}/data'; P = f'{D}/proc/stock'; T = f'{D}/tmp/stock_ts'
os.makedirs(T, exist_ok=True); os.makedirs(f'{P}/weather', exist_ok=True)
SEED = 20261116
REL = {'resstock': 'resstock_amy2018_release_1', 'comstock': 'comstock_amy2018_release_3'}
N_TARGET = {'resstock': 1500, 'comstock': 1500}
RES_COLS = ['out.electricity.total.energy_consumption..kwh', 'out.natural_gas.total.energy_consumption..kwh',
            'out.fuel_oil.total.energy_consumption..kwh', 'out.propane.total.energy_consumption..kwh',
            'out.site_energy.total.energy_consumption..kwh', 'out.electricity.net.energy_consumption..kwh',
            'out.electricity.heating.energy_consumption..kwh', 'out.electricity.heating_hp_bkup.energy_consumption..kwh',
            'out.electricity.heating_fans_pumps.energy_consumption..kwh', 'out.electricity.cooling.energy_consumption..kwh',
            'out.electricity.cooling_fans_pumps.energy_consumption..kwh', 'out.electricity.hot_water.energy_consumption..kwh',
            'out.electricity.pv.energy_consumption..kwh', 'out.natural_gas.heating.energy_consumption..kwh',
            'out.natural_gas.hot_water.energy_consumption..kwh', 'out.load.heating.energy_delivered..kbtu',
            'out.load.cooling.energy_delivered..kbtu', 'out.indoor_temperature.conditioned_space..c',
            'out.indoor_relative_humidity.conditioned_space..percentage', 'out.schedules.heating_setpoint..c',
            'out.schedules.cooling_setpoint..c', 'out.outdoor_air_drybulb_temp..c', 'out.schedules.occupants',
            'out.unmet_hours.heating..hour', 'out.unmet_hours.cooling..hour']
MEAN_PREFIX = ('out.indoor_', 'out.schedules.', 'out.outdoor_')
STATE_TZ = {**{k: -5 for k in 'CT DE DC FL GA IN KY ME MD MA MI NH NJ NY NC OH PA RI SC VT VA WV'.split()},
            **{k: -6 for k in 'AL AR IL IA KS LA MN MS MO NE ND OK SD TN TX WI'.split()},
            **{k: -7 for k in 'AZ CO ID MT NM UT WY'.split()}, **{k: -8 for k in 'CA NV OR WA'.split()}, 'AK': -9, 'HI': -10}


def short(c):
    c = c.replace('out.', '').replace('.energy_consumption..kwh', '_kWh').replace('.energy_consumption', '_kWh')
    c = c.replace('.energy_delivered..kbtu', '_delivered_kBtu').replace('..c', '_C').replace('..percentage', '_pct').replace('..hour', '_h')
    return c.replace('.', '_')


def sample():
    rng = np.random.default_rng(SEED)
    r = pd.read_parquet(f'{P}/resstock_meta.parquet', columns=['bldg_id', 'weight', 'completed_status', 'in.state', 'in.county',
                        'in.geometry_building_type_recs', 'in.ashrae_iecc_climate_zone_2004', 'in.heating_fuel', 'in.hvac_heating_type'])
    r = r[r.completed_status.astype(str) == 'Success'].copy()
    r['stratum'] = r['in.geometry_building_type_recs'].astype(str) + '|' + r['in.ashrae_iecc_climate_zone_2004'].astype(str)
    c = pd.read_parquet(f'{P}/comstock_meta.parquet', columns=['bldg_id', 'weight', 'completed_status', 'in.state', 'in.as_simulated_nhgis_county_gisjoin',
                        'in.comstock_building_type', 'in.as_simulated_ashrae_iecc_climate_zone_2006', 'in.hvac_system_type'])
    c = c[c.completed_status.astype(str) == 'Success'].copy()
    c['stratum'] = c['in.comstock_building_type'].astype(str) + '|' + c['in.as_simulated_ashrae_iecc_climate_zone_2006'].astype(str)
    for name, df, cty in [('resstock', r, 'in.county'), ('comstock', c, 'in.as_simulated_nhgis_county_gisjoin')]:
        w = df.groupby('stratum').weight.sum(); share = w / w.sum()
        n = np.clip(np.round(share * N_TARGET[name]), 4, 60).astype(int)
        picks = []
        for st, k in n.items():
            g = df[df.stratum == st]; k = min(k, len(g))
            p = g.weight.to_numpy(float); p = p / p.sum()
            picks.append(g.iloc[rng.choice(len(g), k, replace=False, p=p)])
        out = pd.concat(picks)[['bldg_id', 'in.state', cty, 'stratum', 'weight']].rename(columns={cty: 'county', 'in.state': 'state'})
        out['state'] = out.state.astype(str); out['county'] = out.county.astype(str)
        # ResStock AK/HI rows carry the county NAME in in.county; the GISJOIN is the first part of in.county_and_puma (fixed in place on 16 Nov)
        out['std_utc_offset_h'] = out.state.map(STATE_TZ)
        out = out.sort_values('bldg_id').reset_index(drop=True)
        out.to_parquet(f'{P}/{name}_sample.parquet', index=False)
        print(name, len(out), 'buildings in', out.stratum.nunique(), 'strata;', out.county.nunique(), 'counties')


def fetch(name):
    smp = pd.read_parquet(f'{P}/{name}_sample.parquet')
    B = 50
    for i in range(0, len(smp), B):
        part = f'{T}/{name}_{i:05d}.parquet'
        if os.path.exists(part): continue
        frames = []
        for _, b in smp.iloc[i:i + B].iterrows():
            key = f'{s.BS}/{REL[name]}/timeseries_individual_buildings/by_state/upgrade=0/state={b.state}/{b.bldg_id}-0.parquet'
            for a in range(4):
                try:
                    pf = pq.ParquetFile(s.fs().open(s.url(key)))
                    cols = RES_COLS if name == 'resstock' else [c for c in pf.schema_arrow.names if c.startswith('out.')]
                    cols = [c for c in cols if c in pf.schema_arrow.names]
                    t = pf.read(columns=['timestamp'] + cols).to_pandas(); break
                except Exception as e:
                    print('retry', key, e, flush=True); time.sleep(5 * (a + 1)); t = None
            if t is None: continue
            t = t.set_index(pd.to_datetime(t.timestamp)).drop(columns='timestamp')
            mean_c = [c for c in t.columns if c.startswith(MEAN_PREFIX)]
            sum_c = [c for c in t.columns if c not in mean_c]
            h = pd.concat([t[sum_c].resample('h', closed='right', label='left').sum(min_count=1),
                           t[mean_c].resample('h', closed='right', label='left').mean()], axis=1)
            h = h[h.index.year == 2018]
            h.columns = [short(c) for c in h.columns]
            h = h.astype('float32'); h.index.name = 'time_lst'; h = h.reset_index(); h.insert(0, 'bldg_id', np.int32(b.bldg_id))
            frames.append(h)
        pd.concat(frames).to_parquet(part + '.tmp', index=False, compression='zstd'); os.replace(part + '.tmp', part)
        print(name, i, 'done', flush=True)
    merge(name)


def merge(name):
    """Streaming merge of the part files (one part in memory at a time; columns unified across parts)."""
    parts = sorted(glob.glob(f'{T}/{name}_*.parquet'))
    cols = []
    for p in parts:
        for c in pq.ParquetFile(p).schema_arrow.names:
            if c not in cols: cols.append(c)
    fields = [pa.field('bldg_id', pa.int32()), pa.field('time_lst', pa.timestamp('ms'))] + \
             [pa.field(c, pa.float32()) for c in cols if c not in ('bldg_id', 'time_lst')]
    schema = pa.schema(fields)
    out = f'{P}/{name}_hourly.parquet'
    w = pq.ParquetWriter(out + '.tmp', schema, compression='zstd', compression_level=9)
    n = 0
    for p in parts:
        t = pq.read_table(p)
        arrs = []
        for f in schema:
            if f.name not in t.column_names:
                arrs.append(pa.nulls(t.num_rows, f.type)); continue
            a = t[f.name].cast(f.type)
            if f.type == pa.float32():           # keep 12 mantissa bits: relative error <= 2.4e-4, compresses ~3x better
                v = a.to_numpy(zero_copy_only=False).astype(np.float32)
                v = (v.view(np.uint32) & np.uint32(0xFFFFF800)).view(np.float32)
                a = pa.array(v, type=pa.float32(), from_pandas=True)
            arrs.append(a)
        w.write_table(pa.Table.from_arrays(arrs, schema=schema), row_group_size=8760 * 25); n += t.num_rows
        del t, arrs
    w.close(); os.replace(out + '.tmp', out)
    print('wrote', name, n, os.path.getsize(out) / 1e6, 'MB', flush=True)


def weather():
    cnt = set()
    for name in REL:
        smp = pd.read_parquet(f'{P}/{name}_sample.parquet')
        for _, b in smp[['state', 'county']].drop_duplicates().iterrows(): cnt.add((name, b.state, b.county))
    done = set()
    for name, st, cy in sorted(cnt):
        if cy in done: continue
        f = f'{P}/weather/{cy}.parquet'
        if os.path.exists(f): done.add(cy); continue
        key = (f'{s.BS}/{REL[name]}/weather/state={st}/{cy}_2018.csv' if name == 'resstock'
               else f'{s.BS}/{REL[name]}/weather/amy2018/{cy}_2018.csv')
        for a in range(4):
            try:
                w = pd.read_csv(s.url(key), parse_dates=['date_time']); break
            except Exception as e:
                print('retry', key, e, flush=True); w = None
                if '404' in str(e): break                     # file not published for this county
                time.sleep(5 * (a + 1))
        if w is None: continue
        w = w.rename(columns={'Dry Bulb Temperature [°C]': 'T_out', 'Relative Humidity [%]': 'RH', 'Wind Speed [m/s]': 'wind_speed',
                              'Wind Direction [Deg]': 'wind_dir', 'Global Horizontal Radiation [W/m2]': 'GHI',
                              'Direct Normal Radiation [W/m2]': 'DNI', 'Diffuse Horizontal Radiation [W/m2]': 'DHI'})
        w['time_lst'] = w.date_time - pd.Timedelta(hours=1)       # period-ending -> hour-beginning
        w = w.drop(columns='date_time').set_index('time_lst').astype('float32')
        a_, b_ = 17.625, 243.04
        g = np.log(w.RH.clip(1, 100) / 100) + a_ * w.T_out / (b_ + w.T_out)
        w['dewpoint'] = (b_ * g / (a_ - g)).astype('float32')
        w['std_utc_offset_h'] = np.float32(STATE_TZ.get(st, np.nan)); w['county'] = cy; w['state'] = st
        w.to_parquet(f, compression='zstd'); done.add(cy)
    print('weather counties', len(done), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    {'sample': sample, 'fetch': lambda: fetch(a[1]), 'merge': lambda: merge(a[1]), 'weather': weather}[a[0]]()
