"""Per-unit table and operating characteristics from CAMPD hourly data (Texas units).

proc/campd/units.parquet : one row per CAMPD unit (unit_id = '<plant_id_epa>_<unit_id_epa>'): names, unit type, fuel,
    EIA plant/generator links (EPA-EIA crosswalk, PUDL core_epa__assn_eia_epacamd), BA, years present, hours operating.
proc/campd/unit_characteristics.parquet : per unit, from years 2018-2022 ONLY (non-exam years; still, for exam PLANTS
    these are exam truth - see DATA.md):
    pmax_gross_MW (p99.5 of full-hour gross load), pmin_frac (p2 of steady full-hour load / pmax),
    ramp_up/down_frac_per_min (p99 of hour-to-hour change between steady full hours / 60 / pmax; a LOWER bound),
    min_up_h / min_down_h (p5 of run / off-spell lengths), starts_per_year, hr_a/b/c (heat input MMBtu/h = a + bP + cP^2,
    robust fit on steady full hours, gross basis), hr_at_pmax, n_fit
proc/campd/unit_characteristics_gen.parquet : the same mapped to EIA (plant_id_eia, generator_id) for the fleet builder.
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, glob
import numpy as np, pandas as pd
PROC = f'{GRID_ROOT}/data/proc'; RAW = f'{GRID_ROOT}/data/raw'
H = f'{PROC}/campd/hourly'
YEARS_CHAR = range(2018, 2023)

st = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{H}/units_tx_*.parquet'))], ignore_index=True)
st['plant_id_epa'] = st['Facility ID'].astype(int); st['unit_id_epa'] = st['Unit ID'].astype(str)
st = st.sort_values('year')
units = st.groupby(['plant_id_epa', 'unit_id_epa']).agg(
    facility_name=('Facility Name', 'last'), unit_type=('Unit Type', 'last'), primary_fuel=('Primary Fuel Type', 'last'),
    secondary_fuel=('Secondary Fuel Type', 'last'), nox_controls=('NOx Controls', 'last'), so2_controls=('SO2 Controls', 'last'),
    stacks=('Associated Stacks', 'last'), first_year=('year', 'min'), last_year=('year', 'max')).reset_index()
units['unit_id'] = units.plant_id_epa.astype(str) + '_' + units.unit_id_epa
xw = pd.read_parquet(f'{RAW}/pudl/core_epa__assn_eia_epacamd.parquet')
xw = xw.sort_values('report_year').drop_duplicates(['plant_id_epa', 'emissions_unit_id_epa', 'generator_id'], keep='last')
gens = xw.groupby(['plant_id_epa', 'emissions_unit_id_epa']).agg(plant_id_eia=('plant_id_eia', 'first'),
                                                                generator_ids=('generator_id', lambda s: sorted(set(map(str, s)))),
                                                                boiler_ids=('boiler_id', lambda s: sorted(set(map(str, s.dropna()))))).reset_index()
units = units.merge(gens.rename(columns={'emissions_unit_id_epa': 'unit_id_epa'}), on=['plant_id_epa', 'unit_id_epa'], how='left')
units['plant_id_eia'] = units.plant_id_eia.fillna(units.plant_id_epa).astype(int)
units['crosswalk'] = units.generator_ids.notna()
pl = pd.read_parquet(f'{RAW}/pudl/out_eia__yearly_plants.parquet', columns=['plant_id_eia', 'report_date', 'balancing_authority_code_eia', 'latitude', 'longitude', 'county'])
pl = pl.sort_values('report_date').dropna(subset=['balancing_authority_code_eia']).groupby('plant_id_eia').last()
units['ba'] = units.plant_id_eia.map(pl.balancing_authority_code_eia)
units['lat'] = units.plant_id_eia.map(pl.latitude); units['lon'] = units.plant_id_eia.map(pl.longitude)
# EIA tech of the linked generators (largest)
fl = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{PROC}/fleet/fleet_*.parquet')], ignore_index=True)
fl['gkey'] = fl.plant_id.astype(str) + '|' + fl.generator_id.astype(str)
cap = dict(zip(fl.gkey, fl.nameplate_MW)); tech = dict(zip(fl.gkey, fl.tech))
def link(r):
    if not isinstance(r.generator_ids, (list, np.ndarray)) or len(r.generator_ids) == 0: return pd.Series({'tech': None, 'eia_nameplate_MW': np.nan})
    ks = [f'{r.plant_id_eia}|{g}' for g in r.generator_ids]
    caps = [cap.get(k, np.nan) for k in ks]; ts = [tech.get(k) for k in ks]
    ts_ok = [t for t in ts if t]
    t = 'gas_cc' if 'gas_cc' in ts_ok else (ts_ok[int(np.nanargmax([c if c == c else -1 for c in caps]))] if ts_ok else None)
    return pd.Series({'tech': t, 'eia_nameplate_MW': np.nansum(caps) if np.isfinite(caps).any() else np.nan})
units = pd.concat([units, units.apply(link, axis=1)], axis=1)


def runs(on):
    """Return list of (start, length, value) runs of a boolean array."""
    on = np.asarray(on, bool)
    if len(on) == 0: return []
    idx = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    starts = np.r_[0, idx]; ends = np.r_[idx, len(on)]
    return [(s, e - s, on[s]) for s, e in zip(starts, ends)]


rows = []; hours = []
for (pe, ue), _ in units.groupby(['plant_id_epa', 'unit_id_epa']):
    parts = []
    for y in range(2018, 2026):
        f = f'{H}/tx_{y}.parquet'
        if not os.path.exists(f): continue
        parts.append(pd.read_parquet(f, columns=['time_utc', 'op_time', 'gross_load_MW', 'heat_input_MMBtu', 'hi_meas'],
                                     filters=[('plant_id_epa', '=', pe), ('unit_id_epa', '=', ue)]))
    d = pd.concat(parts, ignore_index=True).sort_values('time_utc')
    yrs = d.time_utc.dt.year
    op = (d.op_time.fillna(0) > 0).values
    hours.append(dict(plant_id_epa=pe, unit_id_epa=ue, hours_total=len(d), hours_operating=int(op.sum()),
                      hours_operating_2018_2022=int(op[(yrs <= 2022).values].sum()),
                      gen_GWh_gross=float(d.gross_load_MW.fillna(0).sum() / 1e3)))
    c = d[(yrs >= 2018) & (yrs <= 2022)].reset_index(drop=True)
    r = dict(plant_id_epa=pe, unit_id_epa=ue)
    if len(c) == 0 or (c.op_time.fillna(0) > 0).sum() < 100:
        rows.append(r); continue
    opc = (c.op_time.fillna(0) > 0).values
    full = (c.op_time.fillna(0) >= 0.999).values & (c.gross_load_MW.fillna(0) > 0).values
    # steady = full hour whose neighbours are also full hours (no start/stop inside the window)
    steady = full & np.r_[False, full[:-1]] & np.r_[full[1:], False]
    P = c.gross_load_MW.values.astype(float); HI = c.heat_input_MMBtu.values.astype(float)
    if full.sum() < 50: rows.append(r); continue
    pmax = np.nanpercentile(P[full], 99.5)
    r['pmax_gross_MW'] = pmax
    if steady.sum() >= 50:
        r['pmin_frac'] = float(np.nanpercentile(P[steady], 2) / pmax)
        dP = np.diff(P); both = steady[1:] & steady[:-1]
        if both.sum() > 50:
            r['ramp_up_frac_per_min'] = float(max(np.nanpercentile(dP[both], 99), 0) / 60 / pmax)
            r['ramp_down_frac_per_min'] = float(max(-np.nanpercentile(dP[both], 1), 0) / 60 / pmax)
    # contiguous hourly grid needed for run lengths: assume rows are hourly (CAMPD reports every hour of the year)
    rr = runs(opc)
    inner = rr[1:-1]
    up = [L for s, L, v in inner if v]; down = [L for s, L, v in inner if not v]
    if len(up) >= 5: r['min_up_h'] = float(np.percentile(up, 5)); r['median_run_h'] = float(np.median(up))
    if len(down) >= 5: r['min_down_h'] = float(np.percentile(down, 5))
    r['starts_per_year'] = float(sum(1 for s, L, v in rr[1:] if v) / len(set(c.time_utc.dt.year)))
    # heat-rate fit on steady hours with measured heat input
    m = steady & np.isfinite(HI) & (HI > 0) & np.isfinite(P)
    if m.sum() >= 200:
        x = P[m]; y = HI[m]
        for _ in range(2):
            A = np.c_[np.ones_like(x), x, x * x]; coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            res = y - A @ coef; mad = np.median(np.abs(res - np.median(res))) * 1.4826 + 1e-9
            k = np.abs(res) < 4 * mad; x, y = x[k], y[k]
        r.update(hr_a=float(coef[0]), hr_b=float(coef[1]), hr_c=float(coef[2]), n_fit=int(len(x)),
                 hr_at_pmax=float((coef[0] + coef[1] * pmax + coef[2] * pmax ** 2) / pmax),
                 hr_resid_rel=float(np.median(np.abs(res[k])) / np.median(y)))
    rows.append(r)
ch = pd.DataFrame(rows); hrs = pd.DataFrame(hours)
units = units.merge(hrs, on=['plant_id_epa', 'unit_id_epa'], how='left')
uo = units.copy()
for c in ['generator_ids', 'boiler_ids']:
    uo[c] = uo[c].apply(lambda v: ';'.join(map(str, v)) if isinstance(v, (list, np.ndarray)) else None)
for c in uo.columns:
    if str(uo[c].dtype).startswith('string') or str(uo[c].dtype).endswith('[pyarrow]'): uo[c] = uo[c].astype(object)
uo.to_parquet(f'{PROC}/campd/units.parquet', index=False)
ch['unit_id'] = ch.plant_id_epa.astype(str) + '_' + ch.unit_id_epa
ch.to_parquet(f'{PROC}/campd/unit_characteristics.parquet', index=False)
# map to generators: for each EIA generator take the linked CAMPD unit with the largest pmax
g = units[['plant_id_epa', 'unit_id_epa', 'plant_id_eia', 'generator_ids']].dropna(subset=['generator_ids']).explode('generator_ids')
g = g.rename(columns={'generator_ids': 'generator_id'}).merge(ch, on=['plant_id_epa', 'unit_id_epa'])
n_units = g.groupby(['plant_id_eia', 'generator_id']).size().rename('n_campd_units')
g = g.sort_values('pmax_gross_MW', ascending=False).drop_duplicates(['plant_id_eia', 'generator_id']).merge(n_units.reset_index())
n_gens = units.dropna(subset=['generator_ids']).set_index(['plant_id_epa', 'unit_id_epa']).generator_ids.map(len).rename('n_gens_of_unit')
g = g.merge(n_gens.reset_index(), on=['plant_id_epa', 'unit_id_epa'], how='left')
# heat-rate curve only meaningful 1:1 (one unit <-> one generator); otherwise drop it from the generator mapping
g.loc[(g.n_campd_units != 1) | (g.n_gens_of_unit != 1), ['hr_a', 'hr_b', 'hr_c']] = np.nan
g['generator_id'] = g.generator_id.astype(str)
g.to_parquet(f'{PROC}/campd/unit_characteristics_gen.parquet', index=False)
print(len(units), 'units;', int(units.crosswalk.sum()), 'with EIA crosswalk;', (units.ba == 'ERCO').sum(), 'in ERCO')
print(ch.describe().T[['count', '50%']])
