"""Freeze the dev/exam splits (proc/splits.json). Seed 20261113. Deterministic: re-running gives the same file.
Design (DATA.md section 9):
  time    : train_inputs 2018, 2020, 2022 | dev 2019, 2021 | exam 2023, 2024, 2025 | reserve 2026 (not used in Phase 2)
            CAISO OASIS keeps ~3 years, so CAISO prices: dev = 2023-08..2023-12, exam = 2024-2025, reserve 2026.
  plants  : PlantDetective, by EIA plant. dev = small stratified seeded draw of ERCOT plants with >= 8000 CAMPD operating
            hours (all units) still reporting in 2025 (2 coal, 3 gas CC, 3 gas CT, 2 gas steam); exam = every other CAMPD Texas plant. Never split by hour.
  events  : frequency. dev = named events (GB 9 Aug 2019 + the 3 largest medium-confidence GB losses in dev years 2019/2021;
            5 seeded ERCOT events from dev years); exam = every other event before 2026; 2026 events = reserve. Quiet 1 s noise for simulation comes from
            train_inputs years only, at least 1 h away from any event.
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, json, hashlib
import numpy as np, pandas as pd
PROC = f'{GRID_ROOT}/data/proc'; SEED = 20261113
rng = np.random.default_rng(SEED)
sp = {'seed': SEED, 'frozen': '2026-11-13', 'version': 1,
      'rule': 'Only dev may be used for calibration, gates and fitting. exam is for the final report only. '
              'Simulated training may use real INPUTS (weather, demand shapes, prices, noise) from train_inputs years only; '
              'dev-year inputs may also be used for calibration. reserve (2026) is untouched in Phase 2.'}
sp['time'] = {'train_inputs': [2018, 2020, 2022], 'dev': [2019, 2021], 'exam': [2023, 2024, 2025], 'reserve': [2026],
              'boundaries': 'calendar years in UTC (time_utc.year)',
              'caiso_prices': {'dev': ['2023-08-01', '2024-01-01'], 'exam': ['2024-01-01', '2026-01-01'], 'reserve': ['2026-01-01', '2027-01-01'],
                               'why': 'OASIS serves only ~3 years of history (Aug 2023 onward at download time)'}}

# ---- plants ----
u = pd.read_parquet(f'{PROC}/campd/units.parquet')
u = u[u.crosswalk & u.plant_id_eia.notna()]
cap = u.groupby(['plant_id_eia', 'tech']).eia_nameplate_MW.sum().reset_index()
dom = cap.sort_values('eia_nameplate_MW', ascending=False).drop_duplicates('plant_id_eia').set_index('plant_id_eia').tech
pl = u.groupby('plant_id_eia').agg(ba=('ba', 'first'), hours=('hours_operating', 'sum'), name=('facility_name', 'first'), last_year=('last_year', 'max'))
pl['stratum'] = dom.reindex(pl.index).fillna('other')
elig = pl[(pl.ba == 'ERCO') & (pl.hours >= 8000) & (pl.last_year >= 2025)]
quota = {'coal_steam': 2, 'gas_cc': 3, 'gas_ct': 3, 'gas_steam': 2}
dev = []
for s, k in quota.items():
    c = sorted(elig[elig.stratum == s].index.tolist())
    dev += [int(x) for x in rng.choice(c, size=min(k, len(c)), replace=False)]
dev = sorted(dev)
exam = sorted(int(x) for x in pl.index if int(x) not in dev)
sp['plants'] = {'unit': 'EIA plant_id (CAMPD facility id is the same for these plants; units.parquet maps both)',
                'dev': dev, 'exam': exam,
                'dev_detail': {int(p): {'name': pl.loc[p, 'name'], 'stratum': pl.loc[p, 'stratum']} for p in dev},
                'n_dev_units': int(u[u.plant_id_eia.isin(dev)].shape[0]), 'n_exam_units': int(u[u.plant_id_eia.isin(exam)].shape[0]),
                'strata_counts_exam': pl.loc[exam].stratum.value_counts().to_dict()}

# ---- events ----
ev = {}
gbf = f'{PROC}/frequency/gb_events.parquet'
if os.path.exists(gbf):
    g = pd.read_parquet(gbf)
    named = ['GB_20190809T155233']
    cand = g[(g.t0_utc.dt.year.isin([2019, 2021])) & (g.loss_confidence == 'medium') & ~g.event_id.isin(named)].sort_values('loss_MW', ascending=False)
    gdev = named + cand.event_id.head(3).tolist()
    gdev = [e for e in gdev if e in set(g.event_id)]
    res = set(g[g.t0_utc.dt.year >= 2026].event_id)
    ev['GB'] = {'dev': gdev, 'exam': sorted(set(g.event_id) - set(gdev) - res), 'reserve': sorted(res)}
ef = f'{PROC}/frequency/ercot_events.parquet'
if os.path.exists(ef):
    e = pd.read_parquet(ef)
    c = sorted(e[e.time_utc.dt.year.isin([2019, 2021]) & e.mw_loss.notna()].event_id)
    edev = sorted(rng.choice(c, size=min(5, len(c)), replace=False).tolist()) if c else []
    res = set(e[e.time_utc.dt.year >= 2026].event_id)
    ev['ERCO'] = {'dev': edev, 'exam': sorted(set(e.event_id) - set(edev) - res), 'reserve': sorted(res)}
sp['events'] = ev
sp['frequency_noise'] = {'years': [2018, 2020, 2022], 'min_gap_to_event_h': 1}
txt = json.dumps(sp, indent=1, default=str)
sp['sha256_of_content'] = hashlib.sha256(txt.encode()).hexdigest()
json.dump(sp, open(f'{PROC}/splits.json', 'w'), indent=1, default=str)
print('plants dev', dev, {k: len(v['dev']) for k, v in ev.items()}, sp['sha256_of_content'][:12])
