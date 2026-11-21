"""Fixed exam REPLAY subset (for physics replays / any method that needs landscape + hourly weather on exam fires).
Drawn once (seed 20261118 + 7) from exam fires 2017-2024 (HRRR + pre-fire LANDFIRE available) with >= 3 PM growth days
and FEDS/MTBS area ratio 0.5-1.5; 32 from 'main' (2017-21) and 16 from 'recent' (2022-24), allocated by region share,
systematic within region over (fuel group, size class, year). -> proc/baselines/replay_subset.json.
Used only for final exam scoring. Never look at these fires' results before Phase 2 freezes its models."""
import os, json
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
P = f'{_DATA}/data/proc'
OUT = f'{P}/baselines/replay_subset.json'
SEED = 20261118 + 7
if os.path.exists(OUT): raise SystemExit('replay subset exists (frozen)')
f = pd.read_parquet(f'{P}/fires.parquet')
E = f[(f.split == 'exam') & (f.year >= 2017) & (f.feds_growth_days >= 3) & f.feds_ratio.between(0.5, 1.5)]
rng = np.random.default_rng(SEED)
pick = []
for period, n in [('main', 32), ('recent', 16)]:
    e = E[E.exam_period == period]
    cnt = e.region.value_counts()
    alloc = {r: int(round(n * c / len(e))) for r, c in cnt.items()}
    while sum(alloc.values()) > n: alloc[max(alloc, key=lambda k: alloc[k] - n * cnt[k] / len(e))] -= 1
    while sum(alloc.values()) < n: alloc[max(alloc, key=lambda k: n * cnt[k] / len(e) - alloc[k])] += 1
    for r in sorted(alloc):
        if alloc[r] <= 0: continue
        d = e[e.region == r].sort_values(['fuel_group', 'size_class', 'year', 'Event_ID'])
        step = len(d) / alloc[r]; st = rng.uniform(0, step)
        pick += list(d.Event_ID.iloc[sorted({int(st + k * step) for k in range(alloc[r])})])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(seed=SEED, rule=__doc__, fires=pick, n=len(pick)), open(OUT, 'w'), indent=1)
print(len(pick)); print(f[f.Event_ID.isin(pick)][['Event_ID', 'Incid_Name', 'region', 'fuel_group', 'size_class', 'year']].to_string())
