"""Reference numbers on the DEV gate window (WY2000-2014, 52 dev basins): B's U-dev regionalisation, NWM v3.0,
GR4J (calibrated on 1999-2008, so partly in-sample here). Context for the dev gate only."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import waterdata as wd
from waterdata import metrics as M

B = f'{WATER_DATA}/data/proc/baselines'
nn = pd.read_parquet(f'{B}/sims/nn_dev.parquet')
gr = pd.read_parquet(f'{B}/sims/gr4j_gauged.parquet')
nwm = pd.read_parquet(f'{WATER_DATA}/data/proc/nwm/daily_local.parquet')
rows = []
for b in wd.basins('camels', split='dev'):
    obs = wd.streamflow(b, 'caravan').Q_mm.loc[DEVWIN[0]:DEVWIN[1]]
    sims = {'nn_dev': nn[b] if b in nn else None, 'gr4j': gr[b] if b in gr else None}
    g = b.split('_')[1]
    if g in nwm.columns:
        sims['nwm'] = nwm[g] * 86.4 / wd.spec(b)['area_km2']
    for m, s in sims.items():
        if s is None:
            continue
        r = M.summary(obs, s.reindex(obs.index)); r.update({'basin': b, 'method': m}); rows.append(r)
df = pd.DataFrame(rows)
df.to_parquet(f'{ROOT}/results/devref.parquet')
print(df.groupby('method')[['nse', 'kge', 'fhv', 'peak_abs_err_pct_median']].median().round(3))
print(df.groupby('method').size())
