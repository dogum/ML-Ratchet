"""Real GB frequency events for the grid laws (GridData). Dev = GridData's 4 GB dev events; exam = 412 exam events
(read only by exam_grid.py). Demand = NESO generation mix GENERATION of the settlement period (as GridPulse);
IBR share = (wind + embedded wind + solar + imports) / generation."""
import numpy as np, pandas as pd

# GridData's 14 Nov 2026 label correction (repo shared/grid/data/sources/gb_event_label_corrections.csv); the sandbox
# copy of GridData predates it, so it is applied here.
CORRECT = {'GB_20191203T200030': 854.7}


def events(split):
    import griddata as gd
    ev = gd.events('GB', split=split).copy()
    for k, v in CORRECT.items():
        ev.loc[ev.event_id == k, 'loss_MW'] = v
    g = gd.gb_genmix().set_index('time_utc')
    t30 = pd.to_datetime(ev.t0_utc, utc=True).dt.floor('30min')
    G = g.reindex(pd.DatetimeIndex(t30))
    ev['demand_MW'] = G['GENERATION'].values
    ev['ibr_share'] = ((G['WIND'] + G['WIND_EMB'] + G['SOLAR'] + G['IMPORTS']) / G['GENERATION']).values
    ev['inertia_GVAs'] = ev['inertia_outturn_GVAs']
    ev['df_nadir_Hz'] = ev['f0_Hz'] - ev['nadir_Hz']
    return ev.reset_index(drop=True)
