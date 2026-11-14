"""ERCOT frequency events from ERCOT's public monthly Operations Reports (section 'Frequency Events').
Each report lists events with delta frequency >= 60 mHz or MW loss >= 350 MW (criteria stated in the reports),
with date/time (Central Prevailing Time), delta frequency, min (or max) frequency, duration, PMU oscillation data,
MW loss, load, wind (later: IRR) share and system inertia (MW*s; the 2018 header says 'GW-s' but values are MW*s).
Output: proc/frequency/ercot_events.parquet ; coverage in proc/frequency/ercot_events_coverage.json
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, glob, json, re
import numpy as np, pandas as pd, docx
RAW = f'{GRID_ROOT}/data/raw/ercot_ops'; OUT = f'{GRID_ROOT}/data/proc/frequency'
os.makedirs(OUT, exist_ok=True)
urls = dict(l.strip().split('\t') for l in open(f'{RAW}/urls.tsv') if l.strip())


def num(s):
    s = str(s).replace(',', '').replace('%', '').strip()
    try: return float(s)
    except ValueError: return np.nan


rows = []; cov = {}
for f in sorted(glob.glob(f'{RAW}/*.docx')):
    m = os.path.basename(f)[:7]
    try: d = docx.Document(f)
    except Exception as e: cov[m] = f'unreadable: {e}'; continue
    n = 0
    for t in d.tables:
        h0 = [c.text.strip() for c in t.rows[0].cells]
        if not (any('MW Loss' in h for h in h0) and any('Delta Frequency' in h for h in h0)): continue
        h1 = [c.text.strip() for c in t.rows[1].cells] if len(t.rows) > 1 else h0
        hdr = [f'{a} {b}' for a, b in zip(h0, h1)]
        def col(key):
            for i, h in enumerate(hdr):
                if key in h: return i
            return None
        ci = dict(dt=0, df=col('Delta Frequency'), fmm=col('Max/Min'), dur=col('Duration'), mw=col('MW Loss'),
                  load=col('Load'), ren=col('Wind') if col('Wind') is not None else col('IRR'), H=col('Inertia'),
                  osc=col('Oscillation'), damp=col('Damping'))
        for r in t.rows[2:]:
            c = [x.text.strip() for x in r.cells]
            if not c[0] or not re.match(r'\d{1,2}/\d{1,2}/\d{4}', c[0]): continue
            g = lambda k: c[ci[k]] if ci[k] is not None and ci[k] < len(c) else ''
            rows.append(dict(month=m, time_local=c[0], delta_f_Hz=num(g('df')), f_extreme_Hz=num(g('fmm')),
                             duration=g('dur'), mw_loss=num(g('mw')), load_MW=num(g('load')), renew_share_pct=num(g('ren')),
                             renew_kind='wind' if (ci['ren'] is not None and 'Wind' in hdr[ci['ren']]) else 'IRR',
                             inertia_MWs=num(g('H')), osc_mode_Hz=num(g('osc')), damping_pct=num(g('damp')), source_url=urls.get(m)))
            n += 1
    cov[m] = n
e = pd.DataFrame(rows)
t = pd.to_datetime(e.time_local, format='mixed')
# Central Prevailing Time -> UTC (ambiguous fall-back hour: assume first occurrence; nonexistent: shift forward)
e['time_utc'] = t.dt.tz_localize('America/Chicago', ambiguous=True, nonexistent='shift_forward').dt.tz_convert('UTC')
e['under_frequency'] = e.f_extreme_Hz < 60
e = e.drop_duplicates(['time_utc', 'mw_loss']).sort_values('time_utc').reset_index(drop=True)
e['event_id'] = 'ERCO_' + e.time_utc.dt.strftime('%Y%m%dT%H%M%S')
e.to_parquet(f'{OUT}/ercot_events.parquet', index=False)
json.dump({'months_with_report': cov, 'n_events': int(len(e)), 'n_reports': len(cov),
           'months_missing': [str(p) for p in pd.period_range('2018-01', '2026-06', freq='M') if str(p) not in cov]},
          open(f'{OUT}/ercot_events_coverage.json', 'w'), indent=1)
print(len(e), 'events from', len(cov), 'reports'); print(e.describe().T[['count', '50%', 'min', 'max']])
