"""Freeze SnowSense's fresh exam: water year 2026 at the SNOTEL exam stations (frozen 18 Nov 2026, before any rung).

The v1.0 exam (WY2011-2025) was opened on import, so a new rung scored there would only give a seen number. WY2026
(1 Oct 2025 - 30 Sep 2026) had not been read by anyone working on SnowSense. This script fixes, without computing any
score or statistic of the snow itself:
  - the stations: every SNOTEL exam station in WaterData's frozen splits whose WY2026 is complete by WaterData's own rule
    (build_snotel.py: >= 330 SWE days, >= 300 P and T days in the water year);
  - the window: SnowSense reads 1 Aug 2025 - 1 Oct 2026 (its 427-day input); scoring is WY2026 daily SWE, peak SWE,
    peak date and melt-out, exactly as training/evaluate.py scores WY2011-2025;
  - a sha256 of each station's rows (1 Aug 2025 - 1 Oct 2026, every column), because SNOTEL's recent data are
    provisional and can be revised. The frozen rows are copied to models/snowsense/work/exam2_wy2026.parquet
    (git-ignored); a later re-fetch that hashes differently is reported, and the frozen copy is the one scored.

The rule (written before any rung): the exam is scored once, after the next rung's decision is final, for that rung,
v1.0 and every baseline (default and dev-tuned energy balance and degree-day, climatology, the per-station calibrated
degree-day) at the same time. Nothing is chosen on it.

Run: ML_RATCHET_WATER=<data root> python -I training/freeze_exam2_wy2026.py
"""
import hashlib
import json
import os
import sys
import time

import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import _paths  # noqa: E402,F401  (WORK, the water data root)

DATA = os.path.join(os.environ.get('ML_RATCHET_WATER') or os.path.join(_paths.REPO, 'shared', 'water'), 'data')
SPLITS_FILE_SHA = '33f88ca0225180dcce979eceed26bf5157b92455e13bb7b78e135e9fd4d98c6b'
WY = 2026
READ = ('2025-08-01', '2026-10-01')          # the network's input window (427 days), end-of-WY state included
SCORE = ('2025-10-01', '2026-09-30')


def main():
    raw = open(os.path.join(DATA, 'proc', 'splits.json'), 'rb').read()
    assert hashlib.sha256(raw).hexdigest() == SPLITS_FILE_SHA, 'splits.json is not the frozen file'
    exam = json.loads(raw)['snotel']['exam']
    st = pd.read_parquet(os.path.join(DATA, 'proc', 'snotel', 'stations.parquet')).set_index('station_id')
    rows, frames, missing = [], [], []
    for sid in sorted(exam):
        p = os.path.join(DATA, 'proc', 'snotel', 'daily', f'{sid}.parquet')
        if not os.path.exists(p) or sid not in st.index:
            missing.append(sid)
            continue
        if WY not in list(st.loc[sid, 'wy_complete']):
            continue
        d = pd.read_parquet(p).loc[READ[0]:READ[1]]
        blob = d.to_csv(lineterminator='\n').encode()
        rows.append({'station_id': sid, 'n_rows': int(len(d)), 'sha256': hashlib.sha256(blob).hexdigest()})
        frames.append(d.assign(station_id=sid))
    os.makedirs(_paths.WORK, exist_ok=True)
    frozen = pd.concat(frames)
    frozen.to_parquet(os.path.join(_paths.WORK, 'exam2_wy2026.parquet'))
    body = {
        'what': "SnowSense fresh exam: SNOTEL exam stations, water year 2026, frozen before any rung",
        'frozen': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
        'source': 'NRCS SNOTEL Report Generator, fetched 18 Nov 2026 by `mlratchet data water --only snotel`; recent data provisional',
        'splits_sha256': SPLITS_FILE_SHA,
        'eligibility': 'SNOTEL exam station (splits.json) with WY2026 complete: >= 330 SWE days, >= 300 P and T days',
        'read_window': list(READ), 'score_window': list(SCORE),
        'rule': ('Scored once, after the next rung decision is final: that rung, v1.0 and every baseline together. '
                 'Nothing is chosen on it. If a re-fetch hashes differently, the frozen copy is scored and the change reported.'),
        'n_exam_stations': len(exam), 'n_without_data': len(missing), 'n_frozen': len(rows),
        'stations': rows,
    }
    body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    out = os.path.join(SRC, 'exam2_wy2026.json')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(body, f, indent=1)
    print(f'froze {len(rows)} of {len(exam)} exam stations ({len(missing)} without data); {out}; sha256 {body["sha256"][:12]}')


if __name__ == '__main__':
    main()
