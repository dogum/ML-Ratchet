"""FaultFinder exam 2 candidate: Wang et al. 2024 labelled Korean office AHUs (figshare 25909813 v1). 17 Nov 2026.

    python exam2_wang.py fetch      # combined_FDD.xls (0.8 MB, CC BY 4.0) into work/exam2/wang/raw
    python exam2_wang.py build      # checks it against the training noise data; writes training/split_exam2_wang.json

Finding (before anything was scored): the public labelled file is NOT usable as a fresh exam.
  * It holds January 2024 only (5,471 hourly rows in 20 blocks without an AHU name), labelled "Fan fault" or
    "Sensor fault"; there is no "normal" row, so detection AUC, balanced accuracy and false-alarm rate cannot be
    computed.
  * Its readings are the same readings as BuildingData's `korea_office` set (figshare 25932409 v2, Nov 2023 - May
    2024), which fitted FaultFinder v1.0's noise model (start/stop jitter, supply-air tracking tail). This script
    matches every labelled row to korea_office on (time, T_ra, T_sa, T_sa_sp) and maps each block to its AHU.
So the split file records the rejection, the evidence and per-AHU hashes, and freezes no exam rows. Needs xlrd >= 2.0.2
to read the .xls (pip install xlrd, or put it on PYTHONPATH). Scores nothing.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths as _P  # noqa: E402
import exam2_common as X  # noqa: E402

OUT = os.path.join(_P.WORK, 'exam2', 'wang')
RAW, PROC = os.path.join(OUT, 'raw'), os.path.join(OUT, 'proc')
SPLIT = os.path.join(HERE, 'split_exam2_wang.json')
FILES = [dict(name='combined_FDD.xls', url='https://ndownloader.figshare.com/files/46583692',
              md5='dff24b2453a1b787d9bf950736c3c2ac', size=797696)]
SOURCE = dict(dataset='Wang et al. 2024, semi-labelled AHU FDD dataset (large office, Gwacheon, South Korea)',
              version='figshare v1', doi='10.6084/m9.figshare.25909813.v1', licence='CC BY 4.0',
              paper='Data in Brief (2024), PMC11460478')
COLS = {'Ventilation Temperature': 'T_ra', 'Supply Air Temperature': 'T_sa', 'Set Temperature': 'T_sa_sp',
        'Supply Fan': 'fan_sa_status', 'Valve Position': 'x_valve_position',
        'Heating Supply Temperature 1': 'x_T_hw_supply_1', 'Heating Supply Temperature 2': 'x_T_hw_supply_2',
        'Total Heating Pump': 'x_heating_pump_total', 'Heating Pump 1': 'x_heating_pump_1',
        'Heating Pump 2': 'x_heating_pump_2', 'Backup Heating Pump': 'x_heating_pump_backup'}


def build(check=False):
    v = pd.read_excel(os.path.join(RAW, 'combined_FDD.xls'), engine='xlrd')
    v['time_local'] = pd.to_datetime(v['Time'], format='%Y-%m-%d %H')
    v = v.rename(columns=COLS)
    v['block'] = (v.time_local.diff() < pd.Timedelta(0)).cumsum()
    import buildingdata  # noqa: F401  (path set by _paths)
    kpath = os.path.join(_P.BLDG, 'data', 'proc', 'faults', 'data', 'korea_office.parquet')
    k = pd.read_parquet(kpath)
    key = lambda d: pd.Series(list(zip(d.time_local, d.T_ra.astype(float).round(2), d.T_sa.astype(float).round(2),
                                       d.T_sa_sp.astype(float).round(2))))
    kk = pd.DataFrame({'key': key(k), 'run_id': k.run_id.to_numpy()})
    kmap = kk.drop_duplicates('key').set_index('key').run_id
    v['matched_run'] = key(v).map(kmap).to_numpy()
    units = {}
    for b, g in v.groupby('block'):
        ahu = g.matched_run.value_counts().index[0]
        units[f'block_{b:02d}'] = dict(
            ahu_in_korea_office=ahu, n_rows=int(len(g)), first=str(g.time_local.min()), last=str(g.time_local.max()),
            labels=g['Labeling'].value_counts().to_dict(), rows_matching_korea_office=int(g.matched_run.notna().sum()),
            rows_matching_same_ahu=int((g.matched_run == ahu).sum()),
            sha256=X.hash_frame(g.sort_values('time_local'), ['T_ra', 'T_sa', 'T_sa_sp', 'fan_sa_status']))
    n_match = int(v.matched_run.notna().sum())
    doc = dict(
        frozen='2026-11-17', status='rejected: not an exam', source=SOURCE,
        reason=[f'{n_match} of {len(v)} labelled rows ({100 * n_match / len(v):.1f} %) are the same readings (time, '
                'T_ra, T_sa, T_sa_sp to 0.01) as BuildingData korea_office (figshare 25932409 v2, Nov 2023 - May '
                '2024), which fitted FaultFinder v1.0 noise statistics; each of the 20 blocks maps to one AHU.',
                'Only fault rows are published (labels: ' + ', '.join(f'{k} {n}' for k, n in
                                                                    v['Labeling'].value_counts().items()) +
                '); no normal rows, so detection AUC, balanced accuracy and false-alarm rate are undefined.',
                'The record describes Jul-Oct 2023 data; the file holds ' + str(v.time_local.min()) + ' to ' +
                str(v.time_local.max()) + '.'],
        input_gap='Channels T_ra, T_sa, T_sa_sp, supply fan only (4 of 13; hourly, 24 of 96 bins). No T_oa, T_ma, '
                  'damper, cooling valve, fan speed or power, occupancy; the valve position is left missing as in '
                  'the korea_office reader (meaning unclear). FaultFinder can run on it (masks zero the missing '
                  'channels) but the APAR rules that need T_oa / T_ma / damper cannot fire.',
        seed=X.SEED, exam_rows=[], baselines=[], units=units)
    sha = X.write_frozen(SPLIT, doc, check)
    print(f'{n_match}/{len(v)} rows match korea_office; split file {SPLIT} sha256 {sha}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if cmd == 'fetch':
        print(X.fetch_files(FILES, RAW, SOURCE))
    else:
        build(check='--check' in sys.argv)
