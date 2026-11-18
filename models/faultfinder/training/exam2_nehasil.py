"""FaultFinder exam 2: Nehasil, CTU Prague, six real AHUs with induced faults (Mendeley jkw85sm4v4 v1). 17 Nov 2026.

    python exam2_nehasil.py fetch    # 8 files (28 MB, CC BY 4.0) into work/exam2/nehasil/raw
    python exam2_nehasil.py build    # contract mapping, unit-day labels, QC, hashes -> training/split_exam2_nehasil.json

Scores nothing. Plan: NOTEBOOK.md entry of 2026-11-17 "Fresh exam frozen before round-2 rungs". Every AHU is exam (no
dev split); baselines use the platform dev runs. split_exam2_nehasil.json is frozen: build refuses to overwrite it
(--check rebuilds and compares).

Data facts found on reading (before any scoring): 5-min BMS logs 1 Mar - 31 Aug 2019 per AHU; a per-row `fault`
column ("no fault" or the fault name) and a fault schedule sheet. These are 100 % outdoor-air units with heat recovery
(fresh-air and exhaust fans, no mixing box): there is no outdoor-air or mixed-air temperature point and no supply-air
setpoint (the controller's setpoint is for the exhaust-air inlet, i.e. the room).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths as _P  # noqa: E402
import exam2_common as X  # noqa: E402

OUT = os.path.join(_P.WORK, 'exam2', 'nehasil')
RAW, PROC = os.path.join(OUT, 'raw'), os.path.join(OUT, 'proc')
SPLIT = os.path.join(HERE, 'split_exam2_nehasil.json')
BASE = 'https://data.mendeley.com/public-files/datasets/jkw85sm4v4/files/'
FILES = [  # name, file id, size, sha256 as published by the Mendeley API (17 Nov 2026)
    ('AHU1.xlsx', 'bcf67016-890d-44ea-9438-30a923711fe9', 6614970, 'ed51c373f25b4892610402fdd553e75a7e43342657b8759b863681dd0909e9c3'),
    ('AHU2.xlsx', '80707ccc-c9a3-4fae-888c-dd73b2ebce06', 1526394, '814f18e2299a7432af143f33d3b5d4711d23246c8cc25225d65d1f2efb73128a'),
    ('AHU8.xlsx', '7621086b-6b16-48f6-a695-3de15c27256a', 1526318, '131b489a74f1edc5a180c36338a09ecaaf61820184ac93ebedfec127b3a171c8'),
    ('AHU18.xlsx', 'fa29ec0a-69e6-46d8-b8da-9b22f1ffa78e', 6045715, '1aec89da0425959412a2891419280db035046dbd8d541968acb23d9dc65d59f7'),
    ('AHU23.xlsx', '55812f40-9b2b-414f-b396-5cd781fba09c', 6129279, '0ce2955409d78371872234ff9f0b6cc2a6de16d59d0851a00083b7d3c8de7274'),
    ('AHU24.xlsx', '13e6ab01-15e3-44f1-9fcf-7f7f4d051341', 6368430, 'fb91fcb0e4fea1a44247e8201b1b57d917502d17e01f16a7b445eac978685cd1'),
    ('Faults_schedule_and_description.xlsx', '91fd9e88-4db1-45a4-b07e-e62be622a6ef', 25957, '9752e1be8e8d1e8be98a1ab305e8c0cdf5d04af3eddbd5adf32746ff9fa54672'),
    ('Place_tag_explanation.jpg', '5fc16896-57dd-4035-ba0f-c4f796251114', 907307, 'f493877d72826b789bbab8db6dad448fd626ccddced520846a1c2e9ef205acce'),
]
FILES = [dict(name=n, url=BASE + i + '/file_downloaded', size=s, sha256=h) for n, i, s, h in FILES]
SOURCE = dict(dataset='Nehasil, Versatile AHU fault detection - design, field validation and practical application',
              version='Mendeley Data v1 (2 Mar 2021)', doi='10.17632/jkw85sm4v4.1', licence='CC BY 4.0',
              landing='https://data.mendeley.com/datasets/jkw85sm4v4/1')
AHUS = ['AHU1', 'AHU2', 'AHU8', 'AHU18', 'AHU23', 'AHU24']

# Contract mapping: only tags that name a contract channel. Signals are 0-100 % -> 0..1.
MAP = {'air_temperature__real_value__supply_air_outlet': ('T_sa', 1.0),
       'air_temperature__real_value__exahust_air_inlet': ('T_ra', 1.0),          # air leaving the room = return
       'logic__required_value__valve_outdoor_air': ('damper_oa', 1.0),            # OA damper open/closed command
       'signal__required_value__heater_valve': ('valve_hc', 0.01),
       'signal__required_value__chiller_valve': ('valve_cc', 0.01),
       'signal__required_value__fan_fresh_air': ('fan_sa_speed', 0.01),           # supply (fresh-air) fan command
       'logic__real_value__fan_fresh_air': ('fan_sa_status', 1.0)}
NOT_MAPPED = {'air_temperature__required_value__exahust_air_inlet': 'room/return setpoint, not a supply-air setpoint',
              'logic__real_value__air_handling_unit': 'unit running state, not occupancy'}

# Diagnosis map for the file's labels (FaultFinder classes; None = other). 'ambiguous' = the schedule names a
# different valve than the file label: still a fault day for detection, left out of the diagnosis row.
DIAG = {'Dampers are closed during heating regime': 'oa_damper_stuck',
        'Dampers are closed during cooling regime': 'oa_damper_stuck',
        'Dampres are closed during ventilate regime': 'oa_damper_stuck',
        'Heating valve is closed during heating regime': 'hc_valve_stuck',
        'Cooling valve is closed during cooling regime': 'cc_valve_stuck',
        'Heating valve is ON during ventilate regime': 'hc_valve_stuck',
        'Heating valve is stuck in intermediate position during heating regime': 'hc_valve_stuck',
        'Heating valve is open to the maximum level during heating regime': 'hc_valve_stuck',
        'Heating valve is OFF during humidifying regime': 'hc_valve_stuck',
        'Heating valve is stuck in intermediate position during cooling regime': 'ambiguous',
        'Heating valve is open to the maximum level during cooling regime': 'ambiguous',
        'Zone inlet temperature sensor indicates -20 °C': 'sensor_bias_T_sa',
        'Zone inlet temperature sensor indicates 150 °C': 'sensor_bias_T_sa',
        'Zone outlet temperature sensor indicates -20 ° C': 'sensor_bias_T_ra',
        'Zone outlet temperature sensor indicates 150 °C ': 'sensor_bias_T_ra'}
# everything else (pumps, fans off, differential-pressure tubes, quick cycling, heat exchanger closed) -> other

MIN_FAULT_ROWS = 12          # >= 1 h of 5-min rows labelled faulty makes a fault day; 1-11 rows -> excluded
SLOTS_NEEDED, SLOT_S = 72, 900


def read_ahu(a):
    x = pd.read_excel(os.path.join(RAW, f'{a}.xlsx'), sheet_name=0)
    d = pd.DataFrame({'time_local': pd.to_datetime(x['Time'])})
    for src, (dst, k) in MAP.items():
        d[dst] = x[src].astype(float) * k if src in x else np.nan
    for c in X.CONTRACT:
        if c not in d:
            d[c] = np.nan
    d['label'] = x['fault'].astype(str)
    dup = int(d.time_local.duplicated().sum())
    d = d.drop_duplicates('time_local').sort_values('time_local').reset_index(drop=True)
    return X.clean_channels(d), dup, [c for c in x.columns if c not in ('Time', 'fault', 'equipment')]


def schedule():
    s = pd.read_excel(os.path.join(RAW, 'Faults_schedule_and_description.xlsx'), sheet_name='Faults Schedule')
    s.columns = ['fault_no', 'name', 'ahu', 'date', 'start', 'end', 'comment']
    s['date'] = [pd.Timestamp(v) if not isinstance(v, str) else pd.to_datetime(v.strip(), format='%d.%m.%Y')
                 for v in s.date]
    return s


def build(check=False):
    os.makedirs(PROC, exist_ok=True)
    sch = schedule()
    units, days_all, dropped_units, disagree = {}, [], {}, []
    for a in AHUS:
        d, dup, points = read_ahu(a)
        d['date'] = d.time_local.dt.normalize()
        rows = []
        kept = []
        sch_days = set(sch[sch.ahu == a].date.dt.normalize())
        for day, g in d.groupby('date'):
            if day >= pd.Timestamp('2019-08-31'):          # the file's last row is 31 Aug 00:00 only
                continue
            nf = int((g.label != 'no fault').sum())
            names = g.label[g.label != 'no fault'].value_counts()
            if nf == 0:
                lab, ftype = 'normal', 'none'
            elif nf >= MIN_FAULT_ROWS:
                lab, ftype = 'fault', names.index[0]
            else:
                lab, ftype = 'excluded', names.index[0]
            if (nf > 0) != (day in sch_days):
                disagree.append(dict(ahu=a, date=str(day.date()), file_fault_rows=nf, in_schedule=day in sch_days))
            # The frozen split counted each 5-min row as SLOT_S (15 min) of fan runtime; keep that so --check reproduces
            # it. With the true 5 min, 2 of 513 days (AHU24 5 and 11 Mar 2019, both normal) fall under 1 h (NOTEBOOK).
            ok, why, g2 = X.day_qc(g, SLOTS_NEEDED, SLOT_S, runtime_s=SLOT_S)
            if lab == 'excluded':
                ok, why = False, f'fault rows 1-{MIN_FAULT_ROWS - 1} (< 1 h)'
            diag = None
            if lab == 'fault':
                diag = DIAG.get(ftype, 'other')
            rows.append(dict(ahu=a, date=str(day.date()), label=lab, fault_type=ftype, diag_class=diag,
                             n_fault_rows=nf, n_fault_types=int(len(names)), gradable=bool(ok), reason=why))
            if ok:
                kept.append(g2)
        r = pd.DataFrame(rows)
        days_all.append(r)
        if not kept:
            dropped_units[a] = dict(reason='no gradable day: ' + '; '.join(f'{k} {v}' for k, v in
                                                                      r.reason.value_counts().items()),
                                    mapped_channels_with_data=[c for c in X.CONTRACT if d[c].notna().any()])
            continue
        k = pd.concat(kept)
        k['is_fault'] = (k.label != 'no fault').astype(float)
        k[['time_local'] + X.CONTRACT + ['label']].to_parquet(os.path.join(PROC, f'{a}.parquet'), index=False)
        g = r[r.gradable]
        units[a] = dict(role='exam', n_days=int(len(g)), n_fault_days=int((g.label == 'fault').sum()),
                        n_normal_days=int((g.label == 'normal').sum()),
                        first=str(k.time_local.min()), last=str(k.time_local.max()),
                        channels=[c for c in X.CONTRACT if k[c].notna().any()],
                        duplicate_timestamps=dup,
                        dropped_days=r[~r.gradable].reason.value_counts().to_dict(),
                        fault_types={str(t): int(n) for t, n in g[g.label == 'fault'].fault_type.value_counts().items()},
                        sha256=X.hash_frame(k, X.CONTRACT + ['is_fault']))
    days = pd.concat(days_all)
    days.to_csv(os.path.join(PROC, 'unit_days.csv'), index=False)
    g = days[days.gradable]
    exam_days = [dict(ahu=r.ahu, date=r.date, label=r.label, fault_type=r.fault_type,
                      diag_class=r.diag_class if isinstance(r.diag_class, str) else None) for r in g.itertuples()]
    doc = dict(
        frozen='2026-11-17',
        purpose='FaultFinder exam 2 (Nehasil, CTU Prague): real AHUs no version has seen; scored once for v1.0 and '
                'once for any round-2 release, after the round-2 release decision',
        status='exam', source=SOURCE, seed=X.SEED, split='by unit; every AHU that passes QC is exam (no dev part)',
        contract_map={k: v[0] for k, v in MAP.items()}, not_mapped=NOT_MAPPED,
        input_gap='No T_oa, T_ma, T_sa_sp, T_zone, fan power or occupancy point exists (100 % outdoor-air units with '
                  'heat recovery). FaultFinder reads the mapped channels; the missing ones are zero with mask 0. '
                  'The damper point is an open/closed command, not a modulating position. APAR rules needing T_oa, '
                  'T_ma or a supply setpoint cannot fire; only R23 (simultaneous heating and cooling) can.',
        day_rules=dict(unit_day='AHU x local calendar day (5-min rows, local time as logged)',
                       fault_day=f'>= {MIN_FAULT_ROWS} rows (1 h) labelled with a fault in the file',
                       normal_day='no row labelled with a fault',
                       excluded='1-11 fault rows; or failing day QC',
                       labels='per-row file labels take precedence over the schedule sheet (disagreements listed)',
                       qc='temperatures < -45 or > 70 degC -> missing (the induced 150 degC readings become missing); '
                          'a temperature constant all day -> missing; T_sa in >= 72 of 96 15-min bins; fan on >= 1 h'),
        dropped_units=dropped_units,
        schedule_vs_file_disagreements=disagree,
        units=units,
        exam_days=exam_days,
        n=dict(ahus=len(units), days=int(len(g)), fault_days=int((g.label == 'fault').sum()),
               normal_days=int((g.label == 'normal').sum()),
               diagnosable_fault_days=int(g.diag_class.isin([c for c in set(DIAG.values()) if c != 'ambiguous']).sum())),
        rows=[
            dict(id='det_auc', metric='AHU-day detection AUC (P(fault) log-odds as the score)', better='higher',
                 unit_set='all exam_days'),
            dict(id='bal_acc', metric='balanced accuracy at P = 0.5', better='higher', unit_set='all exam_days'),
            dict(id='fa_rate', metric='false-alarm rate at P = 0.5 on normal days', better='lower',
                 unit_set='exam_days with label normal'),
            dict(id='diag_top1', metric='diagnosis top-1 accuracy on fault days with a mapped class', better='higher',
                 unit_set="exam_days with diag_class not in (None, 'other', 'ambiguous')",
                 note='info only (no verdict) if n < 10'),
        ],
        model_input='released faultfinder_weights.npz through FaultFinder.read_day on processed/<AHU>.parquet (contract '
                    'columns, time_local), one call per exam day; P from the shipped temperature. A round-2 release '
                    'is scored the same way with its own shipped calibration. No refit on these data.',
        baselines=[
            dict(name='APAR rule set, platform dev threshold', kind='classical',
                 how='shared/building/data/scripts/baselines.py run_score(day frame, dt_s=300) -> score; fault if '
                     'score > 0.2322 (faults_summary.json ahu_all threshold_dev, fitted on the platform dev runs)',
                 rows=['det_auc (score)', 'bal_acc', 'fa_rate']),
            dict(name='Majority guess (platform dev majority: no fault)', kind='dumb', rows=['bal_acc', 'fa_rate']),
            dict(name='Majority fault class among diagnosable exam days is NOT used (it would read the exam); '
                      'chance = 1 / number of FaultFinder classes (18)', kind='dumb', rows=['diag_top1']),
        ],
        interval=dict(method='95 % percentile bootstrap over exam AHUs (all days of a drawn AHU move together)',
                      resamples=2000, seed=X.SEED,
                      statistic='model metric minus the best baseline metric on the same resample (best = better '
                                'point value on the full exam)',
                      verdict='win if the interval is entirely on the good side of zero, loss if entirely on the bad '
                              'side, tie if it spans zero; n AHUs and n days stated'),
        scored_once_for=['FaultFinder v1.0', 'any round-2 release'],
    )
    sha = X.write_frozen(SPLIT, doc, check)
    print(json.dumps(doc['n']), json.dumps({k: (v['n_days'], v['n_fault_days']) for k, v in units.items()}))
    print('dropped:', json.dumps(dropped_units))
    print(f'{len(disagree)} schedule/file disagreements; split file {SPLIT} sha256 {sha}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if cmd == 'fetch':
        print(X.fetch_files(FILES, RAW, SOURCE))
    else:
        build(check='--check' in sys.argv)
