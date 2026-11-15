"""Forecast-weather exam rows from the shipped weights (numpy, no torch). Added in the repository (14 Nov 2026).

    mlratchet data grid --only forecastwx          # Open-Meteo previous runs -> shared/grid/data/raw/openmeteo/
    python rescore_forecast.py release2024         # the release's exam cache (2024), to reproduce its row
    python rescore_forecast.py full                # the shipped fetch list: 2024-2025 (and 2022 T), run once

Needs what rescore_release.py needs (work/data/wx, prior, calib, banks/real_dev.npz). For each scope the listed cache
files are copied to work/data/openmeteo_<scope>/, the exam bank is rebuilt by realbank.build('exam') with
real.Forecasts reading that folder, then exam.main() runs unchanged with rescore_release.py's numpy stand-in for
evaluate.py. Outputs go to work/data/forecast_<scope>/ (banks/, exam/); the actual-weather bank in work/data/banks/ is
not touched.

'release2024' is the cache the release had when its exam ran, inferred from its exam output: forecast-mode wind rows
exist only for ERCO (so no other BA had a wind point), and the forecast-mode physics-prior error per BA matches only
one set of population points: all 8 in ERCO, CISO, PJM, MISO, SWPP and ISNE (ERCO 6.497 %; 6.513 % with the top 4),
the top 4 plus NYIS_pop_0 in NYIS (5.2997 %), the top 4 in BPAT (5.389 %). That fits an earlier pass that fetched
population points in file order (pid 0-7, BA by BA) until the rate limit stopped it after NYIS_pop_0, then the top-4
pass. ERCO's top wind point only: with it the released ERCO forecast-mode wind row (31.24 % vs physics 47.86 %)
reproduces exactly; with 2 or 3 points it is 27.5 or 26.7 %.
'full' is fetch_openmeteo.py's list as shipped: top 4 population, 3 wind, 3 solar points per BA for 2024 and 2025,
top 4 population points for 2022 (temperature only; the archive has no humidity or radiation forecasts before 2024,
so 2022-2023 rows cannot enter the exam's forecast mode).
"""
import json
import os
import shutil
import sys

import numpy as np

from common import DATA, GRID, BAS, jdump
import rescore_release as RR       # installs the numpy evaluate shim and imports exam
import realbank

exam = RR.exam
RAW_OM = os.path.join(GRID, 'data', 'raw', 'openmeteo')


OM_WIND_ERCO = int(os.environ.get('OM_WIND_ERCO', 1))   # 1 matches the released wind row exactly; 2 or 3 do not
TOPK = {'pop': 4, 'wind': 3, 'solar': 3}
TAG = os.environ.get('SCOPE_TAG', '')          # suffix for side runs (diagnostics)


def scope_files(scope):
    pts = json.load(open(os.path.join(GRID, 'data', 'proc', 'weather', 'points.json')))

    def top(ba, kind, k):
        return [p['pid'] for p in sorted([p for p in pts[ba]['points'] if p['kind'] == kind], key=lambda p: -p['w'])[:k]]

    if scope == 'release2024':
        files = [(pid, 2024) for ba in BAS for pid in top(ba, 'pop', 4 if ba in ('NYIS', 'BPAT') else 8)]
        files += [('NYIS_pop_0', 2024)]
        files += [(pid, 2024) for pid in top('ERCO', 'wind', OM_WIND_ERCO)]
    else:
        files = [(pid, y) for y in (2024, 2025) for ba in BAS for kind in TOPK for pid in top(ba, kind, TOPK[kind])]
        files += [(pid, 2022) for ba in BAS for pid in top(ba, 'pop', 4)]
    return files


def scope_dir(scope):
    """work/data/openmeteo_<scope>/: copies of the scope's cache files."""
    d = f'{DATA}/openmeteo_{scope}{TAG}'
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    missing = []
    for pid, y in scope_files(scope):
        src = os.path.join(RAW_OM, f'{pid}_{y}.json')
        if os.path.exists(src):
            shutil.copy2(src, d)
        else:
            missing.append(f'{pid}_{y}')
    if missing:
        print(f'{scope}: {len(missing)} cache files missing (run `mlratchet data grid --only forecastwx`):',
              ', '.join(missing[:8]), '...' if len(missing) > 8 else '')
        if scope == 'release2024' and not os.environ.get('ALLOW_MISSING'):
            sys.exit(1)
    return d


def run(scope):
    os.environ['DC_OPENMETEO'] = scope_dir(scope)
    sub = f'{DATA}/forecast_{scope}{TAG}'
    os.makedirs(f'{sub}/banks', exist_ok=True)
    os.makedirs(f'{sub}/exam', exist_ok=True)
    shutil.copy2(f'{DATA}/banks/real_dev.npz', f'{sub}/banks/real_dev.npz')
    realbank.DATA = sub                  # realbank writes {DATA}/banks/real_exam.npz
    realbank.build('exam')
    exam.DATA = sub                      # exam reads {DATA}/banks/, writes {DATA}/exam/
    exam.main('release')
    summary(sub, scope)


def summary(sub, scope):
    new = json.load(open(f'{sub}/exam/release.json'))
    old = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exam_r8_wide.json')))
    rows = []
    for mode in ('forecast', 'actual_same_days_as_forecast'):
        for out in ('demand', 'wind', 'solar'):
            a = new['overall'].get(mode, {}).get(out)
            b = old['overall'][mode][out]
            ref = 'ba_day_ahead' if out == 'demand' else 'physics_prior'
            rows.append(dict(row=f'pooled {mode} {out}', released=[b['model'], b[ref], b['hours']],
                             rescored=None if a is None else [a['model'], a[ref], a['hours']]))
    for ba in BAS:
        t = new['table'].get(ba, {})
        for y in ('2024', '2025', 'all'):
            f = t.get(y, {}).get('forecast')
            if f is None:
                continue
            o = old['table'].get(ba, {}).get(y, {}).get('forecast')
            rows.append(dict(row=f'{ba} {y} forecast demand', released=None if o is None else
                             [o['demand']['model'], o['demand']['ba_day_ahead'], o['demand']['hours']],
                             rescored=[f['demand']['model'], f['demand']['ba_day_ahead'], f['demand']['hours']]))
    jdump(rows, f'{sub}/exam/forecast_vs_released.json')
    for r in rows:
        fmt = lambda v: '-' if v is None else f'{v[0]:.2f} vs {v[1]:.2f} ({v[2]} h)'
        print(f"{scope}: {r['row']:44s} released {fmt(r['released']):28s} rescored {fmt(r['rescored'])}")


if __name__ == '__main__':
    run(sys.argv[1])
