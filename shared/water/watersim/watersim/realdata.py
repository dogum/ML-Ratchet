"""Bridge to agent B's real-data package (water/waterdata) with the split rules enforced.

Rules (water/data/proc/splits.json, 'rules'): simulated training may use real FORCING of any basin from non-exam
periods only (before 1989-10-01 and 1999-10-01..2015-09-30) and real ATTRIBUTES only from dev basins. Exam basins'
streamflow is never touched here. Every helper below refuses an exam window.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water
WATER = os.environ.get('WATERSHED_ROOT') or os.environ.get('ML_RATCHET_WATER') or _CODE  # data root
SPLITS = os.path.join(WATER, 'data', 'proc', 'splits.json')


def _wd():
    if WATER not in sys.path:
        sys.path.insert(0, WATER)
    import waterdata
    return waterdata


def splits():
    with open(SPLITS) as f:
        return json.load(f)


def exam_windows():
    t = splits()['temporal']
    return [(t['validation']['start'], t['validation']['end']), (t['recent_exam']['start'], t['recent_exam']['end'])]


def free_windows(first='1951-01-01', last='2015-09-30'):
    """Non-exam forcing windows [(start, end), ...] (inclusive dates)."""
    ex = sorted(exam_windows())
    out, cur = [], pd.Timestamp(first)
    for a, b in ex:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a > cur:
            out.append((cur, min(a - pd.Timedelta(days=1), pd.Timestamp(last))))
        cur = max(cur, b + pd.Timedelta(days=1))
    if cur <= pd.Timestamp(last):
        out.append((cur, pd.Timestamp(last)))
    return [(str(a.date()), str(b.date())) for a, b in out if b > a]


def check_window(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for a, b in exam_windows():
        if not (e < pd.Timestamp(a) or s > pd.Timestamp(b)):
            raise PermissionError(f'{start}..{end} overlaps the exam period {a}..{b}; real forcing for simulation '
                                  f'training must come from non-exam windows {free_windows()}')


def nonexam_forcing(basin_id, start, end):
    """(forcing DataFrame, meta) for any basin, restricted to a non-exam window."""
    check_window(start, end)
    wd = _wd()
    return wd.forcing(basin_id, start, end), wd.forcing_meta(basin_id)


def dev_specs(region='camels'):
    """Contract specs of the dev basins (the only real attributes simulation may borrow)."""
    wd = _wd()
    return [wd.spec(b) for b in wd.basins(region, split='dev')]


def donor_table(region='camels'):
    """Dev-basin attributes as a DataFrame in spec units (slope m/km, conductivity cm/h, mwc mm), for
    generate_basins(prior='camels', attributes=...)."""
    rows = {}
    for s in dev_specs(region):
        r = {k: s.get(k) for k in ('area_km2', 'lat', 'lon', 'elev_mean_m', 'p_mean', 'aridity', 'frac_snow',
                                   'forest_frac', 'soil_porosity', 'carbonate_frac', 'urban_frac', 'irrigation_frac',
                                   'reservoir_storage_frac', 'soil_depth_m', 'max_water_content_mm', 'lai')}
        sm = s.get('slope_mean')
        if sm is not None and str(s.get('slope_unit', '')).startswith('deg'):
            sm = math.tan(math.radians(float(sm))) * 1000.0
        r['slope_mean'] = sm
        kc = s.get('soil_conductivity')
        r['soil_conductivity'] = None if kc is None else (kc / 10.0 if s.get('soil_conductivity_unit') == 'mm/h' else kc)
        rows[s['id']] = r
    return pd.DataFrame.from_dict(rows, orient='index').astype(float)


def random_forcing(rng, years=10, region='camels', basin_id=None, spinup_years=1):
    """A random non-exam forcing segment of `years` (+ spin-up) from any basin of the region.
    Returns (df, meta, info)."""
    wd = _wd()
    rng = np.random.default_rng(rng)
    ids = wd.basins(region)
    b = basin_id or ids[int(rng.integers(len(ids)))]
    span = years + spinup_years
    wins = [(a, e) for a, e in free_windows() if (pd.Timestamp(e) - pd.Timestamp(a)).days >= span * 365]
    if not wins:
        raise ValueError(f'no non-exam window of {span} years')
    w = [(pd.Timestamp(e) - pd.Timestamp(a)).days - span * 365 for a, e in wins]
    k = int(rng.choice(len(wins), p=np.array(w) / max(sum(w), 1) if sum(w) > 0 else None))
    a, e = pd.Timestamp(wins[k][0]), pd.Timestamp(wins[k][1])
    off = int(rng.integers(0, max((e - a).days - span * 365, 0) + 1))
    s0 = (a + pd.Timedelta(days=off)).normalize()
    s1 = s0 + pd.DateOffset(years=span) - pd.Timedelta(days=1)
    df, meta = nonexam_forcing(b, str(s0.date()), str(s1.date()))
    return df, meta, {'basin_id': b, 'start': str(s0.date()), 'end': str(s1.date()),
                      'sim_start': str((s0 + pd.DateOffset(years=spinup_years)).date())}
