"""Shared helpers for FaultFinder's exam-2 freeze scripts (exam2_wang.py, exam2_nehasil.py), 17 Nov 2026.

Download with a plain user agent (resumable), record sha256, apply the day quality control of the plan, hash per unit.
Scores nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date

import numpy as np
import pandas as pd

SEED = 20261117
UA = 'Mozilla/5.0 (research data client)'
# contract channels kept in the processed unit-day tables (FaultFinder reads these names)
CONTRACT = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone', 'damper_oa', 'valve_cc', 'valve_hc', 'fan_sa_speed',
            'fan_sa_power_kW', 'fan_sa_status', 'occ']
TEMPS = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone']


def file_hash(path, algo='sha256'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch_files(files, raw, source):
    """files: [dict(name, url, size?, md5?/sha256?)]. Resumable curl; verifies published hashes; writes SOURCES.json."""
    os.makedirs(raw, exist_ok=True)
    rec = []
    for f in files:
        p = os.path.join(raw, f['name'])
        if not (os.path.exists(p) and (f.get('size') is None or os.path.getsize(p) == f['size'])):
            subprocess.run(['curl', '-L', '-sS', '--fail', '--retry', '5', '-A', UA, '-C', '-', '-o', p, f['url']],
                           check=True)
        r = dict(name=f['name'], url=f['url'], size=os.path.getsize(p), sha256=file_hash(p),
                 downloaded=date.today().isoformat())
        if f.get('md5'):
            r['md5'] = file_hash(p, 'md5')
            if r['md5'] != f['md5']:
                raise SystemExit(f"{f['name']}: md5 {r['md5']} != published {f['md5']}")
        if f.get('sha256') and r['sha256'] != f['sha256']:
            raise SystemExit(f"{f['name']}: sha256 {r['sha256']} != published {f['sha256']}")
        rec.append(r)
    json.dump(dict(source, files=rec), open(os.path.join(raw, 'SOURCES.json'), 'w'), indent=1)
    return rec


def clean_channels(d):
    """Plan QC on a contract frame (in place): impossible temperatures -> NaN, T_sa_sp < 1 -> NaN, actuators clipped
    as faultfinder.raw_day does. Flatline-per-day is applied per day in day_qc."""
    for c in TEMPS + ['T_sa_sp']:
        if c in d:
            x = d[c].astype(float)
            d[c] = x.where((x >= -45) & (x <= 70))
    if 'T_sa_sp' in d:
        d['T_sa_sp'] = d['T_sa_sp'].where(d['T_sa_sp'] >= 1.0)
    return d


def day_qc(g, slots_needed, slot_s):
    """g: one unit-day of a contract frame (time_local column). Returns (ok, reason, g with flatlined temps NaN)."""
    g = g.copy()
    for c in TEMPS:
        if c in g:
            v = g[c].to_numpy(float)
            vv = v[np.isfinite(v)]
            if len(vv) > 10 and vv.max() - vv.min() < 1e-3:
                g[c] = np.nan
    if 'T_sa' not in g or g['T_sa'].notna().sum() == 0:
        return False, 'no T_sa', g
    t = pd.to_datetime(g['time_local'])
    slot = ((t - t.dt.normalize()).dt.total_seconds() // slot_s).astype(int)
    n_slots = slot[g['T_sa'].notna().to_numpy()].nunique()
    if n_slots < slots_needed:
        return False, f'T_sa in < {slots_needed} slots', g
    dt_h = slot_s / 3600.0
    if 'fan_sa_status' in g and g['fan_sa_status'].notna().any():
        on_h = (g['fan_sa_status'] > 0.5).sum() * dt_h
    elif 'fan_sa_speed' in g and g['fan_sa_speed'].notna().any():
        on_h = (g['fan_sa_speed'] > 0.05).sum() * dt_h
    else:   # no fan point: T_sa varying counts as running
        on_h = (g['T_sa'].diff().abs() > 0.05).sum() * dt_h
    if on_h < 1.0:
        return False, 'fan off all day', g
    return True, 'ok', g


def hash_frame(df, cols):
    """sha256 of a unit's processed rows: time_local ISO + the given columns (6 decimals, 'nan' for missing)."""
    h = hashlib.sha256()
    t = pd.to_datetime(df['time_local']).dt.strftime('%Y-%m-%dT%H:%M:%S').to_numpy()
    X = df[cols].to_numpy(float)
    for i in range(len(df)):
        h.update((t[i] + ',' + ','.join('nan' if not np.isfinite(v) else f'{v:.6f}' for v in X[i]) + '\n').encode())
    return h.hexdigest()


def write_frozen(path, doc, check=False):
    """Write the split/spec JSON once. doc['split_sha256'] = sha256 of the canonical units block. Returns file sha256."""
    doc['split_sha256'] = hashlib.sha256(json.dumps(doc['units'], sort_keys=True).encode('utf-8')).hexdigest()
    if os.path.exists(path):
        old = json.load(open(path, encoding='utf-8'))
        same = old.get('split_sha256') == doc['split_sha256']
        if check:
            print('rebuild matches frozen split' if same else 'REBUILD DIFFERS from frozen split')
            return file_hash(path)
        raise SystemExit(f'{path} exists and is frozen; not rewriting it (use --check).')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(doc, f, indent=1, default=str)
        f.write('\n')
    return file_hash(path)
