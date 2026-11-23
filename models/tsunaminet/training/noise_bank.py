"""Real DART noise windows. Every 1-minute (event-mode, T=2) run in the Pacific DART files 2006-2025 that starts more
than 2 days before / 3 days after every NGDC tsunami event (any cause, 2005-2026) becomes a pseudo-event: origin t0' =
run start minus 1-8 min (seeded), then the same detiding and minute gridding as real events (dart.detided_minutes).
Windows with < 3 h of minutes are dropped. Split 80/20 by station-year (seeded) into train and frozen-set noise."""
import os, json, glob, datetime as dt, numpy as np
from _paths import DATA, BANK
import dart
rng = np.random.default_rng(20261122)
ev = json.load(open(os.path.join(DATA, 'ngdc_all_events_2005_2026.json')))['items']
tev = np.array([dt.datetime(e['year'], e['month'] or 1, e['day'] or 1, e.get('hour') or 0, e.get('minute') or 0, tzinfo=dt.timezone.utc).timestamp() for e in ev])
files = sorted(glob.glob(os.path.join(DATA, 'dart', '*.txt.gz')))
ETA, MASK, META = [], [], []
for fn in files:
    b = os.path.basename(fn); st, yr = b[:5], int(b[6:10])
    r = dart.read_year(st, yr)
    if r is None: continue
    T, H, Y = r
    t2 = T[Y == 2]
    if len(t2) < 60: continue
    starts = t2[np.r_[True, np.diff(t2) > 3600]]
    for s in starts:
        if np.any((s - tev > -3 * 86400) & (s - tev < 2 * 86400)): continue
        t0 = s - 60 * rng.uniform(1, 8)
        out = dart.detided_minutes(st, t0)
        if out is None: continue
        eta, mask, info = out
        if mask[:361].sum() < 180: continue
        if np.abs(eta[mask]).max() > 1.0: continue          # gross failures of the tide fit
        ETA.append(eta.astype(np.float16)); MASK.append(mask); META.append((st, yr, float(t0)))
    print(b, len(ETA), flush=True)
ETA = np.stack(ETA); MASK = np.stack(MASK)
sy = sorted(set((m[0], m[1]) for m in META)); perm = rng.permutation(len(sy)); val_sy = set(sy[i] for i in perm[: len(sy) // 5])
is_val = np.array([(m[0], m[1]) in val_sy for m in META])
np.savez_compressed(os.path.join(BANK, 'noise_windows.npz'), eta=ETA, mask=MASK, is_val=is_val,
                    station=np.array([m[0] for m in META]), year=np.array([m[1] for m in META]), t0=np.array([m[2] for m in META]))
print('windows', len(ETA), 'val', is_val.sum())
