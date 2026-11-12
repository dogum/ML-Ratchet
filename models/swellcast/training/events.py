"""Real labelled swell events: IBTrACS storms x NDBC buoys, split into a small dev set (18 storms by seed 2026, see build()) and the exam.

Every (storm, buoy) pair with peak 1-min wind >= 64 kt, 1,000-8,000 km apart over open water, gets a 240-h window centred
on the predicted arrival of 0.06 Hz plus a seeded jitter of +-36 h (the same rule the simulator uses).
Excluded: windows with < 70% data, and 'crowded' pairs (another >= 64 kt storm's window centre at that buoy within 72 h).
'Visible' (decided once, a priori, before any method was run): along the ridge predicted from the TRUE storm, the
energy at 0.045-0.085 Hz is at least 2x (0.3 dex) higher than at the same frequencies 48 h earlier. This uses the truth
but no method's output; it marks events a buoy actually recorded. 'Confirmed' (decided on dev, before any method was scored on the exam): visible AND the buoy's measured mean direction
along that predicted ridge is within 45 deg of the bearing to the storm, so the ridge is this storm's swell, not another
swell that happens to cross the same place. Headline = confirmed; visible and all pairs are also reported."""
import os, sys, json, zlib, numpy as np
import catalog, fetch, realdata as RD, swellsim as SS
ROOT = catalog.ROOT

def jitter(sid, b):
    return np.random.default_rng(zlib.crc32(f'{sid}|{b}'.encode())).uniform(-SS.HALF_H * 0.3, SS.HALF_H * 0.3)   # +-36 h

def ridge_contrast(E, t0_rel, D, tc_grid_times):
    """mean log10 E on the predicted ridge minus 48 h before, 0.045-0.085 Hz; NaN if not measurable"""
    vals = []
    for j, f in enumerate(SS.FGRID):
        if not (0.045 <= f <= 0.085): continue
        ta = t0_rel + D / SS.cg(f) / 3600; i1 = np.searchsorted(tc_grid_times, ta); i0 = np.searchsorted(tc_grid_times, ta - 48)
        if 0 < i1 < len(tc_grid_times) - 1 and 0 < i0:
            a = E[max(0, i1 - 2):i1 + 2, j]; b = E[max(0, i0 - 2):i0 + 2, j]
            if np.isfinite(a).any() and np.isfinite(b).any(): vals.append(np.log10(np.nanmean(a) + 0.01) - np.log10(np.nanmean(b) + 0.01))
    return float(np.mean(vals)) if len(vals) >= 3 else np.nan

def ridge_direction(E, C, S, t0_rel, D, T):
    """excess-energy-weighted mean direction (deg) of the buoy's first moments along the TRUE predicted ridge, 0.045-0.085 Hz"""
    base = np.nanpercentile(np.where(np.isfinite(E), E, np.nan), 25, axis=0); cs = np.zeros(2); n = 0
    for j, f in enumerate(SS.FGRID):
        if not (0.045 <= f <= 0.085): continue
        i = np.searchsorted(T, t0_rel + D / SS.cg(f) / 3600)
        for k in range(max(0, i - 2), min(len(T), i + 2)):
            w = E[k, j] - base[j]
            if np.isfinite(w) and w > 0 and np.isfinite(C[k, j]) and np.isfinite(S[k, j]): cs += w * np.array([C[k, j], S[k, j]]); n += 1
    return float(np.degrees(np.arctan2(cs[1], cs[0])) % 360) if n >= 3 else np.nan

def build():
    C = [c for c in catalog.candidates(min_wind=64, dmin=1000e3, dmax=8000e3) if c['buoy'] in fetch.BUOYS]
    for c in C: c['tc'] = c['t0'] + c['D'] / SS.cg(SS.F_REF) / 3600 + jitter(c['sid'], c['buoy'])
    by_b = {}
    for c in C: by_b.setdefault(c['buoy'], []).append(c['tc'])
    T = (np.arange(SS.NT) * SS.DT_H - SS.HALF_H + SS.DT_H / 2)   # coarse-bin centre times rel. to window centre
    out = []; n_crowd = n_gap = 0
    for c in C:
        others = np.array(by_b[c['buoy']]); c['crowded'] = bool(np.sum(np.abs(others - c['tc']) < 72) > 1)
        (E, Cc, Sc), tc = RD.window(c['buoy'], c['tc']); c['tc_grid'] = tc; c['t0_rel'] = c['t0'] - tc
        c['coverage'] = float(np.isfinite(E[:, 4]).mean()); c['dir_coverage'] = float(np.isfinite(Cc[:, 4]).mean())
        if c['crowded']: n_crowd += 1; continue
        if c['coverage'] < 0.7: n_gap += 1; continue
        c['contrast'] = ridge_contrast(E, c['t0_rel'], c['D'], T); c['visible'] = bool(np.isfinite(c['contrast']) and c['contrast'] >= 0.3)
        c['ridge_dir'] = ridge_direction(E, Cc, Sc, c['t0_rel'], c['D'], T)
        c['confirmed'] = bool(c['visible'] and np.isfinite(c['ridge_dir']) and abs((c['ridge_dir'] - c['dir'] + 180) % 360 - 180) <= 45)
        out.append(c)
    sids = sorted({c['sid'] for c in out}); rng = np.random.default_rng(2026); dev_s = set(rng.choice(sids, 8, replace=False).tolist())
    # that first draw (7 of 8 storms West Pacific) had no confirmed event, so 10 more dev storms are drawn with the same
    # seed, stratified by basin among storms with a confirmed pair (EP 4, NA 3, WP 2, SP 1). The first 8 stay in dev.
    conf = {}
    for c in out:
        if c['confirmed'] and c['sid'] not in dev_s: conf.setdefault(c['basin'], set()).add(c['sid'])
    for b, k in (('EP', 4), ('NA', 3), ('WP', 2), ('SP', 1)):
        pool = sorted(conf.get(b, [])); dev_s |= set(rng.choice(pool, min(k, len(pool)), replace=False).tolist())
    for c in out: c['split'] = 'dev' if c['sid'] in dev_s else 'exam'
    meta = dict(n_candidates=len(C), n_crowded=n_crowd, n_gappy=n_gap, n_kept=len(out), dev_storms=sorted(dev_s))
    json.dump(dict(meta=meta, events=out), open(os.path.join(ROOT, 'data/events.json'), 'w'), indent=0)
    return meta, out

def load(split=None, visible=None):
    """visible: None = all pairs, True = visible, 'confirmed' = visible and the ridge's measured direction within 45 deg of the storm"""
    d = json.load(open(os.path.join(ROOT, 'data/events.json')))['events']
    ok = lambda e: visible is None or (e['confirmed'] if visible == 'confirmed' else e['visible'] == visible)
    return [e for e in d if (split is None or e['split'] == split) and ok(e)]

def windows(evs):
    X = [RD.window(e['buoy'], e['tc'])[0] for e in evs]; return np.stack([np.stack(x) for x in X]) if X else None   # (n, 3, 80, 34)

if __name__ == '__main__':
    meta, out = build(); print(json.dumps(meta))
    import collections
    for sp in ('dev', 'exam'):
        ev = [e for e in out if e['split'] == sp]; vis = [e for e in ev if e['visible']]
        print(sp, len(ev), 'visible', len(vis), 'confirmed', sum(e['confirmed'] for e in ev), collections.Counter(e['basin'] for e in ev if e['confirmed']), 'storms', len({e['sid'] for e in ev}), 'by basin', collections.Counter(e['basin'] for e in vis), 'with dir', sum(e['dir_coverage'] > 0.7 for e in vis))
    v = [e for e in out if e['visible']]; print('D km quartiles visible', np.percentile([e['D'] / 1e3 for e in v], [10, 25, 50, 75, 90]).round())
    print('contrast quartiles all', np.nanpercentile([e['contrast'] for e in out], [10, 25, 50, 75, 90]).round(2))
