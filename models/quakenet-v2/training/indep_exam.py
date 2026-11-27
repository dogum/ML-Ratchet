"""Supplementary exam with independent P picks (27 Nov 2026; audit queue). The released model, unchanged; it chooses nothing.

Input: the continuous blocks written by fetch_cont.py (from 10 minutes before the UTC hour of the origin to 2 hours after;
the reader is not told where in the block P is). For each record the P guess is picked from the record alone:
  vertical, 0.05-0.3 Hz (pipeline.SOS_PICK), recursive STA/LTA 5 s / 60 s (pipeline.sta_lta), searched from the start
  of the hour to 75 minutes after it; take the largest ratio, then walk back to the first sample of that trigger run
  above 3.0 (the trigger level pipeline.pick uses). That sample is p_guess; the release then refines it within +-30 s
  exactly as it always does.
Second reader, 'associated' (a network associator, no catalogue input): every record's trigger onsets (STA/LTA runs above
4.0, the 5 strongest), then a grid search over 12,000 epicentres (depth 33 km) and origin times for the source on which
most stations have a trigger within +-6 s of the ak135 P time; p_guess = that trigger, or the predicted P where a station
has none. The settings (level 4.0, 5 triggers, +-6 s) were chosen on the 40 dev events only (sweep: level 3/4 x 5/10/30
triggers x 6/12 s; best 41% of dev picks within 10 s of the catalogue P) before the exam was run; dev may be run as often
as needed.
Control on the same continuous records: p_guess = the catalogue-predicted ak135 P (what the released exam did).
Graded against USGS on the 400-event multi-station subset, first 3, 5 and all (<= 10) usable stations.
Writes work/indep_exam_dev.json (dev) or work/indep_exam.json (exam; refuses to run twice)."""
import os, sys, json, glob, time, numpy as np
from scipy.signal import sosfiltfilt
from _paths import H, SRC
sys.path.insert(0, os.path.dirname(SRC))
import quakenet2 as QN, pipeline as PL
PRE, SEARCH = 600, 75 * 60

def pick_indep(z, thr=3.0):
    zf = sosfiltfilt(PL.SOS_PICK, z); c = PL.sta_lta(zf, 5, 60); lo, hi = PRE, min(len(z) - 1, PRE + SEARCH)
    m = lo + int(np.argmax(c[lo:hi])); peak = float(c[m])
    if peak < thr: return m, peak
    i = m
    while i > lo and c[i - 1] >= thr: i -= 1
    return i, peak

def triggers(z, thr=4.0, gap=30, top=5):
    """all trigger onsets in the search span: runs of STA/LTA above thr, onset = first sample of the run, ranked by peak"""
    zf = sosfiltfilt(PL.SOS_PICK, z); c = PL.sta_lta(zf, 5, 60); lo, hi = PRE, min(len(z) - 1, PRE + SEARCH); a = c[lo:hi] >= thr
    on = np.nonzero(a & ~np.r_[False, a[:-1]])[0]; off = np.nonzero(a & ~np.r_[a[1:], False])[0]; T = []
    for i, j in zip(on, off):
        pk = float(c[lo + i:lo + j + 1].max())
        if T and lo + i - T[-1][0] < gap: T[-1] = (T[-1][0], max(T[-1][1], pk)); continue
        T.append((lo + i, pk))
    T.sort(key=lambda t: -t[1]); return T[:top]

def _sphere(n=12000):
    i = np.arange(n) + 0.5; lat = np.degrees(np.arcsin(1 - 2 * i / n)); lon = (np.degrees(np.pi * (1 + 5 ** 0.5) * i) + 180) % 360 - 180; return lat, lon
GLAT, GLON = _sphere()

def associate(st, win=6.0, depth=33.0):
    """grid-search associator: the source cell and origin time on which most stations have a trigger within +-win s of
    the ak135 P time (depth 33 km). Ties broken by the summed STA/LTA peaks. Returns per-station p_guess (sample): the
    matched trigger, or the predicted P where the station has none. Uses no catalogue information."""
    K = len(st); D = np.stack([QN.gc_dist(GLAT, GLON, s['lat'], s['lon']) for s in st]); ok = (D >= 20) & (D <= 100); TT = QN.tt('P', D, depth)
    best = (-1, -1.0, None, None)
    for k, s in enumerate(st):              # candidate origins implied by each trigger of each station
        for t, pk in s['trig'][:10]:
            o = s['t_start'] + t - TT[k]; n = np.zeros(len(GLAT)); w = np.zeros(len(GLAT))
            for j, r in enumerate(st):
                pred = o + TT[j] - r['t_start']; tr = np.array([a for a, _ in r['trig']] or [-1e9]); pkj = np.array([b for _, b in r['trig']] or [0.0])
                dd = np.abs(tr[None, :] - pred[:, None]); hit = (dd.min(1) <= win) & ok[j]; n += hit; w += np.where(hit, pkj[dd.argmin(1)], 0)
            g = int(np.lexsort((w, n))[-1])
            if (n[g], w[g]) > best[:2]: best = (n[g], w[g], g, o[g])
    n, w, g, o = best; out = []
    for j, r in enumerate(st):
        pred = o + TT[j, g] - r['t_start']; tr = [a for a, _ in r['trig'] if abs(a - pred) <= win]
        out.append(int(min(tr, key=lambda a: abs(a - pred))) if tr else int(round(pred)))
    return out, dict(n_assoc=int(n), lat=float(GLAT[g]), lon=float(GLON[g]))

def records(keys, SP, order):
    out = {}
    for k in keys:
        e = SP['events'][k]; R = []
        for s in order.get(k, e['stations']):
            f = H + f'data/wf_cont/ev_{k}_{s}.npz'
            if not os.path.exists(f): continue
            z = np.load(f); x = z['x'].astype(np.float64)
            if x.shape[1] < 7800 or not np.isfinite(x).all() or (x.std(1) == 0).any(): continue
            R.append((s, x, json.loads(str(z['meta']))))
            if len(R) >= 10: break
        if len(R) >= 3: out[k] = R
    return out

def errs(P, e):
    d = float(QN.gc_dist(P['lat'], P['lon'], e['lat'], e['lon'])) * QN.KM; az = np.radians(QN.azimuth(P['lat'], P['lon'], e['lat'], e['lon']))
    v = np.array([d * np.cos(az), d * np.sin(az)]); m2 = float(v @ np.linalg.solve(P['cov_km2'], v)); lz = np.log10(max(P['depth'], 1.0)) - np.log10(max(e['depth'], 1.0))
    return dict(epi_km=d, epi_m2=m2, depth_km=abs(P['depth'] - e['depth']), logz=abs(lz), z_ratio=abs(lz) / P['logz_sig'], mw=abs(P['mw'] - e['mw']),
                mw_ratio=abs(P['mw'] - e['mw']) / P['mw_sig'], t0=abs(P['t0'] - e['time']), t0_ratio=abs(P['t0'] - e['time']) / P['t0_sig'])

def summ(L):
    if not L: return None
    a = {k: np.array([r[k] for r in L]) for k in L[0]}
    return dict(n=len(L), epi_med_km=float(np.median(a['epi_km'])), epi_mean_km=float(np.mean(a['epi_km'])), depth_med_km=float(np.median(a['depth_km'])),
                mw_mae=float(np.mean(a['mw'])), t0_med_s=float(np.median(a['t0'])), cov_epi=float(np.mean(a['epi_m2'] < QN.CHI2_68)),
                cov_z=float(np.mean(a['z_ratio'] < 1)), cov_mw=float(np.mean(a['mw_ratio'] < 1)), cov_t0=float(np.mean(a['t0_ratio'] < 1)))

def run(keys, SP, order):
    net = QN.QuakeNet2(); RC = records(keys, SP, order); res = {m: {t: [] for t in ('3', '5', 'all')} for m in ('independent', 'associated', 'catalogue')}; picks = []; apicks = []; per = []
    for k, R in RC.items():
        e = SP['events'][k]; st = {'independent': [], 'associated': [], 'catalogue': []}
        A = [dict(lat=meta['lat'], lon=meta['lon'], t_start=meta['t_start'], trig=triggers(x[0])) for s, x, meta in R]; pa, ainfo = associate(A)
        for (s, x, meta), p in zip(R, pa):
            st['associated'].append(dict(zne=x, lat=meta['lat'], lon=meta['lon'], t_start=meta['t_start'], p_guess=p))
            apicks.append(float(meta['t_start'] + p - meta['grading_p_time']))
        for s, x, meta in R:
            pi, peak = pick_indep(x[0]); pc = int(round(meta['grading_p_time'] - meta['t_start']))
            picks.append(dict(event=k, station=s, err_s=float(meta['t_start'] + pi - meta['grading_p_time']), peak=peak))
            for mode, pg in (('independent', pi), ('catalogue', pc)): st[mode].append(dict(zne=x, lat=meta['lat'], lon=meta['lon'], t_start=meta['t_start'], p_guess=pg))
        row = dict(id=k, K=len(R), assoc=ainfo)
        for mode in st:
            for tag, N in (('3', 3), ('5', 5), ('all', len(R))):
                if N > len(R): continue
                P = net.locate(st[mode][:N]); E = errs(P, e); res[mode][tag].append(E)
                if tag == 'all': row[mode] = dict(lat=P['lat'], lon=P['lon'], depth=P['depth'], mw=P['mw'], epi_km=E['epi_km'])
        per.append(row)
    pe = np.abs([p['err_s'] for p in picks])
    out = dict(n_events=len(RC), n_records=len(picks), picks=dict(median_abs_err_s=float(np.median(pe)), within_10s=float(np.mean(pe <= 10)), within_30s=float(np.mean(pe <= 30)),
               within_60s=float(np.mean(pe <= 60)), over_300s=float(np.mean(pe > 300))),
               picks_associated=dict(median_abs_err_s=float(np.median(np.abs(apicks))), within_10s=float(np.mean(np.abs(apicks) <= 10)), within_30s=float(np.mean(np.abs(apicks) <= 30)),
               within_60s=float(np.mean(np.abs(apicks) <= 60)), over_300s=float(np.mean(np.abs(apicks) > 300))),
               results={m: {t: summ(v) for t, v in d.items()} for m, d in res.items()}, per_event=per, per_pick=picks)
    return out

if __name__ == '__main__':
    which = sys.argv[1]; SP = json.load(open(H + 'splits.json')); order = json.load(open(H + 'data/station_order.json'))
    fn = H + ('indep_exam.json' if which == 'exam' else 'indep_exam_dev.json')
    if which == 'exam' and os.path.exists(fn): sys.exit('The independent-pick exam has already been run (indep_exam.json exists). It runs once.')
    t = time.time(); keys = SP['exam_multi'] if which == 'exam' else SP['dev']
    out = run(keys, SP, order); out.update(split=which, run_utc=time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), seconds=round(time.time() - t))
    json.dump(out, open(fn, 'w'), indent=1)
    print(json.dumps(dict(n_events=out['n_events'], n_records=out['n_records'], picks=out['picks'], picks_associated=out['picks_associated']))); [print(m, t, {a: round(b, 3) for a, b in v.items()}) for m, d in out['results'].items() for t, v in d.items() if v]
