"""Data for the showpiece page: the released graph network (float32, base64), ak135 P table, land outlines, 24 exam
events for the explorer (a fixed draw within depth strata from the 400-event subset: 12 shallow, 12 deep; seed 3), the
exam tables, the ratchet ladder and the twin test. Also writes parity cases for the JS == Python check."""
import sys, json, base64, numpy as np
from _paths import H; sys.path.insert(0, H + 'src')
import train2 as T2
z = np.load(H + 'release/quakenet2_weights.npz'); meta = json.loads(str(z['meta']))
W = {k: dict(b64=base64.b64encode(z[k].astype('<f4').tobytes()).decode(), s=list(z[k].shape)) for k in z.files if k != 'meta'}
tt = np.load(H + 'data/tt_ak135.npz'); TT = dict(depths=tt['depths'].tolist(), P=np.round(tt['P'], 2).tolist())
D = T2.load(H + 'data/exam_samples.npz'); rows = json.load(open(H + 'data/real_nodes.json')); SP = json.load(open(H + 'splits.json'))
preds = json.load(open(H + 'data/exam_preds.json')); ER = json.load(open(H + 'exam_results.json'))
by = {}
for i, p in enumerate(preds): by.setdefault(p['id'], {})[p['tag']] = (i, p)
cand = [k for k in sorted(by) if 'all' in by[k] and by[k]['all'][1]['K'] >= 6]
rng = np.random.default_rng(3); sh = [k for k in cand if SP['events'][k]['depth'] <= 70]; dp = [k for k in cand if SP['events'][k]['depth'] > 70]
pick = sorted(rng.choice(sh, 12, replace=False).tolist() + rng.choice(dp, 12, replace=False).tolist(), key=lambda k: SP['events'][k]['time'])
EV = []
for k in pick:
    i, p = by[k]['all']; idx = p['nodes']; e = SP['events'][k]
    st = []
    for n in idx:
        r = rows[n]; pz = D['n_pz'][n].astype(np.float32); sc = float(np.abs(pz).max() or 1)
        st.append(dict(code=r['station'], lat=r['slat'], lon=r['slon'], **{f: round(float(D['n_r_' + f][n]), 4) for f in ('mw', 'mw_sig', 'dist', 'dist_sig', 'logz', 'logz_sig', 'baz', 'kappa')},
                       ls=round(float(D['n_ls'][n]), 4), peak=round(float(D['n_peak'][n]), 3), tpick=round(float(D['n_tpick'][n]), 2),
                       pz=np.round(pz / sc * 127).astype(int).tolist(), pzs=sc))
    base = {t: dict(v12=by[k][t][1]['v12'], tt=by[k][t][1]['tt'], centroid=by[k][t][1]['centroid'], v2=by[k][t][1]['v2']) for t in by[k]}
    EV.append(dict(id=k, place=e['place'], time=e['time'], lat=e['lat'], lon=e['lon'], depth=e['depth'], mw=e['mw'], magType=e['magType'], st=st, base=base))
RT = json.load(open(H + 'ratchet.json')); TW = json.load(open(H + 'data/twin_compare.json'))
out = dict(meta=meta, w=W, tt=TT, events=EV, exam=ER, ratchet=[{k: r[k] for k in ('rung', 'ckpt', 'change', 'frozen', 'decision', 'why', 'objective', 'dev', 'skill')} for r in RT], twin=TW,
           sigma=json.load(open(H + 'data/sigma_factors.json')), physics=json.load(open(H + 'physics_check.json')))
out['how'] = [
  'Physics: QuakeNet v1.2\'s simulator (Syngine ak135f Green\'s functions, physics v7 with layering), run per station for one shared source: same epicentre, depth, Mw, mechanism, source duration and ocean layer at every station.',
  'Exact distances (the nearest 1° Green\'s function stretched so S−P matches ak135), absolute timing (ak135 P time plus a station residual), and real quiet-time noise from the same station.',
  'Each simulated record is read by the frozen v1.2 network. 10,000 simulated events at 2–10 real GSN stations give 26,683 training graphs; the frozen set has 2,000 events (3,785 graphs).',
  'The graph network (115k weights, 3 rounds of message passing) predicts corrections to the classical combination of v1.2\'s readings, with a bivariate error ellipse and Gaussian bars.',
  'Checks: ak135 table vs direct TauP within 0.08 s and vs iasp91 within 0.24 s; numpy vs torch 6e-6; this page\'s JavaScript vs Python within 0.14 km.',
  'Dev twins (40 dev quakes re-simulated from their USGS moment tensors) measured how v1.2 reads real records differently from simulated ones; physics v2 and v3 build that into the simulated readings.']
out['limits'] = [
  'Magnitude: QuakeNet v2 loses to simply averaging v1.2\'s per-station Mw (0.097 against 0.089 with all stations; 0.114 against 0.109 with 2).',
  'Deep quakes (> 70 km) are read too shallow: median depth error 38 km against 28 km for the classical average with all stations. The dev twins showed it (deep depth ratio 0.74).',
  'Two stations still miss by a median 427 km: most of the information comes from P arrival times, which need several stations.',
  'Long-period records only (1 sample/s), 25–95° from the quake, M ≥ 5.5. The P pick searches ±30 s around a guess; in the exam that guess came from the catalogue time, as in v1.2.',
  'Record start times were rounded to the nearest second when the records were cut; that adds up to ±0.5 s to every pick.',
  'Epicentre ellipses cover 62–68% of exam truths for a nominal 68%; depth bars 60–61%.']
out['ratchetNote'] = 'A rung is kept if it beats the incumbent by more than 0.01 on the frozen simulated set (mean over epicentre, log depth, Mw and origin time of 1 − error / rung-0 error) and the 40 real dev quakes get no more than 5% worse. When the physics changed (val2, val3), every rung was re-scored and every decision re-taken; the table shows the final set, val3. Earlier scores are in ratchet.json.'
out['twinNote'] = 'On twins, v1.2 combined misses by a median 145 km; on the real records of the same quakes, 407 km. v1.2 reads real records about 1.5° farther and real shallow quakes 2.4 times too deep, and real P picks are tighter than simulated ones. Physics v2 and v3 put these measured differences into the simulated readings, so the released network reads real records better (192 km, shallow depth ratio 1.15) and its own twins worse, as intended.'
out['ratchet'] = [r for r in out['ratchet'] if r['frozen'] == 'val3']
json.dump(out, open(H + 'web/data.json', 'w'), separators=(',', ':')); print('events', len(EV), 'bytes', len(json.dumps(out)))
