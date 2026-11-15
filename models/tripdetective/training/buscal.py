"""TripDetective v2.1 bus calibration ladder (15 Nov 2026): maps from the bus network's reading (and the road-load fit)
to bus mass, fitted on ZTBus DEV gate legs only and judged by 5-fold cross-validation over dev missions.

Why. On the 757 dev gate legs the bus network (pb1, released in v2.0) reads +0.9 t heavy and follows passengers
weakly overall (r 0.25). Split by mission (one bus, one day): between missions its mean does not follow the mission's
mean passengers at all (r 0.05), within a mission it does (r 0.47, slope 0.65). So the day's level carries no usable
signal and the leg-to-leg differences do. The truth's level is a convention (19 t + 70 kg per passenger).

Rungs (each a map; coefficients fitted on 4/5 of the dev gate missions, scored on the other 1/5, folds by a hash of
the mission id; the final coefficients are fitted on all dev gate legs):
  bc1  level:       m_hat = k x m                                             (v2.0's exam note: the dev-fitted level)
  bc2  line:        m_hat = a + b x m
  bc3  day-relative: m_hat = a + b x (m - mean m over the mission's legs)
  bc3b + the day's level:   ... + c x (mean m over the mission)
  bc4  + physics:   m_hat = a + b x (m - day mean m) + c x (log b0 - day mean log b0)   b0 = the leg's pooled road-load fit
  bcF  force balance (probe): a + b x (F_b0 - day mean), F_b0 = traction-force fit on the leg (not a model input)
The mission context is every leg of the mission (labelled or not), weighted by its number of 1,024 s windows; only
labelled legs are scored. A leg alone in its mission reads a.
Rule (written after the dev-only probes probe_busdev.py / probe_buscal.py, which suggested these maps): skill = 1 - CV MAE / CV MAE of the dev-median guess; keep a
rung if its skill beats the incumbent's by more than 0.01 AND its correlation with passengers does not fall. The
incumbent at the start is pb1 as shipped (raw reading). The frozen simulated set does not apply: no rung changes the
network. The exam is never computed here.
usage: python buscal.py [ckpt=pb1]   -> work/ratchet_buscal.json, work/buscal_<ckpt>.json (final coefficients)"""
from common import *
import hashlib
import pandas as pd
import evaluate as E, busbank, busladder as BL, features as F

LOG = f'{ROOT}/ratchet_buscal.json'
NFOLD = 5


def fold(mission):
    return int(hashlib.sha256(mission.encode()).hexdigest(), 16) % NFOLD


def leg_table(net, rb):
    """per leg: network mass (kg), log b0 of the pooled road-load fit, number of windows; meta"""
    legs, m, s = BL.real_predict(net, rb)
    o = rb['offs']; meta = {x['trip_id']: x for x in rb['meta']}
    rows = []
    for t, mm in zip(legs, m):
        js = np.where(rb['trip'] == t)[0]
        ps = np.sum([F.pstats(rb['raw'][o[j]:o[j + 1]].T.astype(np.float32), rb['flags'][j]) for j in js], 0)
        sol = F.psolve(ps)
        lb0 = float(np.log(max(sol['b'][0], 50.0))) if sol is not None else np.nan
        x = meta[t]
        rows.append(dict(trip_id=t, mission=x['mission'], bus=x['bus'], season=x['season'], label_ok=x.get('label_ok', True),
                         pax=x['pax'], truth=x['mass_kg'], m=float(mm), lb0=lb0, nwin=len(js)))
    d = pd.DataFrame(rows)
    w = d.nwin.astype(float)
    for k in ['m', 'lb0']:
        ok = np.isfinite(d[k])
        num = (d[k].where(ok, 0) * w * ok).groupby(d.mission).transform('sum'); den = (w * ok).groupby(d.mission).transform('sum')
        d[k + '_day'] = num / den.replace(0, np.nan)
        d[k + '_rel'] = (d[k] - d[k + '_day']).fillna(0.0)
    return d


def design(d, rung):
    one = np.ones(len(d))
    if rung == 'bc2': return np.c_[one, d.m]
    if rung == 'bc3': return np.c_[one, d.m_rel]
    if rung == 'bc3b': return np.c_[one, d.m_rel, d.m_day]
    if rung == 'bc4': return np.c_[one, d.m_rel, d.lb0_rel]
    if rung == 'bcF': return np.c_[one, d.F_rel]
    raise KeyError(rung)


def fit(d, rung):
    if rung == 'raw': return None
    if rung == 'dumb': return np.array([float(np.median(d.truth))])
    if rung == 'bc1': return np.array([float(np.median(d.truth / d.m))])
    return np.linalg.lstsq(design(d, rung), d.truth.values, rcond=None)[0]


def apply(d, rung, c):
    if rung == 'raw': return d.m.values
    if rung == 'dumb': return np.full(len(d), c[0])
    if rung == 'bc1': return d.m.values * c[0]
    return design(d, rung) @ c


def cv(d, rung):
    pred = np.zeros(len(d)); f = d.mission.map(fold).values
    for k in range(NFOLD):
        tr, te = f != k, f == k
        pred[te] = apply(d[te], rung, fit(d[tr], rung))
    return pred


def metrics(pred, d):
    e = pred - d.truth.values
    r = float(np.corrcoef(pred, d.pax)[0, 1]) if np.std(pred) > 0 else 0.0
    sl = float(np.polyfit(busbank.PAX_KG * d.pax, pred, 1)[0])
    return dict(mae=float(np.mean(np.abs(e))), med_ae=float(np.median(np.abs(e))), bias=float(np.median(e)), r=r, slope=sl)


def force_rel(ctx):
    """force-balance probe: traction-force road-load fit per leg (all v > 2 m/s, a > -0.3 seconds), day-relative"""
    import realdata as rd
    sm = lambda x, k=3: np.convolve(x, np.ones(k) / k, mode='same')
    fb = {}
    for t in ctx.trip_id:
        _, a = rd.load_trip(t)
        v = np.nan_to_num(a['v']); g = np.nan_to_num(a['grade']); Fx = sm(np.nan_to_num(np.asarray(a['traction_force'], float)))
        rho = np.where(np.isfinite(a['rho_air']) & (a['rho_air'] > .5), a['rho_air'], 1.2)
        acc = np.zeros_like(v); acc[1:-1] = (v[2:] - v[:-2]) / 2; vs, as_, gs = sm(v), sm(acc), sm(g)
        k = (vs > 2) & (as_ > -0.3) & (Fx > 0)
        if k.sum() < 200: fb[t] = np.nan; continue
        X = np.c_[1.03 * as_ + 9.81 * gs, np.ones(len(v)), 0.5 * rho * vs ** 2][k]
        fb[t] = float(np.linalg.lstsq(X, Fx[k], rcond=None)[0][0])
    x = ctx.trip_id.map(fb); w = ctx.nwin.astype(float); ok = np.isfinite(x)
    day = (x.where(ok, 0) * w * ok).groupby(ctx.mission).transform('sum') / (w * ok).groupby(ctx.mission).transform('sum').replace(0, np.nan)
    return (x - day).fillna(0.0)


LADDER = [('bc1', 'Level: the network reading x one dev-fitted factor (v2.0 exam note)'),
          ('bc2', 'Line: a + b x the network reading'),
          ('bc3', 'Day-relative: dev level + b x (reading - the mission-day mean reading)'),
          ('bc3b', "+ c x the mission-day mean reading (the day's level)"),
          ('bc4', '+ c x (log b0 - day mean): the road-load fit of the leg, day-relative'),
          ('bcF', 'Force balance probe: day-relative traction-force fit alone (traction force is not a model input)')]


def main(ck='pb1'):
    net = E.load_net(f'{ROOT}/ckpt/{ck}.pt')
    ctx = leg_table(net, busbank.load_real('dev', all_legs=True))
    d = ctx[ctx.label_ok & np.isfinite(ctx.truth)].reset_index(drop=True)
    print(len(d), 'scored dev gate legs,', d.mission.nunique(), 'missions,', len(ctx) - len(d), 'context-only legs', flush=True)
    base = metrics(cv(d, 'dumb'), d); mae0 = base['mae']
    skill = lambda m: 1 - m['mae'] / mae0
    inc = dict(rung='raw', m=metrics(d.m.values, d)); inc['skill'] = skill(inc['m'])
    log = [dict(rung='dumb', note='Dev-median guess (cross-validated), the reference', cv=base, skill=0.0, accepted=None),
           dict(rung='raw', note=f'{ck} as shipped (raw reading)', cv=inc['m'], skill=inc['skill'], accepted=True, why='incumbent')]
    print(f"dumb CV MAE {mae0:.0f}; raw {inc['m']['mae']:.0f} skill {inc['skill']:+.3f} r {inc['m']['r']:.2f}")
    for rung, note in LADDER:
        if rung == 'bcF':
            d = d.copy(); d['F_rel'] = force_rel(ctx).loc[ctx.label_ok & np.isfinite(ctx.truth)].values
        m = metrics(cv(d, rung), d); sk = skill(m)
        ok = sk > inc['skill'] + 0.01 and m['r'] >= inc['m']['r'] - 1e-9
        why = (f"CV skill {sk:+.3f} vs {inc['skill']:+.3f}" + ('' if sk > inc['skill'] + 0.01 else ' (needs +0.01)') +
               f"; r {m['r']:.2f} vs {inc['m']['r']:.2f}" + ('' if m['r'] >= inc['m']['r'] - 1e-9 else ' (falls)'))
        if rung == 'bcF': ok = False; why += '; probe only (traction force is not an input of the model)'
        c = fit(d, rung)
        log.append(dict(rung=rung, note=note, cv=m, skill=sk, accepted=bool(ok), why=why, vs=inc['rung'], coef=c.tolist()))
        print(f"{rung:5s} CV MAE {m['mae']:5.0f} median {m['med_ae']:4.0f} r {m['r']:.2f} slope {m['slope']:.2f} skill {sk:+.3f}  "
              f"{'KEPT' if ok else 'rejected'} - {why}  coef {np.round(c, 4).tolist()}", flush=True)
        if ok: inc = dict(rung=rung, m=m, skill=sk)
    # final map: the last kept rung, fitted on every dev gate leg; 1-sigma = 68.3rd percentile of |CV residual| (kg)
    best = inc['rung']; c = fit(d, best); res = np.abs(cv(d, best) - d.truth.values)
    out = dict(ckpt=ck, rung=best, coef=c.tolist(), sigma_kg=float(np.percentile(res, 68.3)), n_dev_legs=len(d),
               dumb_cv_mae=mae0, cv=inc['m'], context='every leg of the mission, weighted by its 1,024 s windows')
    jsave(log, LOG); jsave(out, f'{ROOT}/buscal_{ck}.json')
    print('final', json.dumps(out))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'pb1')
