"""Prior rungs for SwellCast (15 Nov 2026). The network is unchanged; its readings are combined with a prior.

Why: on the exam the dev-median guess beats the network on distance and time. The training simulations spread storm
distances log-uniformly over 700-11,000 km (label sd 0.33 dex), while real confirmed arrivals sit in a much narrower
band. A network trained on the wide prior does not know the narrow one, so its reading should be shrunk toward the
real prior by how sure it is. First this script checks that dev shows the same failure (diagnostic, not a rung).

Rule for these rungs (fixed 15 Nov 2026, before either was scored; copied into ratchet.yaml): the network weights do
not change, so the frozen simulated set cannot judge them (a prior fitted to real storms has no meaning on simulated
ones). Keep a rung if its dev skill rises by more than 0.01 over the incumbent AND no dev median error (distance, time,
direction) gets more than 5% worse. Dev skill = mean over distance and time of 1 - (median error / median error of the
dev-median guess), every number scored leaving each dev storm out.

Rungs, both defined before they were scored:
  r6  distance prior: log10 D_post = precision-weighted mean of the network (its dev-calibrated sigma) and a normal prior
      (mean and sd of log10 D over the dev arrivals).
  r7  r6 + a dispersion prior on time: the window is centred on the predicted arrival of 0.06 Hz swell (+-36 h, uniform),
      so t0 = -D / c_g(0.06 Hz) - jitter: prior mean -D_post / 46.8 km/h, variance 72^2/12 h^2 plus the distance
      uncertainty carried through (d t0 / d log10 D = -ln 10 D / 46.8). Combined with the network's t0 by precision.
Direction is unchanged in both.
  r8  (defined 15 Nov 2026 after r6 and r7 were scored, before r8 was: r7 still loses to the dev-median guess on distance)
      r7 with the network's distance and time sigmas each multiplied by a factor chosen from {1, 1.5, 2, 3, 4, 6, 8, 12}
      on the other dev storms (nested inside the leave-one-storm-out loop): the factor whose blend gives the smallest
      median error there. A larger factor trusts the prior more. The release would carry the factors chosen on all dev.

Dev scoring leaves each dev storm out: its prior and its error-bar factors are fitted on the other 17 storms, and the
dev-median guess it is compared with is the median of the other 17 too. Writes work/prior_dev.json."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as _P
sys.path.insert(0, _P.MODEL)
import events as EV, swellcast as SC
CG06_KMH = 9.81 / (4 * np.pi * 0.06) * 3.6     # group speed of 0.06 Hz swell, km/h (46.8)
JIT_VAR = 72.0 ** 2 / 12                          # window jitter: uniform +-36 h

def angdiff(a, b): return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)

def sigma_factors(P, D, t0, dr):
    """widen-only factors (68% of |error| / sigma inside 1), as calib_sigma.fit"""
    eD = np.abs(P['logD'] - np.log10(D)) / P['logD_sig']; et = np.abs(P['t0'] - t0) / P['t0_sig']
    ea = np.radians(angdiff(P['dir'], dr)) * np.sqrt(P['kappa'])
    return dict(logD=max(1.0, float(np.quantile(eD, 0.68))), t0=max(1.0, float(np.quantile(et, 0.68))), dir=max(1.0, float(np.quantile(ea, 0.68))))

def fit_prior(D_km):
    l = np.log10(np.asarray(D_km, float)); return dict(logD_mean=float(l.mean()), logD_sd=float(l.std(ddof=1)))

def blend(P, prior, use_t0=True):
    """P: network readings with calibrated sigmas (logD, logD_sig, t0, t0_sig, ...). Returns a new dict."""
    Q = dict(P); wn, wp = 1 / P['logD_sig'] ** 2, 1 / prior['logD_sd'] ** 2
    Q['logD'] = (wn * P['logD'] + wp * prior['logD_mean']) / (wn + wp); Q['logD_sig'] = 1 / np.sqrt(wn + wp); Q['D_km'] = 10 ** Q['logD']
    if use_t0:
        tw = -Q['D_km'] / CG06_KMH; sw2 = JIT_VAR + (np.log(10) * Q['D_km'] / CG06_KMH * Q['logD_sig']) ** 2
        a, b = 1 / P['t0_sig'] ** 2, 1 / sw2; Q['t0'] = (a * P['t0'] + b * tw) / (a + b); Q['t0_sig'] = 1 / np.sqrt(a + b)
    return Q

def scale(P, f):
    Q = dict(P); Q['logD_sig'] = P['logD_sig'] * f['logD']; Q['t0_sig'] = P['t0_sig'] * f['t0']; Q['kappa'] = P['kappa'] / f['dir'] ** 2; return Q

def errors(P, D, t0, dr, hasd):
    eD = np.abs(10 ** (P['logD'] - np.log10(D)) - 1) * 100; et = np.abs(P['t0'] - t0); ea = angdiff(P['dir'], dr)
    return eD, et, np.where(hasd, ea, np.nan)

def summary(eD, et, ea, eDd, etd):
    m = dict(D_med_pct=float(np.median(eD)), t0_med_h=float(np.median(et)), dir_med_deg=float(np.nanmedian(ea)))
    m['skill'] = float(np.mean([1 - m['D_med_pct'] / np.median(eDd), 1 - m['t0_med_h'] / np.median(etd)])); return m

GRID = (1, 1.5, 2, 3, 4, 6, 8, 12)

def inflate(P, a):
    Q = dict(P); Q['logD_sig'] = P['logD_sig'] * a['logD']; Q['t0_sig'] = P['t0_sig'] * a['t0']; return Q

def inflation(R, D, t0, dr, sid):
    """r8: on these storms (leave-one-storm-out inside), the sigma multipliers whose r7 blend has the smallest median
    distance error and median time error"""
    best = {}
    for key in ('logD', 't0'):
        sc = []
        for g in GRID:
            e = np.full(len(D), np.nan)
            for s in sorted(set(sid)):
                tr, te = sid != s, sid == s; f = sigma_factors({k: v[tr] for k, v in R.items()}, D[tr], t0[tr], dr[tr]); pr = fit_prior(D[tr])
                a = dict(logD=1.0, t0=1.0); a[key] = g; Q = blend(inflate(scale({k: v[te] for k, v in R.items()}, f), a), pr, use_t0=True)
                e[te] = np.abs(10 ** (Q['logD'] - np.log10(D[te])) - 1) if key == 'logD' else np.abs(Q['t0'] - t0[te])
            sc.append(np.median(e))
        best[key] = float(GRID[int(np.argmin(sc))])
    return best

def loso(ev, R, X=None):
    """leave-one-storm-out readings for every variant. R: raw network readings (uncalibrated sigmas)."""
    D = np.array([e['D'] / 1e3 for e in ev]); t0 = np.array([e['t0_rel'] for e in ev]); dr = np.array([e['dir'] for e in ev])
    hasd = np.array([e['dir_coverage'] > 0.5 for e in ev]); sid = np.array([e['sid'] for e in ev])
    V = {k: {a: np.full(len(ev), np.nan) for a in ('logD', 'logD_sig', 't0', 't0_sig', 'dir', 'kappa')} for k in ('r4', 'r6', 'r7', 'r8', 'dumb')}
    for s in sorted(set(sid)):
        tr, te = sid != s, sid == s
        Rtr = {k: v[tr] for k, v in R.items()}; f = sigma_factors(Rtr, D[tr], t0[tr], dr[tr]); pr = fit_prior(D[tr])
        Pte = scale({k: v[te] for k, v in R.items()}, f)
        a8 = inflation({k: v[tr] for k, v in R.items()}, D[tr], t0[tr], dr[tr], sid[tr])
        out = dict(r4=Pte, r6=blend(Pte, pr, use_t0=False), r7=blend(Pte, pr, use_t0=True), r8=blend(inflate(Pte, a8), pr, use_t0=True),
                   dumb=dict(logD=np.full(te.sum(), np.median(np.log10(D[tr]))), t0=np.full(te.sum(), np.median(t0[tr])),
                             dir=np.full(te.sum(), np.degrees(np.arctan2(np.sin(np.radians(dr[tr])).mean(), np.cos(np.radians(dr[tr])).mean())) % 360)))
        for k, P in out.items():
            for a in V[k]:
                if a in P: V[k][a][te] = P[a]
    return V, D, t0, dr, hasd, sid

def run(R, ev, boot=2000, seed=0):
    V, D, t0, dr, hasd, sid = loso(ev, R)
    E = {k: errors(V[k], D, t0, dr, hasd) for k in V}; eDd, etd, _ = E['dumb']
    res = {k: summary(*E[k], eDd, etd) for k in V}
    for k in ('r4', 'r6', 'r7', 'r8'):
        P = V[k]; res[k]['logD_in_1sig'] = float(np.mean(np.abs(P['logD'] - np.log10(D)) < P['logD_sig'])); res[k]['t0_in_1sig'] = float(np.mean(np.abs(P['t0'] - t0) < P['t0_sig']))
    # bootstrap over storms: how sure is each variant's skill gain over r4?
    rng = np.random.default_rng(seed); S = sorted(set(sid)); idx = {s: np.where(sid == s)[0] for s in S}; gain = {k: [] for k in ('r6', 'r7', 'r8', 'dumb')}
    for _ in range(boot):
        ii = np.concatenate([idx[s] for s in rng.choice(S, len(S))])
        base = summary(*(e[ii] for e in E['r4']), eDd[ii], etd[ii])['skill']
        for k in gain: gain[k].append(summary(*(e[ii] for e in E[k]), eDd[ii], etd[ii])['skill'] - base)
    for k, g in gain.items(): res[k]['skill_gain_vs_r4_boot'] = dict(median=float(np.median(g)), p05=float(np.percentile(g, 5)), p95=float(np.percentile(g, 95)), frac_positive=float(np.mean(np.array(g) > 0)))
    return res, V

if __name__ == '__main__':
    ev = EV.load('dev', 'confirmed'); X = EV.windows(ev); net = SC.SwellCast(); R = net.predict(X, calibrated=False)
    R = {k: v for k, v in R.items() if k in ('logD', 'logD_sig', 't0', 't0_sig', 'dir', 'kappa')}
    # the release as shipped (sigma factors fitted on all of dev) and the in-sample dev-median guess, for reference
    D = np.array([e['D'] / 1e3 for e in ev]); t0 = np.array([e['t0_rel'] for e in ev]); dr = np.array([e['dir'] for e in ev]); hasd = np.array([e['dir_coverage'] > 0.5 for e in ev])
    eD, et, ea = errors(net.predict(X), D, t0, dr, hasd)
    ins = dict(D_med_pct=float(np.median(np.abs(10 ** (np.median(np.log10(D)) - np.log10(D)) - 1) * 100)), t0_med_h=float(np.median(np.abs(np.median(t0) - t0))))
    if '--diagnostic' in sys.argv:      # step 1: does dev show the exam's failure? (r4 and the dev-median guess only)
        V, D_, t0_, dr_, hd_, _ = loso(ev, R); E = {k: errors(V[k], D_, t0_, dr_, hd_) for k in ('r4', 'dumb')}
        out = {k: summary(*E[k], E['dumb'][0], E['dumb'][1]) for k in E}; out['release_as_shipped'] = dict(D_med_pct=float(np.median(eD)), t0_med_h=float(np.median(et)), dir_med_deg=float(np.nanmedian(ea)))
        out['dumb_in_sample'] = ins; out['n'] = len(ev); out['storms'] = len({e['sid'] for e in ev}); print(json.dumps(out, indent=1))
        json.dump(out, open(os.path.join(_P.WORK, 'prior_dev_diagnostic.json'), 'w'), indent=1); sys.exit()
    res, V = run(R, ev)
    out = dict(r8_factors_all_dev=inflation(R, D, t0, dr, np.array([e['sid'] for e in ev])), n=len(ev), storms=len({e['sid'] for e in ev}), release_as_shipped=dict(D_med_pct=float(np.median(eD)), t0_med_h=float(np.median(et)), dir_med_deg=float(np.nanmedian(ea))),
               dumb_in_sample=ins, loso=res, prior_all_dev=fit_prior(D), cg06_kmh=CG06_KMH)
    print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(_P.WORK, 'prior_dev.json'), 'w'), indent=1)
