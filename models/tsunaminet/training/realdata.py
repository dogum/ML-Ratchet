"""Real DART events -> network inputs, held-out forecast targets, and the event's simulated twin (USGS finite-fault
slip moved onto the nearest unit sources). Used by dev evaluation, calibration and (separately) the exam.
Target rule (frozen with splits.json): every deployed site with Green's functions that is not an input, predicted
arrival ta <= 1250 min, >= 50 % minute coverage in [ta-10, ta+180], >= 8 pre-arrival samples, and observed
max |eta| in that window >= 5 x the robust pre-arrival noise sigma. Amplitude = max |eta| over the observed minutes."""
import os, json, datetime as dt, numpy as np
from _paths import DATA
import tsim, dart, features as F
SITES = json.load(open(os.path.join(DATA, 'dart_sites.json')))
RAW = {e['id']: e for e in json.load(open(os.path.join(DATA, 'events_raw.json')))}
TW0, TW1 = -10, 180

def deployed(t0):
    day = dt.datetime.fromtimestamp(t0, dt.timezone.utc).strftime('%Y-%m-%d')
    return [s for s in SITES if any(a <= day and (b == '' or day <= b) for a, b in s['spans'])]

def prepare(ev, gf, geo, delay_frac=0.0):
    fr = geo.frame(ev['lon'], ev['lat']); cen = [u for u in (fr['units'][tsim.HALF], fr['units'][tsim.NCOL + tsim.HALF]) if u >= 0]
    cands, allst = [], []
    for s in deployed(ev['t0']):
        if s['site'] not in gf.G: continue
        r = dart.detided_minutes(s['station'], ev['t0'])
        if r is None: continue
        eta, mask, info = r
        ta = gf.arrival(s['site'], cen)
        if ta is None: continue
        allst.append(dict(site=s['site'], ta=int(ta), eta=eta, mask=mask))
        if ta <= F.TA_MAX:
            x, sc, cov, scale = F.station_block(gf, s['site'], fr['units'], int(ta), eta, mask)
            if x is not None: cands.append((int(ta), cov, scale, (x, sc, s['site'])))
    chosen = F.select(cands)
    X = np.zeros((1, F.K, F.NCH, tsim.NW), np.float32); S = np.zeros((1, F.K, F.NSC), np.float32); M = np.zeros((1, F.K), bool)
    for k, c in enumerate(chosen): X[0, k] = c[3][0]; S[0, k] = c[3][1]; M[0, k] = True
    inputs = [c[3][2] for c in chosen]
    targets = []
    for st in allst:
        if st['site'] in inputs or st['ta'] > 1250: continue
        a, b = max(st['ta'] + TW0, 0), st['ta'] + TW1
        m = st['mask'][a:b + 1]
        if m.mean() < 0.5: continue
        pre = st['mask'][:max(st['ta'] + TW0, 0)]
        if pre.sum() < 8: continue
        pv = st['eta'][:max(st['ta'] + TW0, 0)][pre]; sig = 1.4826 * np.median(np.abs(pv - np.median(pv))) + 1e-4
        amp = float(np.abs(st['eta'][a:b + 1][m]).max())
        if amp < 5 * sig: continue
        targets.append(dict(site=st['site'], ta=st['ta'], a=a, b=b, mask=m, amp=amp, sigma=float(sig), eta=st['eta']))
    return dict(frame=fr, X=X, S=S, M=M, inputs=inputs, input_ta=[c[0] for c in chosen], targets=targets, all=allst)

def ffm_units(ev, geo):
    """USGS finite-fault moment moved to the nearest unit source of the event's zone -> (units, slips in m)."""
    f = RAW[ev['id']]['usgs']['ffm']; sub = np.array(f['sub'])     # lat lon depth slip(cm) moment(dyne-cm)
    zone = geo.chains[geo.frame(ev['lon'], ev['lat'])['chain']][0]['zone']
    cand = np.array([i for i, u in enumerate(geo.U) if u['name'].startswith(zone)])
    c = geo.centers[cand]
    mom = {}
    for la, lo, de, sl, mo in sub:
        k = cand[int(np.argmin(tsim.hav(la, lo % 360, c[:, 1], c[:, 0])))]; mom[k] = mom.get(k, 0.0) + mo * 1e-7  # dyne-cm -> N m
    units = np.array(sorted(mom)); M0 = np.array([mom[k] for k in units])
    return units, M0 / (tsim.MU * tsim.UNIT_AREA)

def predict_amp(gf, site, units, slips, a, b, mask, delay=0.0):
    w = gf.wave(site, units, slips)
    if delay:
        w = np.interp(np.arange(len(w)) - delay, np.arange(len(w)), w, left=0.0)
    return float(np.abs(w[a:b + 1][mask]).max()), w
