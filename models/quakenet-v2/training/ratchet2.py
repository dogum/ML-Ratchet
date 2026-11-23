"""QuakeNet v2's ratchet. A rung is kept only if
  (1) its objective on the frozen simulated set beats the incumbent's by more than 0.01, where the objective is the mean
      over epicentre (error capped at 2,000 km), log depth, Mw and origin time of 1 - mean error / mean error of rung 0
      (QuakeNet v1.2 per station, combined classically) on the same set; and
  (2) the real dev events (40 events: their first 2 stations, and all their stations) get no more than 5% worse: the
      mean over the four outputs of (dev mean error / incumbent's dev mean error) must be <= 1.05.
Every rung is logged in ratchet.json, kept or not. Usage: ratchet2.py score <rung> <ckpt|anchor> <frozen set> "<change>"
Frozen sets: val1 (physics v1: P residual spread 1.5 s), later versions listed in FROZEN."""
import sys, os, json, numpy as np
from _paths import H; sys.path.insert(0, H + 'src')
import score2 as SC, train2 as T2
FROZEN = {'val1': dict(file='bank/val1_samples.npz', res=1.5), 'val2': dict(file='bank/val2_samples.npz', res=1.5), 'val3': dict(file='bank/val3_samples.npz', res=1.5)}
LOG = H + 'ratchet.json'; CAP = 2000.0

def errs_of(P, D, rows):
    e = SC.errors(P, D, rows); e['epi_cap'] = np.minimum(e['epi_km'], CAP); return e

def preds(which, D, rows, res=None, fixed=None, real=False):
    if which == 'anchor': return SC.baselines(D, rows, res, fixed, real, tt_loc=False)[0]
    net, a = SC.load_net(which); return SC.predict(net, a, D, rows, res, fixed, real, key=('real' if real else 'sim', len(D['K'])))

def objective(e, ref):
    sk = {k: float(1 - np.nanmean(e[k]) / np.nanmean(ref[k])) for k in ('epi_cap', 'logz', 'mw', 't0')}; return float(np.mean(list(sk.values()))), sk

def dev_means(e): return {k: float(np.nanmean(e[k])) for k in ('epi_cap', 'logz', 'mw', 't0')}

_cache = {}
def frozen(name):
    if name not in _cache:
        f = FROZEN[name]; D = T2.load(H + f['file']); rows = np.arange(len(D['K'])); ref = errs_of(preds('anchor', D, rows, f['res'], D['res_fixed']), D, rows); _cache[name] = (D, rows, ref)
    return _cache[name]

def score(which, fz):
    D, rows, ref = frozen(fz); f = FROZEN[fz]; e = errs_of(preds(which, D, rows, f['res'], D['res_fixed']), D, rows); o, sk = objective(e, ref)
    Dd = T2.load(H + 'data/dev_samples.npz'); rd = np.arange(len(Dd['K'])); ed = errs_of(preds(which, Dd, rd, real=True), Dd, rd)
    return dict(objective=o, skill=sk, sim=SC.summary(e), dev=dev_means(ed), dev_summary=SC.summary(ed))

def log():
    return json.load(open(LOG)) if os.path.exists(LOG) else []

def rescore(fz, rungs):
    """Physics changed: score every rung (in order) on the new frozen set fz and re-take every decision in order.
    rungs: list of (rung id, ckpt, change). Appends rows with frozen = fz."""
    L = log(); inc = None
    for rung, which, change in rungs:
        s = score(which, fz)
        if inc is None: dec, why = 'baseline', f'Rung 0, re-scored on {fz}.'
        else:
            gain = s['objective'] - inc['objective']; ratio = float(np.mean([s['dev'][k] / inc['dev'][k] for k in s['dev']]))
            ok = gain > 0.01 and ratio <= 1.05; dec = 'kept' if ok else 'rejected'
            why = f"re-taken on {fz}: {gain:+.3f} vs {inc['rung']} ({'beats' if gain > 0.01 else 'short of'} the 0.01 margin); dev error ratio {ratio:.3f} ({'passes' if ratio <= 1.05 else 'fails'} the 1.05 gate)"
        row = dict(rung=rung, ckpt=which, change=change, frozen=fz, decision=dec, why=why, **s); L.append(row)
        if dec in ('kept', 'baseline'): inc = row
        print(json.dumps(dict(rung=rung, frozen=fz, decision=dec, why=why, objective=round(s['objective'], 4))), flush=True)
    json.dump(L, open(LOG, 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == 'rescore':
        fz = sys.argv[2]; L = log(); seen = []
        for r in L:
            if r['rung'] not in [x[0] for x in seen]: seen.append((r['rung'], r['ckpt'], r['change']))
        rescore(fz, seen); sys.exit()
    cmd, rung, which, fz, change = sys.argv[1:6]; L = log(); s = score(which, fz)
    inc = next((r for r in reversed(L) if r['decision'] in ('kept', 'baseline') and r['frozen'] == fz), None)
    if inc is None: dec, why = 'baseline', 'Rung 0: the classical combination every later rung must beat.'
    else:
        gain = s['objective'] - inc['objective']; ratio = float(np.mean([s['dev'][k] / inc['dev'][k] for k in s['dev']]))
        ok = gain > 0.01 and ratio <= 1.05; dec = 'kept' if ok else 'rejected'
        why = f"{gain:+.3f} on {fz} vs {inc['rung']} ({'beats' if gain > 0.01 else 'short of'} the 0.01 margin); dev error ratio {ratio:.3f} ({'passes' if ratio <= 1.05 else 'fails'} the 1.05 gate)"
    row = dict(rung=rung, ckpt=which, change=change, frozen=fz, decision=dec, why=why, **s); L.append(row); json.dump(L, open(LOG, 'w'), indent=1)
    print(json.dumps(dict(rung=rung, decision=dec, why=why, objective=round(s['objective'], 4), skill={k: round(v, 3) for k, v in s['skill'].items()}, dev={k: round(v, 3) for k, v in s['dev'].items()})))
