"""The ratchet. usage: python ratchet.py <rung> "<change>" <frozen_set> [physics]
A rung is kept only if (1) its frozen-set score (mean skill of posterior-median Mw and along-strike centroid against
always guessing the set's median) beats the incumbent's on the same frozen set by more than 0.01, AND (2) its dev
error (Mw MAE/0.3 + centroid MAE/100 km + amplitude MAE/0.2 dex, on the 5 real dev events) is no more than 5 % worse.
Every rung is logged in ratchet.json, kept or not. `rescore` re-takes every decision in order on a new frozen set."""
import os, sys, json, numpy as np
from _paths import ROOT, DATA
import tsim, evallib, dev_eval
LOG = os.path.join(ROOT, 'ratchet.json')
MARGIN, GATE = 0.01, 1.05

def load():
    return json.load(open(LOG)) if os.path.exists(LOG) else dict(rule=__doc__.strip(), margin=MARGIN, dev_gate=GATE, rungs=[], decisions=[])

def phys(name):
    import simbank; P = simbank.load_phys()[name]
    return P.get('delay_frac', 0.0), P.get('amp_gain', 1.0)

def score(rung, frozen, gf, geo, physics='p0', dev=None):
    net = evallib.load_net(rung)
    fz = evallib.frozen_score(net, frozen)
    if dev is not None: return fz, dev[0], dev[1]
    d, g = phys(physics)
    ds, rows = dev_eval.evaluate(net, gf, geo, delay_frac=d, gain=g)
    return fz, ds, rows

def decide(L, frozen):
    """Re-take every decision in order on `frozen` using stored scores."""
    inc = None; dec = []
    for r in L['rungs']:
        s = r['scores'].get(frozen)
        if s is None: continue
        if inc is None:
            keep, why = True, 'First rung.'
        else:
            si = L_by(L, inc)['scores'][frozen]
            d_sim = s['frozen']['score'] - si['frozen']['score']; d_dev = s['dev']['dev_error'] / si['dev']['dev_error']
            keep = d_sim > MARGIN and d_dev <= GATE
            why = f"frozen {s['frozen']['score']:.3f} vs {si['frozen']['score']:.3f} ({d_sim:+.3f}); dev error x{d_dev:.3f}"
        dec.append(dict(rung=r['id'], frozen=frozen, kept=bool(keep), why=why))
        if keep: inc = r['id']
    return dec, inc

def L_by(L, rid): return next(r for r in L['rungs'] if r['id'] == rid)

if __name__ == '__main__':
    rung, change, frozen = sys.argv[1], sys.argv[2], sys.argv[3]
    physics = sys.argv[4] if len(sys.argv) > 4 else 'p0'
    geo = tsim.Geometry(); gf = tsim.GFBank(); L = load()
    r = next((x for x in L['rungs'] if x['id'] == rung), None)
    old = next(iter(r['scores'].values()), None) if r else None     # dev result does not depend on the frozen set
    fz, ds, rows = score(rung, frozen, gf, geo, physics, (old['dev'], old['dev_rows']) if old else None)
    if r is None:
        r = dict(id=rung, change=change, physics=physics, scores={}); L['rungs'].append(r)
    r['scores'][frozen] = dict(frozen=fz, dev=ds, dev_rows=rows, physics=physics)
    dec, inc = decide(L, frozen)
    L['decisions'] = [d for d in L['decisions'] if d['frozen'] != frozen] + dec
    L['incumbent_' + frozen] = inc
    json.dump(L, open(LOG, 'w'), indent=1, default=float)
    print(json.dumps(fz, indent=1)); print(json.dumps(ds, indent=1)); print(dec[-1]); print('incumbent', inc)
