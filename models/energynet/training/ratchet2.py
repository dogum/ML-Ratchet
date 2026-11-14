"""Phase D ratchet (RoadSim physics v2): score every rung in rungs2.json on the frozen simulated set val2 and the dev
gate D2, then take every decision in order.

Physics (target) versions on val2 (dataset.V2COLS): a = RoadSim v2 raw; b = + RoadSim's OBD label model for MAF-
estimated fuel; c = b + v1's on-road wrapper; d = c + v1's plug-in patch; e = b + the plug-in patch. A rung whose
version differs from the incumbent's changes the simulated physics: as in v1, the frozen set then becomes the
candidate's version and the incumbent is re-scored on it (the dev gate is the real judge of such a rung).

Rule (ratchet.yaml), as changed in the open on 14 Nov 2026 after r4 (NOTEBOOK.md):
  model rungs (same physics version as the incumbent): keep if val2 skill beats the incumbent by more than MARGIN and
      dev D2 is not worse by more than GATE (v1's rule);
  physics rungs (the simulated targets or labels change): the frozen set of the new physics favours any network trained
      on it (r4: +0.097 skill while D2 got worse), so the dev gate judges: keep if D2 improves by more than MARGIN and
      the candidate's skill on its own version is not below the incumbent's on its own version by more than MARGIN;
  gate-only rungs (inputs only real data carries, e.g. the bus's curb mass): keep if D2 improves by more than MARGIN
      and the val2 skill does not drop by more than MARGIN.
The first rule's decision is logged next to each physics rung ('rule1').
python ratchet2.py [--rescore name,...]   -> work/ratchet2.json"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, os, argparse
sys.path.insert(0, _P.SRC)
import numpy as np
import evaluate as EV
from model_np import ENet, Ensemble

MARGIN, GATE = 0.01, 0.01
CK = _P.WORK + '/ckpt'
OUT = _P.WORK + '/ratchet2.json'
RUNGS = json.load(open(os.path.join(_P.SRC, 'rungs2.json')))
VERS = ['a', 'b', 'c', 'd', 'e', 'f', 'e+L']


def load(r):
    ps = [f"{CK}/{x}.npz" for x in (r.get('members') or [r.get('net', r['name'])])]
    if not all(os.path.exists(p) for p in ps): return None
    return Ensemble(ps) if len(ps) > 1 else ENet(ps[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--rescore', default='')
    a = ap.parse_args()
    redo = set(x for x in a.rescore.split(',') if x)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cache = {r['name']: r for r in prev.get('rungs', []) if r['name'] not in redo}
    log = {'margin': MARGIN, 'gate': GATE, 'frozen_set': 'val2', 'dev_score': 'D2 (evaluate.dev_gate2, 18 terms)', 'rungs': []}
    inc = None
    for r in RUNGS:
        m = load(r)
        if m is None: continue
        e = dict(r); old = cache.get(r['name'], {})
        e['val'] = dict(old.get('val', {}))
        for v in VERS:
            if v.endswith('+L') and not os.path.exists(f'{_P.WORK}/data/bank_val2/.lab_epa'): continue
            if v not in e['val']: e['val'][v] = EV.val_score2(m, v)
        for v in (r['version'],) + ((inc['version'],) if inc else ()):
            if v not in e['val']: e['val'][v] = EV.val_score2(m, v)
        e['dev'] = old.get('dev') or EV.dev_gate2(m, r.get('bus_spec', False), r.get('epa_hyb', False))
        if inc is None:
            e['decision'] = 'kept'; e['why'] = 'first rung'; e['frozen'] = r['version']
        else:
            fv = r['version']
            sk, sk0 = e['val'][fv]['skill'], inc['val'][fv]['skill']
            dsk = sk - sk0; dD = e['dev']['D'] - inc['dev']['D']
            e['rule1'] = 'kept' if (dsk > MARGIN and dD <= GATE) else 'rejected'
            if r.get('gate_only'):
                ok = dD < -MARGIN and dsk > -MARGIN
                rule = f"gate-only rung: dev D2 must improve by > {MARGIN}, val skill may not drop by > {MARGIN}"
            elif fv != inc['version']:
                own = sk - inc['val'][inc['version']]['skill']
                ok = dD < -MARGIN and own > -MARGIN
                rule = (f"physics rung: dev D2 must improve by > {MARGIN}; own-version skill {sk:.3f} vs incumbent's "
                        f"{inc['val'][inc['version']]['skill']:.3f} on its own version (may not drop by > {MARGIN}); first rule: {e['rule1']}")
            else:
                ok = dsk > MARGIN and dD <= GATE
                rule = f"val skill must rise by > {MARGIN}, dev D2 may not rise by > {GATE}"
            e['decision'] = 'kept' if ok else 'rejected'; e['frozen'] = fv
            e['why'] = (f"val2[{fv}] skill {sk:.3f} vs {sk0:.3f} (d {dsk:+.3f}); dev D2 {e['dev']['D']:.4f} vs {inc['dev']['D']:.4f} "
                        f"(d {dD:+.4f}); {rule}")
        if e['decision'] == 'kept': inc = e
        log['rungs'].append(e)
        print(f"{r['name']:14s} v{r['version']} skill {e['val'][e['frozen']]['skill']:.3f} MdAPE {e['val'][e['frozen']]['MdAPE']:.3f} "
              f"D2 {e['dev']['D']:.4f} -> {e['decision']}  {e.get('why', '')}", flush=True)
    log['incumbent'] = inc['name'] if inc else None
    json.dump(log, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
