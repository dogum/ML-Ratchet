"""The ratchet: score every rung on the frozen simulated validation set of the CURRENT physics version and on the
real dev gate, then re-take every decision in order. A rung is kept if it beats the incumbent's val skill by more than
MARGIN and its dev score D (mean of 15 dev MdAPEs) is not worse than the incumbent's by more than GATE.
python ratchet.py   (rungs listed in RUNGS; writes ratchet.json)"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, os
sys.path.insert(0, _P.SRC)
import numpy as np
import evaluate as EV
from model_np import ENet, Ensemble

MARGIN, GATE = 0.01, 0.01
CK = _P.WORK + '/ckpt'
OUT = _P.WORK + '/ratchet.json'
VERSIONS = {'v1': 'roadsim as frozen (raw simulated energy)',
            'v2': 'v1 + dev-calibrated MAF label model (OBD-estimated fuel reads idle-level fuel during fuel cut-off)',
            'v3': 'v2 + dev-calibrated on-road loads (A/C, accessories, real-world hybrid operation)',
            'v4': 'v3 + dev-calibrated plug-in hardware (PHEV pack power and MG2 torque, so charge-depleting trips stay electric)'}
RUNGS = json.load(open(_P.WORK + '/rungs.json'))


def main():
    have = lambda r: all(os.path.exists(f"{CK}/{x}.npz") for x in (r.get('members') or [r['name']]))
    cur = max(r['version'] for r in RUNGS if have(r))
    log = {'margin': MARGIN, 'gate': GATE, 'current_version': cur, 'versions': VERSIONS, 'rungs': []}
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cache = {r['name']: r for r in prev.get('rungs', [])}
    inc = None
    for r in RUNGS:
        if r.get('members'):
            ps = [f"{CK}/{x}.npz" for x in r['members']]
            if not all(os.path.exists(x) for x in ps): continue
            m = Ensemble(ps)
        else:
            p = f"{CK}/{r['name']}.npz"
            if not os.path.exists(p): continue
            m = ENet(p)
        e = dict(r)
        old = cache.get(r['name'], {})
        e['val'] = dict(old.get('val', {}))
        for v in sorted(VERSIONS):
            if v <= cur and v not in e['val']:
                e['val'][v] = EV.val_score(m, v)
        e['dev'] = old.get('dev') or EV.dev_gate(m)
        sk = e['val'][cur]['skill']; Dv = e['dev']['D']
        if inc is None:
            e['decision'] = 'kept'; e['why'] = 'first rung'
        else:
            dsk = sk - inc['val'][cur]['skill']; dD = Dv - inc['dev']['D']
            ok = dsk > MARGIN and dD <= GATE
            e['decision'] = 'kept' if ok else 'rejected'
            e['why'] = (f"val skill {sk:.3f} vs {inc['val'][cur]['skill']:.3f} (d {dsk:+.3f}, need > {MARGIN}); "
                        f"dev D {Dv:.3f} vs {inc['dev']['D']:.3f} (d {dD:+.3f}, gate {GATE})")
        if e['decision'] == 'kept': inc = e
        log['rungs'].append(e)
        print(f"{r['name']:12s} {r['version']} skill[{cur}] {sk:.3f}  MdAPE {e['val'][cur]['MdAPE']:.3f}  dev D {Dv:.3f}  -> {e['decision']}  {e.get('why', '')}", flush=True)
    log['incumbent'] = inc['name'] if inc else None
    json.dump(log, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
