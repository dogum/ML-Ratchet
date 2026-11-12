"""StarQuake ratchet. Objective: on the frozen simulated set (5,000 simulated red giants in real noise from 42 quiet
dwarfs never used in training, mixed observing lengths), the mean of two hit rates: numax within 5% and dnu within 2%.
A rung is kept only if the objective rises by more than 0.01 AND on the 18 real dev stars (whole light curve and a
90-day piece) the mean of its four median errors, each relative to the incumbent's, is no more than 10% worse.
The exam is computed and logged but never used to decide."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np
sys.path.insert(0, _P.SRC); import snet as SN
LOG = _P.WORK + '/ratchet.json'; CK = _P.WORK + '/ckpt/'

def dev_metrics(net):
    a = SN.real(net, 'dev', 0)[0]; b = SN.real(net, 'dev', 90)[0]
    return dict(full=a, d90=b, keys=[a['numax_med'], a['dnu_med'], b['numax_med'], b['dnu_med']])

def exam_metrics(net):
    return {str(L): SN.real(net, 'exam', L)[0] for L in (0, 365, 90, 54, 27)}

def decide(rec, inc):
    if inc is None: rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01; rel = float(np.mean([a / max(b, 1e-9) for a, b in zip(rec['dev']['keys'], inc['dev']['keys'])])); d_ok = rel <= 1.10
    rec['dev_rel'] = rel; rec['accepted'] = bool(s_ok and d_ok); rec['vs'] = inc['ckpt']
    rec['why'] = f"score {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') + f"; dev errors {100 * (rel - 1):+.0f}% vs the incumbent" + ('' if d_ok else ' (gate: at most +10%)')

def score(ckpt):
    net = SN.load_net(CK + ckpt); v = SN.val(net); return dict(val=v, objective=v['objective'], dev=dev_metrics(net), exam=exam_metrics(net))

def show(rec):
    v = rec['val']; d = rec['dev']; e = rec['exam']
    print(f"{rec['ckpt']}: objective {v['objective']:.3f} (numax hit {v['numax_hit']:.3f}, dnu hit {v['dnu_hit']:.3f}; full-length {v['full']['numax_hit']:.3f}/{v['full']['dnu_hit']:.3f})")
    print('  dev  full: numax %.4f dnu %.4f | 90 d: numax %.4f dnu %.4f' % tuple(d['keys']))
    for L, s in e.items(): print(f"  exam {L:>4}: numax med {s['numax_med']:.4f} hit {s['numax_hit']:.3f} cov {s['numax_cov']:.2f} | dnu med {s['dnu_med']:.4f} hit {s['dnu_hit']:.3f} cov {s['dnu_cov']:.2f}")
    print('  ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])

def main(ckpt, note):
    log = json.load(open(LOG)) if os.path.exists(LOG) else []; inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = dict(ckpt=ckpt, note=note, val_set=os.path.basename(SN.VAL), **score(ckpt)); decide(rec, inc); log.append(rec); json.dump(log, open(LOG, 'w'), indent=1); show(rec)

def rescore():
    """Physics changed: re-score every rung on the current frozen set and re-take every decision in order."""
    log = json.load(open(LOG)); inc = None
    for rec in log:
        rec.update(score(rec['ckpt'])); rec['val_set'] = os.path.basename(SN.VAL); decide(rec, inc); inc = rec if rec['accepted'] else inc; show(rec)
    json.dump(log, open(LOG, 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == '--rescore': rescore()
    else: main(sys.argv[1], sys.argv[2])
