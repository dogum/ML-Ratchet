"""Ratchet bookkeeping: score a trained rung against the incumbent and log the decision in ratchet.json.

python3 ratchet.py <rung> "<description>" [--frozen val_v1]
Rule: keep if sim primary (mean clipped R2 over crown, wind-hidden, lfmc on the frozen sim set) beats the incumbent
on THE SAME frozen set by > 0.01, AND the dev composite (mean Spearman of HS-vs-crown and wind on the 40 dev fires)
is not lower than the incumbent's by more than 0.05 (stated threshold: the dev set is small).
When the physics (frozen set version) changes, the incumbent is re-scored on the new frozen set first.
"""
from common import *
import argparse, torch
import model as M
from train import load_bank, sim_metrics, dev_metrics
from real import load_records

RJ = f'{ROOT}/ratchet.json'
MOD = f'{ROOT}/models'


def load_net(rung):
    if os.path.exists(f'{MOD}/{rung}.ens.json'):
        members = json.load(open(f'{MOD}/{rung}.ens.json'))['members']
        nets = [load_net(m) for m in members]
        return M.Ens([n for n, _ in nets]), nets[0][1]
    c = torch.load(f'{MOD}/{rung}.pt', weights_only=False)
    a = c['args']
    net = M.Net(c['n_seq'], c['n_stat'], a['h'], a['c'], a['d'])
    net.load_state_dict(c['net'])
    return net, c['ns']


def score(rung, frozen):
    net, ns = load_net(rung)
    val = load_bank(frozen)
    sm = sim_metrics(net, ns, val)
    recs = [r for r in load_records(f'{REAL}/dev_records_package.pkl') if r is not None]
    dm = dev_metrics(net, ns, recs)
    return sm, dm


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('rung'); ap.add_argument('desc'); ap.add_argument('--frozen', default='val_v1')
    ap.add_argument('--force-keep', action='store_true', help='first rung only')
    a = ap.parse_args()
    R = json.load(open(RJ)) if os.path.exists(RJ) else dict(rule=__doc__, rungs=[], incumbent=None)
    sm, dm = score(a.rung, a.frozen)
    entry = dict(rung=a.rung, desc=a.desc, frozen=a.frozen, sim_primary=sm['primary'],
                 sim={k: v for k, v in sm.items() if k != 'primary'}, dev=dm)
    inc = R['incumbent']
    if inc is None or a.force_keep:
        entry.update(decision='keep', reason='first rung (baseline network)')
        R['incumbent'] = a.rung
    else:
        ism, idm = score(inc, a.frozen)
        entry['incumbent'] = dict(rung=inc, sim_primary=ism['primary'], dev_composite=idm['composite'])
        d_sim = sm['primary'] - ism['primary']
        d_dev = dm['composite'] - idm['composite']
        ok = d_sim > 0.01 and d_dev >= -0.05
        entry.update(decision='keep' if ok else 'reject',
                     reason=f'sim {sm["primary"]:.4f} vs {ism["primary"]:.4f} (d={d_sim:+.4f}, need > +0.01); '
                            f'dev composite {dm["composite"]:.3f} vs {idm["composite"]:.3f} (d={d_dev:+.3f}, need >= -0.05)')
        if ok:
            R['incumbent'] = a.rung
    R['rungs'].append(entry)
    jdump(R, RJ)
    print(json.dumps({k: entry[k] for k in ('rung', 'decision', 'reason', 'sim_primary')}), json.dumps(dm))
