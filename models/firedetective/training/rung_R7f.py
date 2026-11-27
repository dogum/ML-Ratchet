"""Matched retrain of the blind-wind flag scaling (27 Nov 2026, audit follow-up; sim and dev only).

Two arms, each R6's recipe (3 seeds of the R3 network on train_v1 shards 0-30 + train_v2 shards 0-2 + train_v3
shards 0-59, the 1,340 + 108 + 2,652 sims R6's log lists; elongation input dropped; frozen set val_v3):
  R6c  control: the shipped scaling (the flag wind_hidden is 0 on every wind-visible row, so its sd floors at 1e-3
       and a blind row enters at z = 1000)
  R7f  the flag stored with sd 1 (train.py --flag-sd1)
Rule (ratchet.py): keep if the sim primary on val_v3 beats the incumbent R6 (released weights) by more than 0.01 AND
the dev composite is no more than 0.05 below R6's. R7f is the rung; R6c shows how much of any difference is the
rebuild.

python rung_R7f.py   -> training/rung_R7f.json
"""
from common import *
import ratchet
from train import load_bank, sim_metrics, dev_metrics
from real import load_records

load_net = ratchet.load_net                      # WORK/models/<rung>(.ens.json)
import rescore_release                           # noqa: E402  (patches ratchet.load_net to the released R6)
ratchet.load_net = load_net                      # undo: the ensembles below recurse through ratchet.load_net


def main():
    val = load_bank('val_v3')
    recs = [r for r in load_records(f'{REAL}/dev_records_package.pkl') if r is not None]
    out = {'what': 'R7f: R6 recipe with the blind-wind flag at sd 1; R6c: the same with the shipped scaling (27 Nov 2026)'}
    nets = {'R6': rescore_release._load_net('R6'), 'R6c': load_net('R6c'), 'R7f': load_net('R7f')}
    for name, (net, ns) in nets.items():
        sm = sim_metrics(net, ns, val)
        dm = dev_metrics(net, ns, recs)
        j = feats.SEQ_NAMES.index('wind_hidden')
        out[name] = {'sim_primary': sm['primary'], 'sim': sm, 'dev': dm, 'flag_sd': float(ns['seq_sd'][j])}
    inc = out['R6']
    for name in ('R6c', 'R7f'):
        d_sim = out[name]['sim_primary'] - inc['sim_primary']
        d_dev = out[name]['dev']['composite'] - inc['dev']['composite']
        ok = d_sim > 0.01 and d_dev >= -0.05
        out[name]['vs_R6'] = dict(d_sim=d_sim, d_dev=d_dev, passes_rule=ok,
                                  reason=f"sim {out[name]['sim_primary']:.4f} vs {inc['sim_primary']:.4f} (d={d_sim:+.4f}, "
                                         f"need > +0.01); dev composite {out[name]['dev']['composite']:.3f} vs "
                                         f"{inc['dev']['composite']:.3f} (d={d_dev:+.3f}, need >= -0.05)")
    out['decision'] = 'kept' if out['R7f']['vs_R6']['passes_rule'] else 'rejected'
    jdump(out, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rung_R7f.json'))
    for name in ('R6', 'R6c', 'R7f'):
        r = out[name]
        print(name, 'flag sd', r['flag_sd'], 'primary', round(r['sim_primary'], 4),
              {k: round(v['r2'], 3) for k, v in r['sim'].items() if isinstance(v, dict) and 'r2' in v},
              'dev', {k: round(v, 3) for k, v in r['dev'].items() if isinstance(v, float)})
    print(out['decision'], out['R7f']['vs_R6']['reason'])


if __name__ == '__main__':
    import feats  # noqa: E402
    main()
