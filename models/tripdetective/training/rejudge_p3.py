"""v2.1 ladder (15 Nov 2026): the rule for OUTPUT-CONSTRAINT rungs, changed in the open after p3r17 was rejected.
A physics constraint applied to a trained network's outputs (no retraining) cannot raise the frozen-set skill when the
simulator never produces the case it fixes: no simulated hybrid on p3_val_v7 looks conventional to the logger (every
one has an engine-off share while moving above 0.08), so p3r17 moves the objective by +0.0002 only. For such rungs
the rule becomes: kept if the frozen-set objective does not fall, the dev gate passes (mean dev error ratio <= 1.05),
AND dev VED-type accuracy per trip rises by more than the margin (0.01). Network rungs keep the old rule.
The rule was written after the dev effect of p3r17 was known (probe_devtype.py, probe_offgate.py); the constraint's
threshold was set from simulation only. usage: python rejudge_p3.py  (annotates work/ratchet_p3.json)"""
from common import *
LOG = os.environ.get('TD_RATCHET', f'{ROOT}/ratchet_p3.json')
log = jload(LOG)
inc = None
for r in log:
    if r['ckpt'] == 'p3r17.pt':
        s_ok = r['objective'] >= inc['objective']
        d_rel = r['dev_rel'] <= 1.05
        t_up = r['dev']['vedtype_acc_trip'] - inc['dev']['vedtype_acc_trip']
        r['rule_v21'] = dict(accepted=bool(s_ok and d_rel and t_up > 0.01),
                             why=f"output-constraint rule: objective {r['objective']:.4f} vs {inc['objective']:.4f} (must not fall); dev errors "
                                 f"{100 * (r['dev_rel'] - 1):+.0f}%; dev VED-type accuracy per trip {r['dev']['vedtype_acc_trip']:.3f} vs "
                                 f"{inc['dev']['vedtype_acc_trip']:.3f} ({t_up:+.3f}, needs > +0.01); per car {r['dev']['vedtype_acc_car']:.3f} vs {inc['dev']['vedtype_acc_car']:.3f}")
        r['accepted'] = r['rule_v21']['accepted']
        print(r['ckpt'], r['rule_v21'])
    if r['accepted'] and r['ckpt'] != 'p3r17.pt': inc = r
jsave(log, LOG)
