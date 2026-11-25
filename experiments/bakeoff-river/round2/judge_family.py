"""PREREG section 4: each family's b2 model is one rung against its v1.0 arm.

usage: python judge_family.py <b2 bank name>      -> results/family_rungs.json
Kept if sim-val median NSE (both scored on b2's frozen val) > v1.0's + 0.01 AND dev median NSE >= 0.95 x v1.0's.
Reads work/eval/<run>/own_summary.json (b2 rung: b2 val + dev) and work/eval/v1_<arm>/{onb2,dev}_summary.json.
"""
import json
import os
import sys

import r2

FAMILIES = {   # arm id: (round-1 folder, b2 run name pattern, b2 arm id)
    'tab_gbm': ('arms_tab/tab_gbm', '{b}_gbm', 'tab_gbm_b2'),
    'seq_lstm': ('arms_seq/lstm', '{b}_lstm_r1_s0', 'seq_lstm_b2'),
    'seq_tcn': ('arms_seq/tcn', '{b}_tcn_r1_s0', 'seq_tcn_b2'),
    'seq_ssm': ('arms_seq/ssm', '{b}_ssm_r2_s0', 'seq_ssm_b2'),
}


def s(path, part, key='nse'):
    return json.load(open(path))[part]['ungauged'][key]


def main(bank):
    out = {'bank': bank, 'rule': 'keep if sim val NSE (b2 frozen val, both) > v1.0 + 0.01 AND dev >= 0.95 x v1.0 dev',
           'families': {}}
    for arm, (v1dir, pat, newid) in FAMILIES.items():
        run = pat.format(b=bank)
        e = os.path.join(r2.WORK, 'eval', run, 'own_summary.json')
        v = os.path.join(r2.WORK, 'eval', f'v1_{arm}')
        try:
            rec = {'run': run, 'b2_arm_id': newid,
                   'b2_simval_on_b2': s(e, 'simval'), 'b2_dev': s(e, 'dev'), 'b2_dev_kge': s(e, 'dev', 'kge'),
                   'v1_simval_on_b2': s(os.path.join(v, 'onb2_summary.json'), 'simval'),
                   'v1_dev': s(os.path.join(v, 'dev_summary.json'), 'dev'),
                   'v1_dev_kge': s(os.path.join(v, 'dev_summary.json'), 'dev', 'kge')}
        except FileNotFoundError as x:
            print(arm, 'incomplete:', x)
            continue
        if os.path.exists(os.path.join(r2.WORK, 'eval', run, 'p1val_summary.json')):
            rec['b2_simval_on_p1'] = s(os.path.join(r2.WORK, 'eval', run, 'p1val_summary.json'), 'simval')
        dsim = rec['b2_simval_on_b2'] - rec['v1_simval_on_b2']
        rec['sim_gain'] = dsim
        rec['dev_ratio'] = rec['b2_dev'] / rec['v1_dev']
        rec['kept'] = bool(dsim > 0.01 and rec['b2_dev'] >= 0.95 * rec['v1_dev'])
        out['families'][arm] = rec
        print(f"{arm}: sim on b2 {rec['b2_simval_on_b2']:.4f} vs v1.0 {rec['v1_simval_on_b2']:.4f} ({dsim:+.4f}); "
              f"dev {rec['b2_dev']:.4f} vs {rec['v1_dev']:.4f} (x{rec['dev_ratio']:.3f}) -> "
              f"{'KEPT' if rec['kept'] else 'rejected'}")
    json.dump(out, open(os.path.join(r2.RESULTS, 'family_rungs.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
