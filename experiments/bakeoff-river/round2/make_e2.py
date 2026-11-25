"""Assemble the E2 arm folder (round2/arms/ens_e2/) from results/family_rungs.json: each family's b2 arm if kept,
else its v1.0 arm; plus FlowCast v1.0. Same combination code as E1 (ens_predict_template.py). usage: python make_e2.py
"""
import json
import os
import shutil

import r2

V1 = {'tab_gbm': '../../../arms_tab/tab_gbm', 'seq_lstm': '../../../arms_seq/lstm', 'seq_tcn': '../../../arms_seq/tcn',
      'seq_ssm': '../../../arms_seq/ssm'}
FC = '../../../harness/arms_ref/flowcast_v1'


def main():
    fr = json.load(open(os.path.join(r2.RESULTS, 'family_rungs.json')))['families']
    members, ids = [], []
    for arm in ('tab_gbm', 'seq_lstm', 'seq_tcn', 'seq_ssm'):
        if arm == 'seq_lstm' or arm == 'tab_gbm' or arm in fr:
            pass
        rec = fr.get(arm)
        if rec is None:
            raise SystemExit(f'{arm}: family rung not judged yet')
        if rec['kept']:
            members.append(f'../{rec["b2_arm_id"]}'); ids.append(rec['b2_arm_id'])
        else:
            members.append(V1[arm]); ids.append(arm)
    members.insert(1, FC); ids.insert(1, 'flowcast_v1')
    d = os.path.join(r2.ROUND2, 'arms', 'ens_e2')
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(r2.ROUND2, 'ens_predict_template.py'), os.path.join(d, 'predict.py'))
    json.dump({'members': members, 'ids': ids, 'weights': 'equal (1/5)'}, open(os.path.join(d, 'members.json'), 'w'),
              indent=1)
    extra = ['members.json'] + members
    if any(m.startswith('../../../arms_seq') or m.startswith('../seq_') for m in members):
        extra.append('../../../arms_seq/common')
    print(json.dumps({'members': members, 'ids': ids, 'extra_files': extra}, indent=1))
    return members, ids, extra


if __name__ == '__main__':
    main()
