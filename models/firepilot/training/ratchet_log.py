"""Append a rung to ratchet.json: python3 ratchet_log.py <rung> <change> <kept|rejected> <reason> [incumbent]"""
import json, os, sys, time
import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
from eval_rung import summary
ROOT = _paths.WORK
fn = os.path.join(ROOT, 'ratchet.json')
R = json.load(open(fn)) if os.path.exists(fn) else dict(
    rule='keep a rung if frozen-sim val mean saved fraction beats the incumbent by > 0.01 AND the real dev gate '
         '(12 dev fires x 2 budgets, sim on real landscape+weather) mean saved fraction is no more than 5 % '
         '(relative) worse than the incumbent', frozen_set='val_p1 (160 synthetic scenarios, physics p1)',
    gate='gate_p1 (24 cases)', rungs=[])
rung, change, decision, reason = sys.argv[1:5]
v, g = summary(rung, 'val'), summary(rung, 'gate')
R['rungs'] = [r for r in R['rungs'] if r['rung'] != rung]
R['rungs'].append(dict(rung=rung, change=change, decision=decision, reason=reason, val=v, gate=g,
                       incumbent=sys.argv[5] if len(sys.argv) > 5 else None, logged=time.strftime('%Y-%m-%d %H:%M')))
json.dump(R, open(fn, 'w'), indent=1)
print(rung, decision, 'val %.3f gate %.3f' % (v['saved_mean'], g['saved_mean']))
