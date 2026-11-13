"""Print the NMC811 ladder (work/nmc/ratchet_nmc.json) as the ratchet.yaml block "v2 — NMC811 (LG M50T)", with the exam
columns reported after the fact (never used to choose)."""
import _paths as _P
import os, json, yaml

if __name__ == '__main__':
    log = json.load(open(os.path.join(_P.WORK, 'nmc', 'ratchet_nmc.json'))); so = log['scored_on']
    rungs = []
    for g in log['rungs']:
        rungs.append(dict(id=g['id'], change=g['change'], sim=round(g['sim'][so], 3),
                          dev={'partial_mAh': round(g['dev']['q_partial_mAh'], 1), 'full_mAh': round(g['dev']['q_full_mAh'], 1)},
                          decision=g['decision'], why=g['why'],
                          metrics=[round(g['exam']['q_partial_mAh'], 1), round(g['exam']['q_full_mAh'], 1)] + [round(g['sim'].get(s, float('nan')), 3) for s in log['frozen_sets']]))
    print(yaml.safe_dump({'rungs': rungs}, sort_keys=False, allow_unicode=True, width=200))
