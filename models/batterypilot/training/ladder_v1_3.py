"""The v1.3 ladder (GridSim physics 1.2, 15 Nov 2026): a physics change re-takes the ladder.  Every logged decision
of v1.1 and of the v1.2 attempt is re-taken in order on the new frozen set v6 and the real dev gate, then new rungs.

    python ladder_v1_3.py      -> training/ratchet_v1_3.json (reads WORK/data/scores/<rung>__v6_frozen.json)

Physics 1.2 adds GridSim's real-time scarcity the day-ahead market cannot see (gridsim.market, default fit, fitted on
ERCOT 2018/2020/2022 prices); BatteryPilot's scarcity calibration was re-fitted on dev with it (calibration_v1_2.json),
the banks and the frozen set v6 rebuilt with the v4 seeds.  (A first build on the first GridSim fit, frozen set v5,
is logged in the NOTEBOOK as probes; the dev twin test sent the process back for two train-year refits.)
Rule unchanged: keep a rung if its frozen capture beats the incumbent's by more than 0.01 AND its dev capture is not
more than 0.01 below the incumbent's.  R0 is the starting incumbent.  R5 (the backbone alone) is a classical
reference, not a candidate.
Release (the improvement-pass rule, fixed before this ladder): the last incumbent is released as v1.3 only if its dev
capture is above v1.1's (0.6382) and it also beats v1.1 re-scored on the new frozen set by the ladder rule.
Commands run from training/ with GRIDSIM_PHYSICS=1.2, B=../work/data/banks; each scored by
`ratchet.py score <name> v6_frozen`.
"""
import json
import os

import _paths

ROOT = _paths.WORK
MARGIN = 0.01
FROZEN = 'v6_frozen'
RUNGS = [
    ('R0', 'rolling', 'Classical rolling-horizon LP on day-ahead prices (re-plans at 14:00, follows the plan)', None),
    ('R1', 'r1_v4', 'MLP value-curve policy trained on sim v1 (GridSim merit-order prices only); 45 path-years, 1,500 '
                    'steps. Its bank has no scarcity of any kind, so physics 1.2 leaves it bit-identical (v4_train_v1) '
                    'and the v1.1 network is re-scored, not retrained', 'train.py r1_v4 v4_train_v1 --steps 1500'),
    ('R2', 'r2_v6', 'Same network, trained on sim v2 with physics 1.2 and the v1.3 dev calibration: 80 ERCOT path-years '
                    '(v6_train_e p0000-p0079), 1,500 steps', 'train.py r2_v6 "$B/v6_train_e/p00[0-7][0-9].npz" --steps 1500'),
    ('R3', 'r3_v6', 'Train on all 120 ERCOT + 45 CAISO-like path-years (v6_train_e + v6_train_c), 2,500 steps',
     'train.py r3_v6 "$B/v6_train_e/*.npz,$B/v6_train_c/*.npz" --steps 2500'),
    ('R4', 'r4_v6', 'Fine-tune R3 on the same 165 path-years with 21-day windows, batch 64, learning rate 5e-4 '
                    '(900 steps)',
     'train.py r4_v6 "$B/v6_train_e/*.npz,$B/v6_train_c/*.npz" --init r3_v6 --steps 900 --window 504 --batch 64 --lr 5e-4'),
    ('R5', 'bb', 'Reference, not a candidate: the day-ahead value-curve backbone alone (backbone.py), executed by the '
                 'offer rule against the RT price; no network', None),
    ('R6', 'r6_v6', 'Hybrid: the network also reads the backbone curve and outputs a residual on it, kept '
                    'non-increasing by a running minimum; 80 ERCOT path-years, 1,500 steps',
     'train.py r6_v6 "$B/v6_train_e/p00[0-7][0-9].npz" --steps 1500 --hybrid'),
    ('R8', 'r8_v6', 'R6 with a strictly decreasing curve (each knot at least 0.01 below the last); 80 ERCOT '
                    'path-years, 1,500 steps',
     'train.py r8_v6 "$B/v6_train_e/p00[0-7][0-9].npz" --steps 1500 --hybrid --min-step 0.01'),
    ('R9', 'r9_v6', 'R8 on all 120 ERCOT + 45 CAISO-like path-years, 2,500 steps',
     'train.py r9_v6 "$B/v6_train_e/*.npz,$B/v6_train_c/*.npz" --steps 2500 --hybrid --min-step 0.01'),
]
NEW = [      # v1.3 rungs after the re-take
    ('R10', 'r3_v5', 'The R3 recipe (all 120 ERCOT + 45 CAISO-like path-years, 2,500 steps) trained on the v5 banks '
                     '(GridSim 1.2 first fit, superseded). Put on this ladder because its dev score (0.658) was seen '
                     'in the v5 probes: chosen with dev knowledge, judged by the rule',
     'train.py r3_v5 "$B/v5_train_e/*.npz,$B/v5_train_c/*.npz" --steps 2500'),
    ('R11', 'r11_v6', 'R9 fine-tuned with the R4 recipe: 21-day windows, batch 64, learning rate 5e-4, 900 steps, on '
                      'the same 165 path-years',
     'train.py r11_v6 "$B/v6_train_e/*.npz,$B/v6_train_c/*.npz" --hybrid --min-step 0.01 --init r9_v6 --steps 900 '
     '--window 504 --batch 64 --lr 5e-4'),
]
REF = ('bb',)


def score(name, frozen=FROZEN):
    fn = f'{ROOT}/data/scores/{name}__{frozen}.json'
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    cal = json.load(open(f'{ROOT}/data/calibration_v1_2.json'))
    out = dict(rule=__doc__, physics='GridSim 1.2 (1.1 + gridsim.market RT scarcity)',
               calibration=cal['calibrated'],
               frozen=f'{FROZEN} (24 ERCOT + 9 CAISO-like path-years; seeds 7001/7002 as v4_frozen; GridSim 1.2 '
                      f'default fit)', rungs=[],
               not_retaken=[dict(rung='R7', why='stopped in the v1.2 attempt for a flaw R8 fixed (flat pieces of the '
                                                'running-minimum curve); not re-run')])
    inc = None
    for rid, name, change, cmd in RUNGS + NEW:
        s = score(name)
        if s is None:
            out['rungs'].append(dict(rung=rid, name=name, change=change, status='not scored'))
            continue
        e = dict(rung=rid, name=name, change=change, command=cmd, frozen=FROZEN,
                 frozen_capture=s['frozen_capture'], dev_capture=s['dev_capture'], dev_by_set=s['dev_by_set'],
                 dev_by_duration=s.get('dev_by_duration'), frozen_cycles=s['frozen_cycles'],
                 dev_cycles=s['dev_cycles'])
        if inc is None:
            e.update(kept=True, why='starting incumbent')
        elif name in REF:
            e.update(kept=False, why='reference only: a classical policy, not a candidate network')
        else:
            df = s['frozen_capture'] - inc['frozen_capture']; dd = s['dev_capture'] - inc['dev_capture']
            ok = df > MARGIN and dd >= -MARGIN
            e.update(kept=bool(ok), vs=inc['rung'], d_frozen=round(df, 4), d_dev=round(dd, 4),
                     why=(f"frozen {s['frozen_capture']:.4f} vs {inc['rung']} {inc['frozen_capture']:.4f} "
                          f"({df:+.4f}{' > ' if df > MARGIN else ' <= '}{MARGIN}); dev {s['dev_capture']:.4f} vs "
                          f"{inc['dev_capture']:.4f} ({dd:+.4f}, gate {'passed' if dd >= -MARGIN else 'failed'}) -> "
                          f"{'kept, new incumbent' if ok else 'rejected'}"))
        out['rungs'].append(e)
        if e['kept']:
            inc = e
    out['incumbent'] = inc['rung'] if inc else None
    v11 = score('r2_v4')
    out['released'] = None
    if inc is not None and v11 is not None and inc['name'] != 'rolling':
        df = inc['frozen_capture'] - v11['frozen_capture']; dd = inc['dev_capture'] - v11['dev_capture']
        ok = df > MARGIN and dd >= -MARGIN and inc['dev_capture'] > 0.6382
        out['release_check'] = dict(
            v1_1=dict(name='r2_v4', frozen_capture=v11['frozen_capture'], dev_capture=v11['dev_capture']),
            d_frozen=round(df, 4), d_dev=round(dd, 4), release=bool(ok),
            why=(f"{inc['rung']} frozen {inc['frozen_capture']:.4f} vs v1.1 {v11['frozen_capture']:.4f} ({df:+.4f}); "
                 f"dev {inc['dev_capture']:.4f} vs v1.1 0.6382 ({dd:+.4f}) -> "
                 f"{'release as v1.3' if ok else 'not released: v1.1 stays'}"))
        if ok:
            out['released'] = inc['rung']
    # the seed-to-seed spread of the kept R8 (it cleared the margin by 0.0073): the same recipe with seeds 1 and 2
    seeds = {n: score(n) for n in ('r8_v6', 'r8_v6_s1', 'r8_v6_s2')}
    if all(seeds.values()):
        fz = [v['frozen_capture'] for v in seeds.values()]; dv = [v['dev_capture'] for v in seeds.values()]
        import numpy as np
        out['seed_spread_R8'] = dict(
            runs={n: dict(frozen=v['frozen_capture'], dev=v['dev_capture']) for n, v in seeds.items()},
            frozen_mean=round(float(np.mean(fz)), 4), frozen_sd=round(float(np.std(fz, ddof=1)), 4),
            dev_mean=round(float(np.mean(dv)), 4), dev_sd=round(float(np.std(dv, ddof=1)), 4),
            note='seed 0 is the released R8; all three clear R2 (0.7546) by more than 0.01 and pass its dev gate')
    # probes on the superseded frozen set v5 (GridSim 1.2 first fit, calibration_v1_2_firstfit.json); no decision
    out['probes_v5'] = {n: dict(frozen=v['frozen_capture'], dev=v['dev_capture'])
                        for n in ('rolling', 'r1_v4', 'r2_v5', 'r3_v5', 'bb', 'r2_v4', 'r3_v4', 'r4_v4', 'r4_v3',
                                  'r6_hyb', 'r8_hyb', 'r9_hyb') if (v := score(n, 'v5_frozen'))}
    out['v1_1_rungs_rescored_on_v6'] = {n: dict(frozen=v['frozen_capture'], dev=v['dev_capture'])
                                        for n in ('r2_v4', 'r3_v4', 'r4_v4', 'r4_v3', 'r6_hyb', 'r8_hyb', 'r9_hyb')
                                        if (v := score(n))}
    fn = os.path.join(_paths.SRC, 'ratchet_v1_3.json')
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    for e in out['rungs']:
        print(e['rung'], e.get('frozen_capture'), e.get('dev_capture'), e.get('kept'), e.get('why', e.get('status')))
    print('incumbent at the end:', out['incumbent'], '| release:', out.get('release_check', {}).get('why'))


if __name__ == '__main__':
    main()
