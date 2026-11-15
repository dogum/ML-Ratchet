"""The v1.2 ladder (15 Nov 2026): the day-ahead value-curve backbone (backbone.py) with a learned residual, on the same
v4 frozen set and real dev gate as v1.1, with the release's rule unchanged.  The incumbent is v1.1 (R2).

    python ladder_v1_2.py      -> training/ratchet_v1_2.json (reads WORK/data/scores/<rung>__v4_frozen.json)

Rule (as released): keep a rung if its frozen capture beats the incumbent's by more than 0.01 AND its dev capture is
not more than 0.01 below the incumbent's.
"""
import json, os
import _paths

ROOT = _paths.WORK
MARGIN = 0.01
RUNGS = [
    ('R2', 'r2_v4', 'v1.1 as released (incumbent): MLP value curve, 80 ERCOT path-years, 1,500 steps', None),
    ('R5', 'bb', 'Reference, not a candidate: the day-ahead value-curve backbone alone (backbone.py), executed by the '
                 'offer rule against the RT price; no network', None),
    ('R6', 'r6_hyb', 'Hybrid: the network also reads the backbone curve (asinh(lambda/ps) at the 5 knots) and outputs a '
                     'residual on it, kept non-increasing by a running minimum; last layer zero at the start (= the '
                     'backbone); R2 recipe: 80 ERCOT path-years, 1,500 steps',
     'train.py r6_hyb v4_train_e/p00[0-7][0-9] --steps 1500 --hybrid'),
    ('R7', 'r7_hyb', 'R6 on all 120 ERCOT + 45 CAISO-like path-years, 2,500 steps (R3 recipe). Stopped at step 775: '
                     'its training capture had fallen to 0.21 against 0.37 for R3 on the same windows (flat pieces of '
                     'the running-minimum curve: the offer rule divides by the knot-to-knot drop). Not scored',
     'train.py r7_hyb v4_train_e,v4_train_c --steps 2500 --hybrid'),
    ('R8', 'r8_hyb', 'R6 with a strictly decreasing curve: each knot at least 0.01 (asinh units) below the previous one, '
                     'the floor the v1.0/v1.1 softplus steps use; 80 ERCOT path-years, 1,500 steps',
     'train.py r8_hyb v4_train_e/p00[0-7][0-9] --steps 1500 --hybrid --min-step 0.01'),
    ('R9', 'r9_hyb', 'R8 on all 120 ERCOT + 45 CAISO-like path-years, 2,500 steps (R3 recipe)',
     'train.py r9_hyb v4_train_e,v4_train_c --steps 2500 --hybrid --min-step 0.01'),
]


def score(name, frozen='v4_frozen'):
    fn = f'{ROOT}/data/scores/{name}__{frozen}.json'
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    out = dict(rule=__doc__, physics='GridSim 1.1', frozen='v4_frozen (as v1.1)', rungs=[])
    inc = None
    for rid, name, change, cmd in RUNGS:
        s = score(name)
        if s is None:
            out['rungs'].append(dict(rung=rid, name=name, change=change, status='not scored'))
            continue
        e = dict(rung=rid, name=name, change=change, command=cmd, frozen='v4_frozen',
                 frozen_capture=s['frozen_capture'], dev_capture=s['dev_capture'], dev_by_set=s['dev_by_set'],
                 dev_by_duration=s.get('dev_by_duration'), frozen_cycles=s['frozen_cycles'],
                 dev_cycles=s['dev_cycles'])
        if inc is None:
            e.update(kept=True, why='incumbent (v1.1 release)')
        elif name == 'bb':
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
    out['released'] = inc['rung'] if inc else None
    fn = os.path.join(_paths.SRC, 'ratchet_v1_2.json')
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    for e in out['rungs']:
        print(e['rung'], e.get('frozen_capture'), e.get('dev_capture'), e.get('kept'), e.get('why', e.get('status')))
    print('released:', out['released'])


if __name__ == '__main__':
    main()
