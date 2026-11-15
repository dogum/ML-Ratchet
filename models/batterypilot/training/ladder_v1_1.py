"""The v1.1 ladder (GridSim physics 1.1, leak fix of 14 Nov 2026): re-take the logged decisions in order on the
v4 frozen set and the real dev gate, with the release's rule unchanged.

    python ladder_v1_1.py      -> training/ratchet_v1_1.json (reads WORK/data/scores/<rung>__v4_frozen.json,
                                  WORK/data/leak/*.json and the frozen-set scores of the leak measurement)

Rule (as released): keep a rung if its frozen capture beats the incumbent's by more than 0.01 AND its dev capture is
not more than 0.01 below the incumbent's.  R0 is the starting incumbent.
"""
import json, os
import _paths

ROOT = _paths.WORK
MARGIN = 0.01
RUNGS = [
    ('R0', 'rolling', 'Classical rolling-horizon LP on day-ahead prices (re-plans at 14:00, follows the plan)', None),
    ('R1', 'r1_v4', 'MLP value-curve policy trained by policy search on sim v1 (GridSim merit-order prices, no '
                    'scarcity curve, no outage weather, no RT noise); 45 path-years (v4_train_v1), 1,500 steps',
     'train.py r1_v4 v4_train_v1 --steps 1500'),
    ('R2', 'r2_v4', 'Same network and policy search, trained on sim v2 with the v1.1 dev calibration: 80 ERCOT '
                    'path-years (v4_train_e p0000-p0079), 1,500 steps',
     'train.py r2_v4 v4_train_e/p00[0-7][0-9] --steps 1500'),
    ('R3', 'r3_v4', 'Train on all 120 ERCOT path-years + 45 CAISO-like path-years (v4_train_e + v4_train_c), 2,500 '
                    'steps', 'train.py r3_v4 v4_train_e,v4_train_c --steps 2500'),
    ('R4', 'r4_v4', 'Fine-tune R3 on the same 165 path-years with 21-day windows, batch 64, learning rate 5e-4 '
                    '(900 steps)',
     'train.py r4_v4 v4_train_e,v4_train_c --init r3_v4 --steps 900 --window 504 --batch 64 --lr 5e-4'),
]


def score(name, frozen='v4_frozen'):
    fn = f'{ROOT}/data/scores/{name}__{frozen}.json'
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    out = dict(rule=__doc__, physics='GridSim 1.1 (outage defaults from pre-2023 sources; coal adder -5.73 $/MWh)',
               calibration=json.load(open(f'{ROOT}/data/calibration_v1_1.json'))['calibrated'],
               frozen='v4_frozen (24 ERCOT + 9 CAISO-like path-years; seeds 7001/7002, the same as the leak '
                      'measurement sets m10_frozen and m11rc_frozen)', rungs=[])
    inc = None
    for rid, name, change, cmd in RUNGS:
        s = score(name)
        if s is None:
            out['rungs'].append(dict(rung=rid, name=name, change=change, status='not scored'))
            continue
        e = dict(rung=rid, name=name, change=change, command=cmd, frozen='v4_frozen',
                 frozen_capture=s['frozen_capture'], dev_capture=s['dev_capture'], dev_by_set=s['dev_by_set'],
                 dev_by_duration=s.get('dev_by_duration'), frozen_cycles=s['frozen_cycles'], dev_cycles=s['dev_cycles'])
        if inc is None:
            e.update(kept=True, why='starting incumbent')
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
    ref = score('r4_v3')
    if ref:
        out['reference_released_v1_0_R4_on_v4_frozen'] = {k: ref[k] for k in ('frozen_capture', 'dev_capture',
                                                                              'frozen_cycles', 'dev_by_duration')}
    leak = {}
    for t in ('rel_calib_physics1.0', 'rel_calib_physics1.1', 'v1_1_calib_physics1.1'):
        fn = f'{ROOT}/data/leak/{t}.json'
        if os.path.exists(fn):
            d = json.load(open(fn))
            leak[t] = {k: d[k] for k in ('physics', 'coal_adder', 'calib', 'stageA_loss', 'stats', 'real', 'uri',
                                         'outages', 'twin')}
    for fz in ('m10_frozen', 'm11rc_frozen', 'v4_frozen'):
        for r in ('rolling', 'r4_v3'):
            s = score(r, fz)
            if s:
                leak.setdefault('frozen', {}).setdefault(fz, {})[r] = dict(frozen_capture=s['frozen_capture'],
                                                                           dev_capture=s['dev_capture'],
                                                                           frozen_cycles=s['frozen_cycles'])
    out['leak_measurement'] = leak
    fn = os.path.join(_paths.SRC, 'ratchet_v1_1.json')
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    for e in out['rungs']:
        print(e['rung'], e.get('frozen_capture'), e.get('dev_capture'), e.get('kept'), e.get('why', e.get('status')))
    print('released:', out['released'])


if __name__ == '__main__':
    main()
