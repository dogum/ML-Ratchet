"""RoadSim physics v2 stays reproducible by name: simulate(..., options='roadsim-physics-v2') must reproduce the v2
behaviour fingerprint (tests/golden/roadsim_v2_fingerprint.json, written 14 Nov 2026) to 1e-9. Later presets
(roadsim-physics-v2.1, ...) may change hybrid and plug-in runs only: every conventional, BEV, bus and Class 8 run of a
later preset must match the v2 golden to 1e-9 too (the v2.1 guard, shared/vehicle/roadsim/CALIBRATION.md).
Physics v2.1 reproduces its own golden (tests/golden/roadsim_v2_1_fingerprint.json, written 14 Nov 2026 before any v3
change) to 1e-9; v3 (the hardware pass) is a later preset under the same non-hybrid guard."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'checks'))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim'))

LATER = ['roadsim-physics-v2.1', 'roadsim-physics-v3']


def _gold():
    import fingerprint as fp
    return json.load(open(fp.GOLDS['roadsim-physics-v2'], encoding='utf-8'))


def test_roadsim_reproduces_v2_fingerprint():
    import fingerprint as fp
    bad = fp.compare(_gold(), fp.run('roadsim-physics-v2'))
    assert not bad, f'{len(bad)} runs changed under roadsim-physics-v2:\n' + '\n'.join(bad[:20])


def test_later_presets_leave_non_hybrids_at_v2():
    import fingerprint as fp
    from roadsim.options import PRESETS
    for name in LATER:
        if name not in PRESETS:
            continue
        bad = fp.compare(_gold(), fp.run(name), keys=lambda k: not fp.is_hybrid_key(k))
        assert not bad, f'{name}: {len(bad)} non-hybrid runs differ from v2:\n' + '\n'.join(bad[:20])


def test_roadsim_reproduces_v2_1_fingerprint():
    import fingerprint as fp
    gold = json.load(open(fp.GOLDS['roadsim-physics-v2.1'], encoding='utf-8'))
    bad = fp.compare(gold, fp.run('roadsim-physics-v2.1'))
    assert not bad, f'{len(bad)} runs changed under roadsim-physics-v2.1:' + chr(10) + chr(10).join(bad[:20])
