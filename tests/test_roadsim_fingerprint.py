"""RoadSim v2 physics ships behind options that are off by default: with every option off, it must still reproduce
the v1 behaviour fingerprint (shared/vehicle/checks/fingerprint.py) to 1e-9."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'checks'))


def test_roadsim_reproduces_v1_fingerprint():
    import fingerprint as fp
    gold = json.load(open(fp.GOLD, encoding='utf-8'))
    bad = fp.compare(gold, fp.run())
    assert not bad, f'{len(bad)} runs changed with every v2 option off:\n' + '\n'.join(bad[:20])
