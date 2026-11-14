"""Build the calibrated RoadSim twins of every VED dev / exam hybrid car once (hp_env.build_twins), cached in WORK.
usage: build_twins.py [dev|exam ...]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hp_env import build_twins, HYBRID_ARCHS

if __name__ == '__main__':
    import realdata as rd
    vids = []
    for split in sys.argv[1:] or ['dev', 'exam']:
        idx = rd.load_trips(split, arch=[a for a in HYBRID_ARCHS if a != 'hev_series'], label_ok=True, meta_only=True)
        vids += sorted(set(idx.vehicle_id))
    t = build_twins(vids, workers=int(os.environ.get('HP_WORKERS', '5')))
    print(len(t), 'twins')
