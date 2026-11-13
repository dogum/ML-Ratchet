"""Write a few real VED exam trips (fixed random draw within VED types, seed 7; Apache-2.0 data) to release/examples/
in the trip contract, for demo.py."""
from common import *
import realdata as rd

KEYS = ['t', 'v', 'grade', 'T_amb', 'wind_speed', 'wind_dir', 'heading', 'rho_air', 'fuel_rate', 'engine_rpm', 'batt_current',
        'batt_voltage', 'soc', 'aux_power', 'valid']


def main(ntr=5):
    out = f'{ROOT}/release/examples'; os.makedirs(out, exist_ok=True)
    vt = rd.ved_vehicles(); rng = np.random.default_rng(7); idx = rd.ved_index()
    for t in ['ICE', 'HEV', 'PHEV', 'EV']:
        cars = sorted(idx[(idx.split == 'exam') & (idx.ved_type == t) & idx.q_label_ok].groupby('vehicle_id').size().loc[lambda s: s >= ntr].index)
        vid = cars[int(rng.integers(len(cars)))]
        arrs = {}; metas = []
        for k, (m, a) in enumerate(rd.load_trips('exam', vehicles=[vid], label_ok=True, limit=ntr, seed=7)):
            for key in KEYS: arrs[f'trip{k}_{key}'] = a[key].astype(np.float32)
            metas.append(dict(trip_id=m['trip_id'], fuel_source=m['fuel_source'], km=round(m['distance_m'] / 1000, 2)))
        r = vt.loc[vid]
        info = dict(vehicle_id=vid, published=dict(ved_type=r.ved_type, weight_class_lb=None if not np.isfinite(r.gen_weight_lb) else float(r.gen_weight_lb),
                                                   displacement_L=None if not np.isfinite(r.displacement_L) else float(r.displacement_L), engine=r.engine_str),
                    trips=metas, source='Vehicle Energy Dataset (Oh, LeBlanc, Peng 2020), Apache-2.0; processed by RoadData')
        np.savez_compressed(f'{out}/{vid}.npz', info=np.array(json.dumps(info)), **arrs)
        print(vid, t, len(metas))


if __name__ == '__main__':
    main()
