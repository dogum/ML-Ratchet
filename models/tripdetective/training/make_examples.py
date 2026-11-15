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



def bus_example():
    """One ZTBus exam leg (fixed random draw, seed 7; CC BY 4.0, Widmer et al. 2023) for the bus reader in demo.py."""
    zl = rd.ztbus_full(); zl = zl[(zl.split == 'exam') & zl.q_label_ok & zl.q_route_ok].sort_values('trip_id')
    row = zl.iloc[int(np.random.default_rng(7).integers(len(zl)))]
    m, a = rd.load_trip(row.trip_id)
    T = np.where(np.isfinite(a['T_amb']), a['T_amb'], np.nanmean(a['T_amb'])); rho = np.where(np.isfinite(a['rho_air']) & (a['rho_air'] > 0.5), a['rho_air'], 1.2)
    info = dict(trip_id=row.trip_id, bus=row.vehicle_id, km=round(float(row.distance_m) / 1000, 2), passengers_mean=round(float(row.passengers_mean), 1),
                season=row.season, source='ZTBus (Widmer, Ritter, Onder, Scientific Data 2023), CC BY 4.0; processed by RoadData')
    np.savez_compressed(f'{ROOT}/release/examples/ZTBus_exam_leg.npz', info=np.array(json.dumps(info)), v=np.nan_to_num(a['v']).astype(np.float32),
                        power_W=np.nan_to_num(a['elec_power']).astype(np.float32), grade=np.nan_to_num(a['grade']).astype(np.float32),
                        T_amb=T.astype(np.float32), rho_air=rho.astype(np.float32), valid=a['valid'])
    print('bus example', info)


def bus_mission():
    """v2.1: every leg of the mission (one bus, one day) of bus_example()'s leg (same draw, seed 7), for the day-relative
    bus reader (TripDetectiveBus.read_mission). Legs without a passenger label are kept as context, truth None."""
    zl = rd.ztbus_full(); ex = zl[zl.split == 'exam']
    cand = ex[ex.q_label_ok & ex.q_route_ok].sort_values('trip_id')
    row = cand.iloc[int(np.random.default_rng(7).integers(len(cand)))]
    legs = ex[ex.mission == row.mission].sort_values('leg')
    arrs, metas = {}, []
    for k, (_, r) in enumerate(legs.iterrows()):
        m, a = rd.load_trip(r.trip_id)
        T = np.where(np.isfinite(a['T_amb']), a['T_amb'], np.nanmean(a['T_amb']) if np.isfinite(a['T_amb']).any() else 12.0)
        rho = np.where(np.isfinite(a['rho_air']) & (a['rho_air'] > 0.5), a['rho_air'], 1.2)
        for key, x in [('v', np.nan_to_num(a['v'])), ('power_W', np.nan_to_num(a['elec_power'])), ('grade', np.nan_to_num(a['grade'])),
                       ('T_amb', T), ('rho_air', rho), ('valid', a['valid'])]:
            arrs[f'leg{k}_{key}'] = np.asarray(x, np.float32)
        lab = bool(r.q_label_ok) and np.isfinite(r.passengers_mean)
        metas.append(dict(trip_id=r.trip_id, km=round(float(r.distance_m) / 1000, 2), minutes=round(float(r.duration_s) / 60, 1),
                          passengers_mean=round(float(r.passengers_mean), 1) if lab else None))
    info = dict(mission=row.mission, bus=row.vehicle_id, season=row.season, example_leg=row.trip_id, legs=metas,
                source='ZTBus (Widmer, Ritter, Onder, Scientific Data 2023), CC BY 4.0; processed by RoadData')
    np.savez_compressed(f'{ROOT}/release/examples/ZTBus_exam_mission.npz', info=np.array(json.dumps(info)), **arrs)
    print('bus mission', row.mission, len(metas), 'legs')


if __name__ == '__main__':
    main()
    bus_example()
    bus_mission()
