"""The seeded showcase draw for the page: 8 exam trips, one per stratum, chosen by seed and trip attributes only.
python3 showcase.py   -> showcase.json (frozen; done once in Phase 1)

Reads only RoadData's trip index columns that describe the input (architecture, distance, duration, mean speed,
mean temperature, climb, quality flag) for VED exam trips and the ZTBus v1 leg table (distance); never a label, never
a result. Strata (eligible: q_label_ok, >= 3 km and >= 300 s):
  city      conventional, mean speed < 30 km/h          highway   conventional, mean speed > 70 km/h
  hilly     conventional or hybrid, climb >= 15 m/km    cold      conventional, mean temperature < -10 C
  hybrid    power-split hybrid                          ev        battery-electric (2013 Leaf)
  plugin    plug-in hybrid (either kind)                bus       a ZTBus exam leg (78 traced legs), >= 3 km
One trip per stratum, drawn with default_rng(config.SEED) in that order, never the same vehicle twice."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLS = ['trip_id', 'vehicle_id', 'split', 'architecture', 'distance_m', 'duration_s', 'v_mean', 'T_amb_mean',
        'elev_gain_m', 'q_label_ok']


def main():
    p = os.path.join(C.HARNESS, 'showcase.json')
    if os.path.exists(p):
        raise SystemExit('showcase.json exists (frozen)')
    import realdata as rd
    d = rd.ved_index()[COLS]
    d = d[(d.split == 'exam') & d.q_label_ok & (d.distance_m >= 3000) & (d.duration_s >= 300)].copy()
    d['kmh'] = d.v_mean * 3.6; d['climb_per_km'] = d.elev_gain_m / (d.distance_m / 1000)
    strata = [('city', (d.architecture == 'conv') & (d.kmh < 30)),
              ('highway', (d.architecture == 'conv') & (d.kmh > 70)),
              ('hilly', d.architecture.isin(['conv', 'hev_powersplit']) & (d.climb_per_km >= 15)),
              ('cold', (d.architecture == 'conv') & (d.T_amb_mean < -10)),
              ('hybrid', d.architecture == 'hev_powersplit'),
              ('ev', d.architecture == 'bev'),
              ('plugin', d.architecture.str.startswith('phev'))]
    rng = np.random.default_rng(C.SEED)
    used, trips = set(), []
    for name, m in strata:
        c = d[m & ~d.vehicle_id.isin(used)].sort_values('trip_id')
        r = c.iloc[int(rng.integers(len(c)))]
        used.add(r.vehicle_id)
        trips.append(dict(stratum=name, trip_id=r.trip_id, vehicle_id=r.vehicle_id, arch=r.architecture,
                          dist_km=round(float(r.distance_m) / 1000, 2), dur_s=float(r.duration_s), kmh=round(float(r.kmh), 1),
                          T=round(float(r.T_amb_mean), 1), climb_per_km=round(float(r.climb_per_km), 1), n_eligible=int(len(c))))
    z = pd.read_parquet(os.path.join(C.ROAD_DATA, 'proc', 'hd', 'ztbus_legs.parquet'))
    z = z[(z.split == 'exam') & z.q_label_ok & (z.distance_m >= 3000)].sort_values('trip_id')
    r = z.iloc[int(rng.integers(len(z)))]
    trips.append(dict(stratum='bus', trip_id=r.trip_id, vehicle_id=r.vehicle_id, arch='bev', dist_km=round(float(r.distance_m) / 1000, 2),
                      dur_s=float(r.duration_s), T=round(float(r.T_amb_mean), 1), n_eligible=int(len(z))))
    json.dump({'seed': C.SEED, 'rule': __doc__.split('\n', 2)[2], 'trips': trips}, open(p, 'w'), indent=1)
    for t in trips:
        print(t)


if __name__ == '__main__':
    main()
