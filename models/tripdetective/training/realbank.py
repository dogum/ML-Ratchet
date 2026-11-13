"""Real VED trips -> the same raw channels as the bank. Trips longer than L s are cut into consecutive windows
(the network pools windows like trips). A fixed random draw of at most N trips per car (seed 0), never cherry-picked.
usage: python3 realbank.py dev|exam [max_trips_per_car]"""
from common import *
import realdata as rd
import features as F

FSRC_EST = {'maf_x_trims_over_afr': True, 'absload_rpm_displacement': True, 'obd_fuel_rate_Lph': False, 'none': False}


def flags_for(m, a):
    fin = lambda k: float(np.isfinite(a[k]).mean()) > 0.5
    return dict(has_fuel=m.get('fuel_source', 'none') != 'none' and fin('fuel_rate'),
                fuel_est=FSRC_EST.get(m.get('fuel_source', 'none'), False),
                has_rpm=fin('engine_rpm'), has_batt=fin('batt_current') and fin('batt_voltage'), has_aux=fin('aux_power'))


def windows(n, Lw=L, min_len=120):
    if n <= Lw: return [(0, n)]
    k = int(np.ceil(n / Lw)); edges = np.linspace(0, n, k + 1).astype(int)
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= min_len]


def build(split, maxper=40, out=None):
    vt = rd.ved_vehicles()
    vids = vt[vt.split == split].index.tolist()
    R, offs, FL, TRIP, VID, WIN, META = [], [0], [], [], [], [], []
    for vid in vids:
        for m, a in rd.load_trips(split, vehicles=[vid], label_ok=True, limit=maxper, seed=0):
            fl = flags_for(m, a)
            raw = F.raw_from_log(a, fl)
            for w, (s, e) in enumerate(windows(raw.shape[1])):
                R.append(raw[:, s:e].T.astype(np.float16)); offs.append(offs[-1] + e - s)
                FL.append(F.flags_vec(fl)); TRIP.append(m['trip_id']); VID.append(vid); WIN.append(w)
            META.append(dict(trip_id=m['trip_id'], vid=vid, km=m['distance_m'] / 1000, T=m['T_amb_mean'], fuel_source=m['fuel_source'],
                             fuel_g=m.get('fuel_g'), kWh=m.get('elec_kWh_net_discharge'), dur=m['duration_s']))
    fn = out or f'{ROOT}/data/real_{split}.npz'
    np.savez(fn, raw=np.concatenate(R), offs=np.array(offs), flags=np.array(FL), trip=np.array(TRIP), vid=np.array(VID),
             win=np.array(WIN), meta=np.array(json.dumps(META)))
    print(split, len(vids), 'cars', len(META), 'trips', len(TRIP), 'windows')


def load(split):
    z = np.load(f'{ROOT}/data/real_{split}.npz')
    return {k: z[k] for k in z.files if k != 'meta'} | {'meta': json.loads(str(z['meta']))}


if __name__ == '__main__':
    build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
