"""Buses (Phase D): simulated electric city buses for TripDetective's bus mass ladder, and the real ZTBus legs.

A bus is shown to the network as a scaled car: power / BUS_SCALE, mass / BUS_SCALE, CdA / BUS_SCALE (the road-load
equation is linear in them, so a 21 t bus at 300 kW reads like a 1,750 kg car at 25 kW). The bus network has its own
weights (tripdetective_bus_weights.npz); the light-duty one never sees buses.

Simulated buses: an articulated or 12 m battery-electric bus (RoadSim class 'bus', architecture 'bev'), curb mass
log-uniform 14-24 t, passengers 0-110 x 70 kg (RoadData's ZTBus convention), CdA 5-8 m2, Crr 0.006-0.010, traction
machine 250-400 kW, accessories 2-8 kW, a PTC heater of 6-24 kW (the v2 bus PTC saturates at 24 kW; ZTBus dev buses
draw ~9 kW at stops in winter), heat set point 16-22 C, series regeneration. Routes: speed, GNSS grade and temperature
of ZTBus DEV legs from the 'route' half of the dev missions (the other half is the dev gate; no label is used),
plus CBD14 windows and synthetic stop-and-go. Logged like a trolleybus: total DC power (traction + accessories) each
second, held 1 s, quantised 100 W, no SOC, no fuel, no rpm.
Real ZTBus legs: elec_power -> the battery channel; truth mass = 19,000 kg + 70 kg x mean passengers of the leg
(check_ztbus.py convention). Dev missions only for choices; the 1,115 exam missions once, at the end.
usage: python busbank.py sim <tag> <n_buses> <seed> [K=6] [prior=b1]  |  python busbank.py real dev|exam"""
from common import *
import copy, time, hashlib
from roadsim import simulate
import features as F, routes

BUS_SCALE = 12.0
CURB_KG = 19000.0
PAX_KG = 70.0
CHUNK = 50


def mission_half(mission):
    """dev missions split in two by a hash of the mission id: 'route' (sim routes) and 'gate' (dev gate)"""
    return 'route' if int(hashlib.sha256(mission.encode()).hexdigest(), 16) % 2 == 0 else 'gate'


_POOL = None


def route_pool():
    """1,024 s windows of ZTBus dev legs from the route half (speed, grade, temperature only)"""
    global _POOL
    if _POOL is None:
        fn = f'{ROOT}/data/bus_route_pool.npz'
        if not os.path.exists(fn):
            import realdata as rd
            W = []
            for m, a in rd.load_trips('dev', source='ZTBus_full', label_ok=True, route_ok=True):
                if mission_half(m['mission']) != 'route': continue
                n = len(a['v'])
                for s0 in range(0, n - 300, L):
                    e = min(n, s0 + L)
                    W.append(np.stack([np.nan_to_num(a['v'][s0:e]), np.nan_to_num(a['grade'][s0:e]),
                                       np.where(np.isfinite(a['T_amb'][s0:e]), a['T_amb'][s0:e], np.nanmean(a['T_amb'])),
                                       np.where(np.isfinite(a['rho_air'][s0:e]) & (a['rho_air'][s0:e] > 0.5), a['rho_air'][s0:e], 1.2)]).astype(np.float32))
            lens = np.array([w.shape[1] for w in W]); offs = np.r_[0, np.cumsum(lens)]
            np.savez(fn, data=np.concatenate(W, 1), offs=offs)
            print('bus route pool', len(W), 'windows', flush=True)
        z = np.load(fn); _POOL = (z['data'], z['offs'])
    return _POOL


def bus_route(rng):
    u = rng.random()
    if u < 0.75:
        d, o = route_pool(); i = int(rng.integers(len(o) - 1)); w = d[:, o[i]:o[i + 1]]
        n = w.shape[1]
        r = dict(v=w[0].astype(float), grade=w[1].astype(float), T_amb=w[2].astype(float), rho_air=w[3].astype(float))
        src = 'ztbus_dev'
    else:
        if u < 0.85:
            from roadsim import cycle
            v = np.asarray(cycle('cbd14')['v'] if isinstance(cycle('cbd14'), dict) else cycle('cbd14'), float)
            reps = int(np.ceil(L / len(v))) + 1; v = np.tile(v, reps); s0 = int(rng.integers(0, len(v) - L)); v = v[s0:s0 + int(rng.integers(400, L + 1))]
            src = 'cbd14'
        else:
            v, _ = routes.synthetic_route(rng)
            v = np.minimum(v['v'], rng.uniform(11, 16)); src = 'synthetic'
        n = len(v); d = np.cumsum(v)
        r = dict(v=v, grade=routes._grade_profile(rng, d, rng.uniform(0.005, 0.03)), T_amb=np.full(n, rng.uniform(-2, 32)), rho_air=np.full(n, 1.2))
    n = len(r['v'])
    r.update(elev=400 + np.cumsum(r['grade'] * r['v']), heading=np.zeros(n), wind_speed=np.zeros(n), wind_dir=np.zeros(n),
             surface=np.zeros(n, int))
    return r, src


def draw_bus(rng, prior='b1'):
    curb = float(np.exp(rng.uniform(np.log(14000), np.log(24000))))
    pax = float(rng.uniform(0, 110))
    s = {'id': 'bus', 'class': 'bus', 'architecture': 'bev', 'mass_kg': round(curb, 1), 'payload_kg': round(PAX_KG * pax, 1),
         'CdA_m2': round(float(rng.uniform(5.0, 8.0)), 3), 'Crr': round(float(rng.uniform(0.006, 0.010)), 5), 'Crr_speed_coef': 0.002,
         'wheel_radius_m': 0.5, 'motors': [{'role': 'traction', 'max_power_kW': round(float(rng.uniform(250, 400))), 'eff_map': 'ipm_generic_150', 'max_rpm': 6000}],
         'battery': {'chemistry': 'LFP', 'capacity_kWh': 200, 'nominal_V': 650, 'soc_window': [0.05, 0.98]},
         'aux': {'base_W': round(float(rng.uniform(2000, 8000))), 'HVAC': {'enabled': True, 'ptc_max_W': round(float(rng.uniform(6000, 24000))),
                                                                          'heat_on_below_C': round(float(rng.uniform(8, 18)), 1)}},
         'physics': {'hvac_T_set': round(float(rng.uniform(16, 22)), 1)},
         'control': {'strategy': 'bev_regen_series', 'params': {}}}
    return s, pax


def truth(spec, pax):
    m = float(spec['mass_kg']) + float(spec['payload_kg'])
    return dict(arch='bev', cls='bus', strategy='bev_regen_series', mass=m / BUS_SCALE, mass_kg=m, curb_kg=float(spec['mass_kg']), pax=pax,
                disp=np.nan, CdA=float(spec['CdA_m2']) / BUS_SCALE, Crr=float(spec['Crr']), ev_frac=np.nan, P_on_kW=np.nan, v_ev=np.nan)


FL_BUS = dict(has_fuel=False, fuel_est=False, has_rpm=False, has_batt=True, has_aux=False)


def bus_raw(v, grade, T, rho, P_W, valid=None):
    """1 Hz bus log -> raw channels (power / BUS_SCALE)"""
    n = len(v)
    a = dict(v=v, grade=grade, T_amb=T, rho_air=rho, wind_speed=np.zeros(n), wind_dir=np.zeros(n), heading=np.zeros(n),
             batt_current=np.asarray(P_W, float) / 1000.0 / BUS_SCALE, batt_voltage=np.full(n, 1000.0), soc=np.full(n, 0.5),
             fuel_rate=np.full(n, np.nan), engine_rpm=np.full(n, np.nan), aux_power=np.full(n, np.nan),
             valid=np.ones(n) if valid is None else valid)
    return F.raw_from_log(a, FL_BUS)


def build_sim(tag, nbus, seed, K=6, prior='b1'):
    d = f'{ROOT}/bank/{tag}'; os.makedirs(d, exist_ok=True)
    try:
        if not os.path.exists(f'{d}/info.json'):
            jsave(dict(tag=tag, nveh=nbus, seed=seed, K=K, prior=prior, physics=PHYSICS, bus_scale=BUS_SCALE), f'{d}/info.json')
    except OSError:
        pass
    for c0 in range(0, nbus, CHUNK):
        fn = f'{d}/chunk_{c0 // CHUNK:03d}.npz'
        if os.path.exists(fn): continue
        try:
            fd = os.open(fn[:-4] + '.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time(); rng = np.random.default_rng([seed, c0])
        R, offs, FL, LED, TV, KM, EN, META = [], [0], [], [], [], [], [], []
        for i in range(c0, min(nbus, c0 + CHUNK)):
            spec, pax = draw_bus(rng, prior)
            tr = truth(spec, pax); tr['vid'] = i; tr['spec'] = spec
            for k in range(K):
                r, src = bus_route(rng)
                T0 = float(np.mean(r['T_amb']))
                out = simulate(copy.deepcopy(spec), r, init_soc=0.6, init_temps=dict(cabin=20.0, battery=T0), rng=int(rng.integers(1 << 30)),
                               options=PHYSICS, check=False)
                P = np.asarray(out['batt_current'], float) * np.asarray(out['batt_voltage'], float)
                P = np.round(P / 100) * 100
                raw = bus_raw(np.round(np.asarray(out['v'], float) * 3.6 * 10) / 36, r['grade'], r['T_amb'], r['rho_air'], P)
                R.append(raw.T.astype(np.float16)); offs.append(offs[-1] + raw.shape[1]); FL.append(F.flags_vec(FL_BUS))
                LED.append(np.zeros(6, np.float32)); TV.append(i); KM.append(out['summary']['distance_m'] / 1000)
                EN.append([0, 0, T0, tr['mass']])
            META.append(tr)
        np.savez(fn, raw=np.concatenate(R).astype(np.float16), offs=np.array(offs, np.int64), flags=np.array(FL), ledger=np.array(LED),
                 veh=np.array(TV), km=np.array(KM), energy=np.array(EN, np.float32),
                 meta=np.array(json.dumps(META, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))))
        os.remove(fn[:-4] + '.lock')
        print(f'bus chunk {c0 // CHUNK} {time.time() - t0:.0f}s', flush=True)


def build_real(split, maxlegs=None, seed=0):
    """ZTBus legs -> windows. dev: the gate half of the dev missions; exam: every exam leg with a usable label."""
    import realdata as rd
    zl = rd.ztbus_full(); zl = zl[(zl.split == split) & zl.q_label_ok]
    if split == 'dev':
        zl = zl[zl.mission.map(mission_half) == 'gate']
    if maxlegs and len(zl) > maxlegs:
        zl = zl.sample(maxlegs, random_state=seed)
    R, offs, FL, TRIP, VID, META = [], [0], [], [], [], []
    for tid, row in zl.sort_values('trip_id').set_index('trip_id').iterrows():
        m, a = rd.load_trip(tid)
        T = np.where(np.isfinite(a['T_amb']), a['T_amb'], np.nanmean(a['T_amb']) if np.isfinite(a['T_amb']).any() else 12.0)
        rho = np.where(np.isfinite(a['rho_air']) & (a['rho_air'] > 0.5), a['rho_air'], 1.2)
        P = np.nan_to_num(np.asarray(a['elec_power'], float))
        raw = bus_raw(np.nan_to_num(a['v']), np.nan_to_num(a['grade']), T, rho, P, a.get('valid'))
        n = raw.shape[1]
        import realbank
        for s0, e in realbank.windows(n):
            R.append(raw[:, s0:e].T.astype(np.float16)); offs.append(offs[-1] + e - s0); FL.append(F.flags_vec(FL_BUS))
            TRIP.append(tid); VID.append(row.vehicle_id)
        pax = float(row.passengers_mean) if np.isfinite(row.passengers_mean) else np.nan
        META.append(dict(trip_id=tid, bus=row.vehicle_id, mission=row.mission, km=float(row.distance_m) / 1000, T=float(row.T_amb_mean),
                         pax=pax, pax_frac=float(row.passengers_frac), season=row.season, route_ok=bool(row.q_route_ok),
                         mass_kg=CURB_KG + PAX_KG * pax if np.isfinite(pax) else np.nan))
    fn = f'{ROOT}/data/bus_{split}.npz'
    np.savez(fn, raw=np.concatenate(R), offs=np.array(offs), flags=np.array(FL), trip=np.array(TRIP), vid=np.array(VID),
             meta=np.array(json.dumps(META)))
    print('bus', split, len(META), 'legs', len(TRIP), 'windows', flush=True)


def load_real(split):
    z = np.load(f'{ROOT}/data/bus_{split}.npz')
    return {k: z[k] for k in z.files if k != 'meta'} | {'meta': json.loads(str(z['meta']))}


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'sim':
        build_sim(a[2], int(a[3]), int(a[4]), int(a[5]) if len(a) > 5 else 6, a[6] if len(a) > 6 else 'b1')
    else:
        build_real(a[2])
