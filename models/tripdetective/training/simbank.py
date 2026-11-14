"""Simulation bank: random light-duty vehicles x K trips each (real Ann Arbor routes, cycle windows, synthetic routes),
simulated in RoadSim, passed through the OBD log emulator, stored as ragged float16 raw channels + truths.
Resumable: one npz per chunk of vehicles.
usage: python3 simbank.py <tag> <n_vehicles> <seed> [K=8] [prior=v1]"""
from common import *
import time, copy, traceback
from roadsim import simulate
import fleet, routes, obdlog, features as F
import priors

CHUNK = 100


def behaviour(on, P, v):
    """engine on/off behaviour truth from the true (simulated) signals of all a vehicle's trips:
    ev_frac = share of moving seconds with the engine off; P_on = wheel-power threshold [kW] that best separates
    engine-on from engine-off seconds (moving, positive power); v_ev = 95th pct of speed with the engine off and positive power."""
    mv = v > 2
    if mv.sum() < 50: return np.nan, np.nan, np.nan
    ev = float(1 - on[mv].mean())
    m = mv & (P > 0)
    if m.sum() < 50 or on[m].all() or (~on[m]).all():
        return ev, np.nan, np.nan
    p = np.sort(P[m]); o = on[m][np.argsort(P[m])]
    # threshold minimising (on below) + (off above)
    err = np.cumsum(o) + ((~o).sum() - np.cumsum(~o))
    k = int(np.argmin(err))
    Pon = float(p[k] / 1000)
    voff = v[m & ~on]
    vev = float(np.percentile(voff, 95)) if len(voff) > 20 else np.nan
    return ev, Pon, vev


def sim_vehicle(spec, K, rng, prior):
    tr = fleet.truth(spec)
    cfg = obdlog.logger_config(spec['architecture'], rng)
    priors.logger_tweak(cfg, spec, rng, prior)
    trips = []; ons, Ps, vs = [], [], []
    for k in range(K):
        route, src = routes.draw_route(rng)
        simr = routes.perturb_for_sim(route, rng)
        s2 = copy.deepcopy(spec)
        extra = float(rng.choice([0, 0, 0, 75, 150]) + rng.normal(0, 10))
        s2['payload_kg'] = max(0.0, float(spec.get('payload_kg') or 0) + extra)
        arch = spec['architecture']
        lo, hi = (spec.get('battery') or {}).get('soc_window', [0.3, 0.8])
        if arch.startswith('hev'): soc0 = rng.uniform(lo + 0.05, hi - 0.05)
        elif arch.startswith('phev'): soc0 = rng.uniform(0.3, 0.95) if rng.random() < 0.5 else rng.uniform(lo, lo + 0.06)
        elif arch == 'bev': soc0 = rng.uniform(0.2, 0.95)
        else: soc0 = None
        temps = WARM if rng.random() < 0.85 else None
        if temps is not None and prior != 'v1':
            temps = dict(WARM, battery=float(np.mean(simr['T_amb']) + rng.uniform(5, 15)))   # pack warm from driving
        out = simulate(s2, simr, init_temps=temps, init_soc=soc0, rng=int(rng.integers(1 << 30)), options='roadsim-physics-v1')
        lg = out['ledger']
        if not lg.get('ok', 1): continue
        log = obdlog.emulate(out, cfg, rng, spec)
        a = {k2: route[k2] for k2 in ['grade', 'T_amb', 'wind_speed', 'wind_dir', 'heading', 'rho_air']} | log
        fl = dict(has_fuel=cfg['fuel'] is not None, fuel_est=cfg['fuel'] in ('maf', 'absload'), has_rpm=cfg['rpm'],
                  has_batt=cfg['batt'], has_aux=cfg['aux'])
        raw = F.raw_from_log(a, fl)
        km = out['summary']['distance_m'] / 1000
        led = np.array([lg['aero'], lg['rolling'], lg['grade'], lg['brake'],
                        lg['eng_loss'] + lg['mot_loss'] + lg['trans_loss'] + lg['batt_loss'], lg['aux']]) / max(km, 1e-3)
        trips.append(dict(raw=raw.astype(np.float16), flags=F.flags_vec(fl), ledger=led.astype(np.float32), km=km,
                          energy_kJ=float(lg['fuel'] + lg['batt_chem_out']), fuel_kJ=float(lg['fuel']), src=src,
                          T=float(np.mean(route['T_amb'])), mass_trip=float(s2['mass_kg'] + s2['payload_kg'])))
        if spec['architecture'] not in ('conv', 'bev'):
            ons.append(np.asarray(out['eng_on']) > 0.5); Ps.append(np.asarray(out['P_wheel'])); vs.append(np.asarray(out['v']))
    if ons:
        tr['ev_frac'], tr['P_on_kW'], tr['v_ev'] = behaviour(np.concatenate(ons), np.concatenate(Ps), np.concatenate(vs))
    else:
        tr['ev_frac'] = tr['P_on_kW'] = tr['v_ev'] = np.nan
    tr['logger'] = {k: cfg[k] for k in ['fuel', 'rpm', 'batt', 'aux']}
    return tr, trips


def build(tag, nveh, seed, K=8, prior='v1'):
    d = f'{ROOT}/bank/{tag}'; os.makedirs(d, exist_ok=True)
    jsave(dict(tag=tag, nveh=nveh, seed=seed, K=K, prior=prior, emu=obdlog.EMU_VERSION, prior_version=priors.VERSION.get(prior)), f'{d}/info.json')
    for c0 in range(0, nveh, CHUNK):
        fn = f'{d}/chunk_{c0 // CHUNK:03d}.npz'
        if os.path.exists(fn): continue
        t0 = time.time()
        rng = np.random.default_rng([seed, c0])
        R, offs, FL, LED, TV, KM, EN, META = [], [0], [], [], [], [], [], []
        for i in range(c0, min(nveh, c0 + CHUNK)):
            try:
                spec = fleet.draw_vehicle(rng, cls_w=priors.V3_CLS if prior == 'v3' else None, prior=prior)
                spec = priors.vehicle_prior(spec, rng, prior)
                tr, trips = sim_vehicle(spec, K, rng, prior)
            except Exception:
                traceback.print_exc(); continue
            if len(trips) < 2: continue
            tr['vid'] = i; tr['spec'] = spec
            for t in trips:
                R.append(t['raw'].T); offs.append(offs[-1] + t['raw'].shape[1]); FL.append(t['flags']); LED.append(t['ledger'])
                TV.append(i); KM.append(t['km']); EN.append([t['energy_kJ'], t['fuel_kJ'], t['T'], t['mass_trip']])
            META.append(tr)
        np.savez(fn, raw=np.concatenate(R).astype(np.float16), offs=np.array(offs, np.int64), flags=np.array(FL), ledger=np.array(LED),
                 veh=np.array(TV), km=np.array(KM), energy=np.array(EN, np.float32),
                 meta=np.array(json.dumps(META, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))))
        print(f'chunk {c0 // CHUNK} {len(META)} veh {len(TV)} trips {time.time() - t0:.0f}s', flush=True)


def load_bank(tag, maxchunks=None):
    """concatenate chunks -> dict(raw (sum_n, C) f16, offs, flags, ledger, veh, km, energy, meta list)"""
    import glob
    fs = sorted(glob.glob(f'{ROOT}/bank/{tag}/chunk_*.npz'))[:maxchunks]
    R, O, FL, LED, V, KM, EN, META = [], [], [], [], [], [], [], []; base = 0
    for f in fs:
        z = np.load(f)
        R.append(z['raw']); O.append(z['offs'][:-1] + base); base += len(z['raw'])
        FL.append(z['flags']); LED.append(z['ledger']); V.append(z['veh']); KM.append(z['km']); EN.append(z['energy'])
        META += json.loads(str(z['meta']))
    O.append(np.array([base]))
    return dict(raw=np.concatenate(R), offs=np.concatenate(O), flags=np.concatenate(FL), ledger=np.concatenate(LED),
                veh=np.concatenate(V), km=np.concatenate(KM), energy=np.concatenate(EN), meta=META)


if __name__ == '__main__':
    a = sys.argv
    build(a[1], int(a[2]), int(a[3]), int(a[4]) if len(a) > 4 else 8, a[5] if len(a) > 5 else 'v1')
