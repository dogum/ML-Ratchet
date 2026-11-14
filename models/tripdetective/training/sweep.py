"""Identifiability sweep (Phase D; frozen, physics v2): does the estimate follow the physics of one car?
N base cars drawn from a prior, stratified by powertrain type (QUOTA: the VED-era mass prior keeps few BEVs, and a first
60-car draw on 14 Nov held one; it was rebuilt stratified before any rung was scored on it); each is driven over the same K real Ann Arbor routes (train_noise) three times: as
drawn, with +800 kg, and with CdA x 1.5. The logger (channels, cadences, MAF bias) and every random draw of the run and
the emulator are the same across the three, so only the physics differs. Stored like a bank (simbank layout); vehicle
id = 3 x car + variant (0 base, 1 +800 kg, 2 CdA x 1.5).
Score of a network (sweep_scores): median over cars of the pooled mass estimate's change (kg) for +800 kg, and the
median CdA ratio for x1.5, pooling K trips and 2 trips. V2_PLAN target: mass responds to +800 kg by >= 500 kg.
usage: python sweep.py <tag> <unused> <seed> [K=8] [prior=v6]   (64 cars: QUOTA)"""
from common import *
import copy, time
from roadsim import simulate
import fleet, routes, obdlog, priors, features as F

DM = 800.0
FC = 1.5
QUOTA = {'ICE': 24, 'HEV': 16, 'PHEV': 12, 'EV': 12}


def build(tag, n, seed, K=8, prior='v6'):
    d = f'{ROOT}/bank/{tag}'; os.makedirs(d, exist_ok=True)
    fn = f'{d}/chunk_000.npz'
    if os.path.exists(fn): return
    jsave(dict(tag=tag, n=n, seed=seed, K=K, prior=prior, physics=PHYSICS, dm=DM, fc=FC), f'{d}/info.json')
    rng = np.random.default_rng(seed)
    R, offs, FL, LED, TV, KM, EN, META = [], [0], [], [], [], [], [], []
    t0 = time.time()
    left = dict(QUOTA)
    for c in range(sum(QUOTA.values())):
        while True:
            spec = fleet.draw_vehicle(rng, cls_w=priors.V3_CLS if priors.mass_prior(prior) else None, prior=prior)
            if left[VEDTYPE[spec['architecture']]] > 0: break
        left[VEDTYPE[spec['architecture']]] -= 1
        spec = priors.vehicle_prior(spec, rng, prior)
        cfg = (obdlog.logger_config_ved if priors.logger_kind(prior) == 'ved' else obdlog.logger_config)(spec['architecture'], rng)
        cfg['p_dropout'] = 0.0
        rts = [routes.real_route(rng)[0] for _ in range(K)]
        seeds = rng.integers(1 << 30, size=K)
        arch = spec['architecture']
        for var in range(3):
            s = copy.deepcopy(spec)
            if var == 1: s['mass_kg'] = float(s['mass_kg']) + DM
            if var == 2: s['CdA_m2'] = round(float(s['CdA_m2']) * FC, 4)
            tr = fleet.truth(s); tr['vid'] = 3 * c + var; tr['base'] = c; tr['variant'] = var; tr['spec'] = s
            for k in range(K):
                r2 = np.random.default_rng(int(seeds[k]))
                simr = routes.perturb_for_sim(rts[k], r2)
                soc0 = 0.6 if arch.startswith('hev') else (0.8 if arch != 'conv' else None)
                out = simulate(s, simr, init_temps=WARM, init_soc=soc0, rng=int(r2.integers(1 << 30)), options=PHYSICS)
                log = obdlog.emulate(out, cfg, r2, s)
                a = {k2: rts[k][k2] for k2 in ['grade', 'T_amb', 'wind_speed', 'wind_dir', 'heading', 'rho_air']} | log
                fl = dict(has_fuel=cfg['fuel'] is not None, fuel_est=cfg['fuel'] in ('maf', 'absload'), has_rpm=cfg['rpm'],
                          has_batt=cfg['batt'], has_aux=cfg['aux'])
                raw = F.raw_from_log(a, fl)
                R.append(raw.T.astype(np.float16)); offs.append(offs[-1] + raw.shape[1]); FL.append(F.flags_vec(fl))
                LED.append(np.zeros(6, np.float32)); TV.append(3 * c + var); KM.append(out['summary']['distance_m'] / 1000)
                EN.append([0, 0, float(np.mean(rts[k]['T_amb'])), float(s['mass_kg'] + s.get('payload_kg', 0))])
            META.append(tr)
        if c % 10 == 9: print('sweep car', c + 1, f'{time.time() - t0:.0f}s', flush=True)
    np.savez(fn, raw=np.concatenate(R).astype(np.float16), offs=np.array(offs, np.int64), flags=np.array(FL), ledger=np.array(LED),
             veh=np.array(TV), km=np.array(KM), energy=np.array(EN, np.float32),
             meta=np.array(json.dumps(META, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))))


def sweep_scores(net, bank, ks=(8, 2)):
    import evaluate as E
    from model import from_reg
    meta = {m['vid']: m for m in bank['meta']}
    vids = bank['veh']; o = bank['offs']
    out = {}
    for k in ks:
        inp, grp, gv = [], [], []
        for v in sorted(meta):
            js = np.where(vids == v)[0][:k]
            for j in js:
                inp.append(F.net_input(bank['raw'][o[j]:o[j + 1]].T.astype(np.float32), bank['flags'][j])); grp.append(len(gv))
            gv.append(v)
        P = E.predict(net, inp, np.array(grp))
        m = {v: float(from_reg(0, P['mu'][i, 0])) for i, v in enumerate(gv)}
        cda = {v: float(from_reg(2, P['mu'][i, 2])) for i, v in enumerate(gv)}
        bases = sorted({meta[v]['base'] for v in gv})
        dm = np.array([m[3 * b + 1] - m[3 * b] for b in bases])
        rc = np.array([cda[3 * b + 2] / cda[3 * b] for b in bases])
        arch = [meta[3 * b]['arch'] for b in bases]
        out[f'k{k}'] = dict(n=len(bases), mass_response_kg=float(np.median(dm)), mass_response_iqr=np.percentile(dm, [25, 75]).round(0).tolist(),
                            mass_response_by_kind={kd: float(np.median(dm[[VEDTYPE[a] == kd for a in arch]])) for kd in VEDTYPES
                                                   if sum(VEDTYPE[a] == kd for a in arch) >= 2},
                            cda_ratio=float(np.median(rc)), true_dm=DM, true_cda_ratio=FC)
    return out


if __name__ == '__main__':
    a = sys.argv
    build(a[1], int(a[2]), int(a[3]), int(a[4]) if len(a) > 4 else 8, a[5] if len(a) > 5 else 'v6')
