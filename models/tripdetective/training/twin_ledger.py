"""Twin test for the network, and the only place the energy ledger can be graded: each car's real trips are re-driven by
its RoadSim twin (published mass and displacement, nearest generated body, v2 priors), logged with the car's own
channel set. We compare (a) the network on the real log vs on the twin log (does it read them alike?) and (b) the
network's ledger on the twin log vs the twin's true ledger (simulated truth only: real cars have no ledger truth).
usage: python3 twin_ledger.py <ckpt> <split> [trips_per_car=12] [max_cars]"""
from common import *
import copy, pandas as pd
import realdata as rd
from roadsim import simulate
import fleet, obdlog, priors, realbank, features as F
import evaluate as E
from model import from_reg
from twin import FSRC


def run(ckpt, split='dev', ntr=12, maxcars=None):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    vt = rd.ved_vehicles(); cars = vt[vt.split == split]
    if maxcars:
        cars = cars.groupby('ved_type', group_keys=False).apply(lambda g: g.sample(min(len(g), max(2, int(maxcars * len(g) / len(cars)))), random_state=0))
    rows, led = [], []
    for vid, row in cars.iterrows():
        try: spec0 = fleet.twin_spec(vid, row)
        except Exception as ex: print(vid, ex); continue
        rin, tin, ledt = [], [], []
        for m, a in rd.load_trips(split, vehicles=[vid], label_ok=True, limit=ntr, seed=1):
            if len(a['v']) > L: continue
            rng = np.random.default_rng(abs(hash(m['trip_id'])) % 2**32)
            sp = priors.vehicle_prior(copy.deepcopy(spec0), rng, 'v2')
            soc0 = float(np.nanmedian(a['soc'][:30])) if np.isfinite(a['soc'][:30]).any() else None
            out = simulate(sp, a, init_temps=dict(WARM, battery=float(np.nanmean(a['T_amb'])) + 10), init_soc=soc0, options='roadsim-physics-v1')
            cfg = obdlog.logger_config(row.architecture, rng); cfg['fuel'] = FSRC.get(m['fuel_source'])
            cfg['rpm'] = bool(np.isfinite(a['engine_rpm']).mean() > 0.5); cfg['batt'] = bool(np.isfinite(a['batt_current']).mean() > 0.5)
            cfg['aux'] = bool(np.isfinite(a['aux_power']).mean() > 0.5); cfg['p_dropout'] = 0
            lg = obdlog.emulate(out, cfg, rng, sp); lg['valid'] = a['valid']
            fl = realbank.flags_for(m, a)
            sa = {k: a[k] for k in ['grade', 'T_amb', 'wind_speed', 'wind_dir', 'heading', 'rho_air']} | lg
            rin.append(F.net_input(F.raw_from_log(a, fl), F.flags_vec(fl))); tin.append(F.net_input(F.raw_from_log(sa, fl), F.flags_vec(fl)))
            km = out['summary']['distance_m'] / 1000; Lg = out['ledger']
            ledt.append(np.array([Lg['aero'], Lg['rolling'], Lg['grade'], Lg['brake'], Lg['eng_loss'] + Lg['mot_loss'] + Lg['trans_loss'] + Lg['batt_loss'], Lg['aux']]) / km)
        if len(rin) < 2: continue
        Pr = E.predict(net, rin, np.zeros(len(rin), int)); Ps = E.predict(net, tin, np.zeros(len(tin), int))
        truth_m = spec0['mass_kg']
        rows.append(dict(vid=vid, ved_type=row.ved_type, n=len(rin), mass_true=truth_m, disp_true=row.displacement_L,
                         mass_real=float(from_reg(0, Pr['mu'][0, 0])), mass_twin=float(from_reg(0, Ps['mu'][0, 0])),
                         disp_real=float(from_reg(1, Pr['mu'][0, 1])), disp_twin=float(from_reg(1, Ps['mu'][0, 1])),
                         type_real=VEDTYPE[ARCHS[int(Pr['arch'][0].argmax())]], type_twin=VEDTYPE[ARCHS[int(Ps['arch'][0].argmax())]]))
        for j in range(len(tin)):
            led.append(dict(vid=vid, ved_type=row.ved_type, **{f't_{k}': float(ledt[j][i]) for i, k in enumerate(LEDGER)},
                            **{f'p_{k}': float(Ps['led_mu'][j, i]) for i, k in enumerate(LEDGER)}, **{f's_{k}': float(Ps['led_sig'][j, i]) for i, k in enumerate(LEDGER)}))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows); l = pd.DataFrame(led)
    d.to_csv(f'{ROOT}/data/twin_net_{split}.csv', index=False); l.to_csv(f'{ROOT}/data/twin_ledger_{split}.csv', index=False)
    rep = dict(n_cars=len(d), n_trips=len(l),
               mass_real_vs_twin_mdape=float(np.median(np.abs(d.mass_real / d.mass_twin - 1))),
               mass_twin_err_mdape=float(np.median(np.abs(d.mass_twin / d.mass_true - 1))), mass_real_err_mdape=float(np.median(np.abs(d.mass_real / d.mass_true - 1))),
               mass_real_minus_twin_median=float(np.median(d.mass_real - d.mass_twin)),
               disp_real_vs_twin_mdape=float(np.nanmedian(np.abs(d.disp_real / d.disp_twin - 1))),
               type_agree=float((d.type_real == d.type_twin).mean()),
               ledger={k: dict(mae=float(np.mean(np.abs(l[f'p_{k}'] - l[f't_{k}']))), mean_true=float(np.mean(l[f't_{k}'])),
                               mae_median_guess=float(np.mean(np.abs(np.median(l[f't_{k}']) - l[f't_{k}']))),
                               in1s=float(np.mean(np.abs(l[f'p_{k}'] - l[f't_{k}']) < l[f's_{k}']))) for k in LEDGER})
    jsave(rep, f'{ROOT}/twin_ledger_{split}.json'); print(json.dumps(rep, indent=1))
    return rep


if __name__ == '__main__':
    a = sys.argv
    run(a[1], a[2] if len(a) > 2 else 'dev', int(a[3]) if len(a) > 3 else 12, int(a[4]) if len(a) > 4 else None)
