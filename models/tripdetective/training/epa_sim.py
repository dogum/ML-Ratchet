"""EPA exam families: what is gradable. The EPA test-car list has real vehicle parameters (test weight, coast-down
road-load targets A/B/C, displacement, rated power, architecture) but NO real trip logs. So the only test possible is
semi-real: build each exam vehicle's spec from its real EPA record, drive it in RoadSim over real Ann Arbor routes,
log it with the OBD emulator, and ask whether the Detective recovers the real parameters. This tests generalisation
to real parameter combinations, not to real logs. Fixed random draw (seed 0) within architecture strata, MY >= 2012.
usage: python3 epa_sim.py <ckpt> [n_per_arch=50] [K=4]"""
from common import *
import copy, pandas as pd
import realdata as rd
from roadsim import simulate
from roadsim.vehicle import spec_from_abc, fill_spec
import fleet, routes, obdlog, features as F, priors
import evaluate as E
from model import from_reg
from exam import roadload_F

LBF = 4.4482216; MPH = 0.44704
EPA2ARCH = {'conv': 'conv', 'hev': 'hev_powersplit', 'bev': 'bev'}


def epa_spec(r, rng):
    arch = EPA2ARCH[r.architecture]
    cls = 'car' if r.veh_type == 'Car' else ('pickup' if 'PICKUP' in str(r.model).upper() or 'F150' in str(r.model).upper() else 'suv')
    if cls in fleet.ALLOWED and arch not in fleet.ALLOWED[cls]: cls = 'suv'
    lib = [s for s in fleet._library() if s['architecture'] == arch and s['class'] == cls]
    s = copy.deepcopy(lib[int(rng.integers(len(lib)))])
    s['mass_kg'] = float(r.test_mass_kg); s['payload_kg'] = 0.0
    s = spec_from_abc(s, r.A_lbf, r.B_lbf_mph, r.C_lbf_mph2, mass_kg=s['mass_kg'])
    if s.get('engine') and np.isfinite(r.displacement_L):
        s['engine']['displacement_L'] = float(r.displacement_L)
        if np.isfinite(r.rated_hp): s['engine']['max_power_kW'] = float(r.rated_hp) * 0.7457 * (0.75 if arch != 'conv' else 1.0)
    s = priors.vehicle_prior(s, rng, 'v2')
    s['id'] = f"epa_{r.name}"; fill_spec(s)
    return s, arch


def run(ckpt, nper=50, K=4):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    e = rd.epa_testcars(split='exam')
    e = e[(e.model_year >= 2012) & e.test_mass_kg.notna() & e.A_lbf.notna() & (e.C_lbf_mph2 > 0.008) & (e.A_lbf > 5)]
    rows = []
    for a in ['conv', 'hev', 'bev']:
        x = e[e.architecture == a]
        x = x[x.displacement_L.notna()] if a != 'bev' else x
        rows.append(x.sample(min(nper, len(x)), random_state=0))
    e = pd.concat(rows)
    rng = np.random.default_rng(0); out = []
    for vid, r in e.iterrows():
        try:
            s, arch = epa_spec(r, rng)
        except Exception as ex:
            print('skip', vid, ex); continue
        cfg = obdlog.logger_config(arch, rng, fuel_src='maf'); cfg['rpm'] = arch != 'bev'; cfg['batt'] = arch == 'bev'; cfg['aux'] = False
        inp = []
        for k in range(K):
            route, _ = routes.real_route(rng); simr = routes.perturb_for_sim(route, rng)
            out_ = simulate(s, simr, init_temps=dict(WARM, battery=float(np.mean(simr['T_amb'])) + 10), init_soc=0.6 if arch != 'bev' else 0.8)
            lg = obdlog.emulate(out_, cfg, rng, s)
            a_ = {k2: route[k2] for k2 in ['grade', 'T_amb', 'wind_speed', 'wind_dir', 'heading', 'rho_air']} | lg
            fl = dict(has_fuel=arch != 'bev', fuel_est=arch != 'bev', has_rpm=arch != 'bev', has_batt=arch == 'bev', has_aux=False)
            inp.append(F.net_input(F.raw_from_log(a_, fl), F.flags_vec(fl)))
        P = E.predict(net, inp, np.zeros(K, int)); P1 = E.predict(net, inp[:1], np.zeros(1, int))
        mu = P['mu'][0]
        rec = dict(vid=vid, arch=arch, my=int(r.model_year), model=f'{r.make} {r.model}', etw=float(r.test_mass_kg), disp=float(r.displacement_L) if np.isfinite(r.displacement_L) else np.nan,
                   arch_pred=ARCHS[int(P['arch'][0].argmax())], arch_pred1=ARCHS[int(P1['arch'][0].argmax())],
                   mass_pred=float(from_reg(0, mu[0])), mass_pred1=float(from_reg(0, P1['mu'][0, 0])), disp_pred=float(from_reg(1, mu[1])),
                   CdA_pred=float(from_reg(2, mu[2])), Crr_pred=float(from_reg(3, mu[3])))
        for sp in [20, 40, 60]:
            v = sp * MPH
            rec[f'F{sp}'] = (r.A_lbf + r.B_lbf_mph * sp + r.C_lbf_mph2 * sp * sp) * LBF
            rec[f'F{sp}_pred'] = float(roadload_F(rec['mass_pred'], rec['Crr_pred'], rec['CdA_pred'], sp))
        out.append(rec)
    d = pd.DataFrame(out); d.to_csv(f'{ROOT}/data/epa_sim.csv', index=False)
    rep = {}
    for a in ['conv', 'hev_powersplit', 'bev', 'all']:
        x = d if a == 'all' else d[d.arch == a]
        if not len(x): continue
        rep[a] = dict(n=len(x), arch_acc_pooled=float((x.arch == x.arch_pred).mean()), arch_acc_1trip=float((x.arch == x.arch_pred1).mean()),
                      vedtype_acc=float(np.mean([VEDTYPE[p] == VEDTYPE[t] for p, t in zip(x.arch_pred, x.arch)])),
                      mass_mdape_pooled=float(np.median(np.abs(x.mass_pred / x.etw - 1))), mass_mdape_1trip=float(np.median(np.abs(x.mass_pred1 / x.etw - 1))),
                      mass_bias=float(np.median(x.mass_pred / x.etw - 1)),
                      disp_mdape=float(np.nanmedian(np.abs(x.disp_pred / x.disp - 1))) if a != 'bev' else None,
                      **{f'F{sp}_mdape': float(np.median(np.abs(x[f'F{sp}_pred'] / x[f'F{sp}'] - 1))) for sp in [20, 40, 60]},
                      **{f'F{sp}_bias': float(np.median(x[f'F{sp}_pred'] / x[f'F{sp}'] - 1)) for sp in [20, 40, 60]})
    jsave(rep, f'{ROOT}/epa_sim.json'); print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
