"""Precompute the page's lab: three base cars (conventional, power-split hybrid, BEV) with mass, CdA and strategy changed
on a grid, each driven by RoadSim over the same two real Ann Arbor routes (train_noise), logged by the OBD emulator.
The browser runs the network on these logs live. usage: python3 build_lab.py <ckpt>"""
from common import *
import copy, base64
from roadsim import simulate
from roadsim.generator import generate_vehicle
import routes, obdlog, features as F, priors
import evaluate as E
from model import from_reg

MASS = [-200, 0, 200, 400, 600]
CDA = [0.8, 1.0, 1.2, 1.4]
STRAT = {'conv': ['conv_rule', 'conv_stopstart'], 'hev_powersplit': ['aecms', 'powersplit_rule', 'ecms'], 'bev': ['bev_regen_series', 'bev_one_pedal']}


def base_specs():
    out = {}
    rng = np.random.default_rng(5)
    want = {'conv': (1550, 2.4), 'hev_powersplit': (1500, 1.8), 'bev': (1750, None)}
    for arch, (m, d) in want.items():
        best = None
        for _ in range(300):
            s = generate_vehicle(rng, 'car', arch)
            e = abs(s['mass_kg'] + s['payload_kg'] - m) / m + (abs(s['engine']['displacement_L'] - d) / d if d else 0)
            if best is None or e < best[0]: best = (e, s)
        s = best[1]; s['payload_kg'] = 80.0; s['mass_kg'] = float(m - 80)
        if s.get('engine'): s['engine']['eff_scale'] = 0.88
        out[arch] = s
    return out


def pick_routes(n=2):
    rng = np.random.default_rng(11); tn = routes.pool()['tn']; out = []
    while len(out) < n:
        r = tn[int(rng.integers(len(tn)))]
        if 450 <= len(r['v']) <= 900 and r['v'].mean() > 6:
            hw = r['wind_speed'] * np.cos(np.deg2rad(r['wind_dir'] - r['heading']))
            q = dict(id=r['id'], v=np.round(r['v'].astype(float), 2), grade=np.round(r['grade'].astype(float), 5), elev=r['elev'],
                     T_amb=np.round(r['T_amb'].astype(float), 1), wind_speed=np.round(hw.astype(float), 2), wind_dir=np.zeros(len(hw)),
                     heading=np.zeros(len(hw)), rho_air=np.round(r['rho_air'].astype(float), 3), surface=r['surface'])
            out.append(q)
    return out


def build(ckpt):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    bases = base_specs(); R = pick_routes()
    lab = dict(routes=[], configs=[], mass=MASS, cda=CDA, strat=STRAT)
    for r in R:
        lab['routes'].append(dict(id=r['id'], n=len(r['v']), v=r['v'].tolist(), grade=r['grade'].tolist(),
                                  T=r['T_amb'].tolist(), hw=r['wind_speed'].tolist(), rho=r['rho_air'].tolist()))
    for arch, base in bases.items():
        for st in STRAT[arch]:
            for dm in MASS:
                for fc in CDA:
                    s = copy.deepcopy(base); s['mass_kg'] = base['mass_kg'] + dm; s['CdA_m2'] = round(base['CdA_m2'] * fc, 4)
                    s['control'] = {'strategy': st, 'params': {}}
                    rng = np.random.default_rng(abs(hash((arch, st, dm, fc))) % 2**31)
                    cfg = obdlog.logger_config(arch, rng, fuel_src='maf'); cfg['rpm'] = arch != 'bev'; cfg['batt'] = arch == 'bev'; cfg['aux'] = False
                    cfg['p_dropout'] = 0
                    logs, inp, led = [], [], []
                    for r in R:
                        trip = {k: r[k] for k in routes.KEYS}
                        out = simulate(s, trip, init_temps=dict(WARM, battery=float(np.mean(r['T_amb'])) + 10), init_soc=0.6 if arch != 'bev' else 0.8, options='roadsim-physics-v1')
                        lg = obdlog.emulate(out, cfg, rng, s)
                        a = {k: r[k] for k in ['grade', 'T_amb', 'wind_speed', 'wind_dir', 'heading', 'rho_air']} | lg
                        a['v'] = r['v']; a['valid'] = np.ones(len(r['v']))           # shared logged speed, no dropouts
                        fl = dict(has_fuel=arch != 'bev', fuel_est=arch != 'bev', has_rpm=arch != 'bev', has_batt=arch == 'bev', has_aux=False)
                        raw = F.raw_from_log(a, fl)
                        raw[5] = np.round(raw[5] * 1000) / 1000; raw[6] = np.round(raw[6] * 1000) / 1000
                        raw[7] = np.round(raw[7] * 10) / 10; raw[8] = np.round(raw[8] * 1000) / 1000
                        inp.append(F.net_input(raw, F.flags_vec(fl)))
                        km = out['summary']['distance_m'] / 1000; L_ = out['ledger']
                        led.append([round(float(x) / km, 1) for x in [L_['aero'], L_['rolling'], L_['grade'], L_['brake'],
                                                                     L_['eng_loss'] + L_['mot_loss'] + L_['trans_loss'] + L_['batt_loss'], L_['aux']]])
                        logs.append(dict(fuel=np.round(raw[5] * 1000).astype(int).tolist(), rpm=np.round(raw[6] * 1000).astype(int).tolist()) if arch != 'bev'
                                    else dict(batt=np.round(raw[7] * 10).astype(int).tolist(), soc=np.round(raw[8] * 1000).astype(int).tolist()))
                    P = E.predict(net, inp, np.zeros(len(inp), int))
                    lab['configs'].append(dict(arch=arch, strat=st, dm=dm, fc=fc, mass=s['mass_kg'] + s['payload_kg'], CdA=s['CdA_m2'], Crr=s['Crr'],
                                               disp=(s.get('engine') or {}).get('displacement_L'), logs=logs, ledger=led,
                                               py=dict(arch=P['arch'][0].round(4).tolist(), mass=float(from_reg(0, P['mu'][0, 0])),
                                                       CdA=float(from_reg(2, P['mu'][0, 2])), disp=float(from_reg(1, P['mu'][0, 1])))))
                    print(arch, st, dm, fc, lab['configs'][-1]['py'], flush=True)
    json.dump(lab, open(f'{ROOT}/web/lab.json', 'w'), separators=(',', ':'))
    print('lab.json', os.path.getsize(f'{ROOT}/web/lab.json') // 1024, 'kB')


if __name__ == '__main__':
    build(sys.argv[1])
