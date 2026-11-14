"""Diagnosis of the generated hybrids' fuel economy (RoadSim v3.1 pass, item D). Simulation only: no EPA result is read.

Generated car and SUV full hybrids (generate_vehicles with the preset's generator options) run the check_generator
procedure (hot UDDS and HWFET at the donor's test weight, payload 0, SOC-corrected). Each counterfactual changes one
generated feature to the EPA twins' rule (roadsim/hybrid_hw.py) and re-runs the same bodies:
  crr_v        the donor's own speed term of the rolling resistance (the generator clips it to [-0.01, 0.02])
  eff          engine eff_scale 1 (the generator draws it lognormal, sd 3 %)
  soc          the twins' SOC window (power-split 0.4-0.8, others 0.3-0.8)
  strat        the twins' strategy (A-ECMS) instead of the drawn one
  all          every change above
The median ratio counterfactual / base per architecture says how much each feature costs.

--transplant: the other direction. The EPA twins of the dev-family hybrid records (check_hybrids_epa.py --set dev specs,
no EPA result read) get one generator rule at a time, drawn with a fixed seed: 'gear' (power-split final drive
U(2.9, 3.7) and MG2 ratio U(2.6, 3.4); P2 6-speed DCT / 8-speed AT), 'machines' (power-split MG1 = engine x U(0.45,
0.65), MG2 = system x U(0.55, 0.8) with system = engine / U(0.65, 0.85); series traction = system x U(0.85, 1.0),
generator = engine x 1.05; P2 motor U(30, 60) kW), 'battery' (0.8-1.9 kWh, 220-280 V, window 0.35-0.8 / 0.3-0.8),
'engine' (displacement min(disp, P / 50 kW/L), eff_scale lognormal sd 3 %, no model year), 'all'.

usage: python gen_hybrid_diag.py [--options roadsim-physics-v3] [--n 400]
"""
import os, sys, json, copy, argparse
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(VEH, 'roadsim')); sys.path.insert(0, VEH)
import _opts
import check_generator as CG
from epa_procedures import test_spec
from roadsim.generator import generate_vehicles, _cal

VARS = ('base', 'crr_v', 'eff', 'soc', 'strat', 'all')


def variant(s, k):
    s = copy.deepcopy(s)
    a = s['architecture']
    if k in ('crr_v', 'all'):
        g = s['generator']; T = _cal()['testcar'][g['donor']]
        s['Crr_speed_coef'] = float(T['Crr_v'][g['donor_row']])
    if k in ('eff', 'all'):
        s['engine']['eff_scale'] = 1.0
    if k in ('soc', 'all'):
        s['battery']['soc_window'] = [0.4, 0.8] if a == 'hev_powersplit' else [0.3, 0.8]
    if k in ('strat', 'all'):
        s['control'] = {'strategy': 'aecms', 'params': {}}
    return s


def one(job):
    s, opt = job
    _opts.set(opt)
    out = {}
    for k in VARS:
        v = test_spec(variant(s, k)); v['payload_kg'] = 0.0
        try:
            out[k] = (CG.fe(v, 'udds'), CG.fe(v, 'hwfet'))
        except Exception as ex:
            out[k] = (None, None)
    return dict(arch=s['architecture'], cls=s['class'], strategy=(s.get('control') or {}).get('strategy'),
                crr_v=s['Crr_speed_coef'], crr_v_donor=float(_cal()['testcar'][s['generator']['donor']]['Crr_v'][s['generator']['donor_row']]),
                fe=out)


TVARS = ('base', 'gear', 'machines', 'battery', 'engine', 'all')


def transplant(s, k, seed):
    rng = np.random.default_rng(seed)
    s = copy.deepcopy(s); a = s['architecture']
    Pe = s['engine']['max_power_kW']
    d = dict(fd=float(rng.uniform(2.9, 3.7)), r2=float(rng.uniform(2.6, 3.4)), f1=float(rng.uniform(0.45, 0.65)),
             f2=float(rng.uniform(0.55, 0.8)), fe=float(rng.uniform(0.65, 0.85)), ft=float(rng.uniform(0.85, 1.0)),
             p2=float(rng.uniform(30, 60)), cap=float(rng.uniform(0.8, 1.9)), V=float(rng.choice([220.0, 245.0, 260.0, 280.0])),
             eff=float(np.exp(rng.normal(0, 0.03))), n=int(rng.choice([6, 8])))
    if k in ('gear', 'all'):
        if a == 'hev_powersplit':
            s['final_drive'] = round(d['fd'], 3); s['transmission']['mg2_ratio'] = round(d['r2'], 3)
        elif a == 'hev_parallel_p2':
            from roadsim.vehicle import GEAR_SETS
            tr = 'DCT' if d['n'] == 6 else 'AT'
            g = GEAR_SETS.get((tr, d['n'])) or GEAR_SETS[('AT', 8)]
            top0 = s['transmission']['gears'][-1]
            s['final_drive'] = s['final_drive'] * top0 / g[-1]
            s['transmission'] = dict(type=tr, gears=list(g), torque_converter=False)
    if k in ('machines', 'all'):
        P = Pe / d['fe']
        for m in s['motors']:
            if m['role'] == 'MG1': m['max_power_kW'] = round(Pe * d['f1'], 1); m.pop('max_torque_Nm', None)
            if m['role'] == 'MG2': m['max_power_kW'] = round(P * d['f2'], 1); m.pop('max_torque_Nm', None); m['max_rpm'] = 17000
            if m['role'] == 'traction': m['max_power_kW'] = round(P * d['ft'], 1); m.pop('max_torque_Nm', None)
            if m['role'] == 'generator': m['max_power_kW'] = round(Pe * 1.05, 1)
            if m['role'] == 'P2': m['max_power_kW'] = round(d['p2'], 1); m.pop('max_torque_Nm', None)
    if k in ('battery', 'all'):
        s['battery'].update(capacity_kWh=round(d['cap'], 2), nominal_V=d['V'],
                            soc_window=[0.35, 0.8] if a == 'hev_powersplit' else [0.3, 0.8])
    if k in ('engine', 'all'):
        s['engine']['displacement_L'] = round(min(s['engine']['displacement_L'], Pe / 50), 1)
        s['engine']['eff_scale'] = round(d['eff'], 3)
        s.pop('model_year', None)
    return s


def one_t(job):
    s, opt, seed = job
    _opts.set(opt)
    out = {}
    for k in TVARS:
        v = test_spec(transplant(s, k, seed)); v['payload_kg'] = 0.0
        try:
            out[k] = (CG.fe(v, 'udds'), CG.fe(v, 'hwfet'))
        except Exception:
            out[k] = (None, None)
    return dict(arch=s['architecture'], name=s.get('name'), fe=out)


def main_transplant(a):
    import check_hybrids_epa as CH
    recs, _ = CH.dev_records()
    specs = [r['spec'] for r in recs if r['spec']['architecture'] in ('hev_powersplit', 'hev_parallel_p2', 'hev_series')
             and r['group'] not in ('mild',)]
    jobs = [(s, _opts.get(), 1000 + i) for i, s in enumerate(specs)]
    with Pool(a.workers) as p:
        rows = p.map(one_t, jobs, chunksize=1)
    res = {}
    for arch in ('hev_powersplit', 'hev_parallel_p2', 'hev_series'):
        R = [r for r in rows if r['arch'] == arch and r['fe']['base'][0]]
        if not R:
            continue
        res[arch] = dict(n=len(R))
        for k in TVARS[1:]:
            ru = [r['fe'][k][0] / r['fe']['base'][0] for r in R if r['fe'][k][0]]
            rh = [r['fe'][k][1] / r['fe']['base'][1] for r in R if r['fe'][k][1]]
            res[arch][k] = (round(100 * (float(np.median(ru)) - 1), 1), round(100 * (float(np.median(rh)) - 1), 1))
        print('transplant', arch, res[arch])
    json.dump(dict(about=__doc__.strip(), options=_opts.get(), summary=res, rows=rows),
              open(os.path.join(VEH, 'roadsim', 'logs', 'gen_hybrid_transplant.json'), 'w', encoding='utf-8'), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--workers', type=int, default=14)
    _opts.init()
    if _opts.get() is None:
        _opts.set('roadsim-physics-v3')
    ap.add_argument('--transplant', action='store_true')
    a = ap.parse_args()
    if a.transplant:
        return main_transplant(a)
    vs = [v for v in generate_vehicles(a.n, seed=20261114, options=_opts.get())
          if v['architecture'] in ('hev_powersplit', 'hev_parallel_p2', 'hev_series') and v['class'] in ('car', 'suv')]
    with Pool(a.workers) as p:
        rows = p.map(one, [(v, _opts.get()) for v in vs], chunksize=1)
    res = {}
    for arch in ('hev_powersplit', 'hev_parallel_p2', 'hev_series'):
        R = [r for r in rows if r['arch'] == arch and r['fe']['base'][0]]
        if not R:
            continue
        res[arch] = dict(n=len(R), base_udds=round(float(np.median([r['fe']['base'][0] for r in R])), 1),
                         base_hwy=round(float(np.median([r['fe']['base'][1] for r in R])), 1))
        for k in VARS[1:]:
            ru = [r['fe'][k][0] / r['fe']['base'][0] for r in R if r['fe'][k][0]]
            rh = [r['fe'][k][1] / r['fe']['base'][1] for r in R if r['fe'][k][1]]
            res[arch][k] = (round(100 * (float(np.median(ru)) - 1), 1), round(100 * (float(np.median(rh)) - 1), 1))
        print(arch, res[arch])
    clip = [r for r in rows if abs(r['crr_v'] - r['crr_v_donor']) > 1e-9]
    print(f'clipped Crr speed term: {len(clip)} of {len(rows)} hybrids; donor values', sorted(round(r['crr_v_donor'], 4) for r in clip)[:20])
    json.dump(dict(about=__doc__.strip(), options=_opts.get(), summary=res, rows=rows),
              open(os.path.join(VEH, 'roadsim', 'logs', 'gen_hybrid_diag.json'), 'w', encoding='utf-8', newline='\n'), indent=1, default=float)


if __name__ == '__main__':
    main()
