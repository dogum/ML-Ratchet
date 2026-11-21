"""Export ReservoirPilot's weights, the six exam dams and Python reference cases for the browser instrument.

Writes site/public/instruments/reservoirpilot/:
  weights.{json,bin}   the released network (float32: 5,314 weights, 21 KB; float16 would move a dam's J by up to
                       0.4 % through the 10-year closed loop, and the page shows the exam's J beside its own)
  dams.{json,bin}      the six exam dams of examples/exam_dams.json (inflow, demand and inflow calendars, the record),
                       float32, with WaterData's DP J re-run on those float32 inputs
  reference.json       the model's own Python (reservoirpilot.Pilot, sop, tuned_rule, cost_components) and WaterData's
                       dp on the float32 inputs, recorded and with what-ifs on the input (drier / wetter decade, a
                       flood pulse, a smaller reservoir with more demand, a noisy storage gauge)

Run from the repository root:  python models/reservoirpilot/web/export_instrument.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[1]
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'shared' / 'water' / 'data' / 'scripts'))
from mlratchet.webexport import instrument_dir, write_json, write_pack  # noqa: E402
import reservoirpilot as rp  # noqa: E402
from run_baselines_reservoir import dp as waterdata_dp  # noqa: E402  (the exam's DP, unchanged)

K = rp.K
USE = {'fcon': 'flood control', 'irri': 'irrigation', 'elec': 'hydropower', 'supp': 'water supply',
       'navi': 'navigation', 'recr': 'recreation', 'other': 'other'}
CDEC = {'grand_148': 'Oroville'}            # WaterData's CDEC crosswalk (ORO); the other dams go by GRanD id
PULSE_LEN = 31
f32 = lambda a: np.asarray(a, np.float32).astype(np.float64)  # noqa: E731


def pulse_shape(k):
    return (k / 3) * math.exp(1 - k / 3)


def gaussians(seed, n):
    """mulberry32 + Box-Muller, bit for bit as the site kit's rng (net.ts gaussians)."""
    M = 0xFFFFFFFF
    a = seed & M
    out = np.empty(n)

    def u():
        nonlocal a
        a = (a + 0x6D2B79F5) & M
        t = ((a ^ (a >> 15)) * (1 | a)) & M
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M)) & M) ^ t
        return ((t ^ (t >> 14)) & M) / 4294967296

    for i in range(n):
        x = 0.0
        while x == 0.0:
            x = u()
        y = u()
        out[i] = math.sqrt(-2 * math.log(x)) * math.cos(2 * math.pi * y)
    return out


def make_task(d, w):
    """The dam's exam task with what-ifs on the input (net.ts makeTask). Returns (task, seen inflow or None)."""
    n, q = len(d['I']), d['Qmax']
    cap = d['cap'] * w['capacity']
    I = f32(d['I']) * w['inflow']
    if w['pulse'] > 0:
        t0 = w['pulseDay'] - 3
        for k in range(PULSE_LEN):
            t = t0 + k
            if 0 <= t < n:
                I[t] += w['pulse'] * q * pulse_shape(k)
    seen = None
    if w['gauge'] > 0:
        e = gaussians(w['seed'], n + 1); sg = (w['gauge'] / 100) * cap
        seen = I + (sg * e[1:] - sg * e[:-1]) / K
    dates = pd.date_range(d['start'], periods=n, freq='D')
    Dc = f32(d['Dcurve']) if w['demand'] == 1 else f32(d['Dcurve']) * w['demand']
    task = {'I': I, 'dates': dates, 'Dcurve': Dc, 'Icurve': f32(d['Icurve']), 'Qmax': q, 'cap': cap,
            'S0': d['S0'] * w['capacity']}
    task['D'] = Dc[dates.day_of_year - 1]
    return task, seen


def dp_run(task):
    R = waterdata_dp(task['I'], task['D'], task['Qmax'], task['cap'], task['S0'])
    S = np.empty(len(R)); s = task['S0']
    for t in range(len(R)):                     # dp's own forward pass, replayed for the storage path
        avail = s + task['I'][t] * K; s1 = avail - R[t] * K
        s1 = min(s1, task['cap']); s = max(s1, 0.0); S[t] = s
    return R, S


def comps(R, S, task):
    c = rp.cost_components(R, S, task)
    return [c['J'], c['J_short'], c['J_flood'], c['shortage_ratio'], c['shortage_days_frac'], c['flood_days_per_yr'],
            c['max_excess_ratio'], c['storage_mean_frac'], c['storage_end_minus_start_frac'], c['empty_days_frac'],
            c['full_days_frac']]


DAYS = (0, 365, 1000, 2000, -1)
BASE = dict(inflow=1.0, pulse=0.0, pulseDay=0, capacity=1.0, gauge=0.0, seed=1, demand=1.0)
VARIANTS = [
    ('recorded', {}, True),
    ('drier decade x0.7', dict(inflow=0.7), True),
    ('wetter x1.3 + flood pulse 2 Qmax at day 1500', dict(inflow=1.3, pulse=2.0, pulseDay=1500), True),
    ('capacity x0.6, demand x1.2', dict(capacity=0.6, demand=1.2), True),
    ('storage gauge error 0.3 % of capacity', dict(gauge=0.3, seed=7), False),
]

if __name__ == '__main__':
    out = instrument_dir('reservoirpilot')
    pol = rp.Pilot.load()
    w32 = {k: np.asarray(v, np.float32) for k, v in pol.w.items()}
    write_pack(out / 'weights', w32, dtype='f32', meta={'fset': pol.fset, 'head': pol.head, 'version': '1.0',
                                                         'params': int(sum(v.size for k, v in w32.items() if k[0] in 'Wb'))})
    net = rp.Pilot({k: v.astype(np.float64) for k, v in w32.items()}, pol.fset, pol.head)

    dams = json.loads((MODEL / 'examples' / 'exam_dams.json').read_text(encoding='utf-8'))
    exam = json.loads((MODEL / 'training' / 'exam_results.json').read_text(encoding='utf-8'))['resopsus_exam']
    arrays, infos, reference = {}, [], []
    for i, d in enumerate(dams):
        task, _ = make_task(d, BASE)
        Rd, Sd = dp_run(task)
        Jdp = rp.cost_components(Rd, Sd, task)['J']
        for k, key in (('I', 'I'), ('Dcurve', 'Dcurve'), ('Icurve', 'Icurve'), ('Robs', 'R_obs'), ('Sobs', 'S_obs')):
            arrays[f'{k}{i}'] = np.asarray(d[key], np.float32)
        cap_years = d['cap'] / (float(np.mean(d['Icurve'])) * 365.25 * K)
        name = CDEC.get(d['res_id'])
        infos.append(dict(res_id=d['res_id'], label=(name + ' · ' if name else '') + 'GRanD ' + d['res_id'].split('_')[1],
                          main_use=d['main_use'], use_name=USE.get(d['main_use'], d['main_use']), start=d['start'],
                          n=len(d['I']), Qmax=d['Qmax'], cap=d['cap'], S0=d['S0'], cap_years=round(cap_years, 3),
                          exam_J=d['exam_J'], dp_J=Jdp))
        print(f"{d['res_id']}: DP J {Jdp:.6g} (exam {d['exam_J']['dp']:.6g}), cap {cap_years:.2f} years of inflow")
        for label, change, with_dp in VARIANTS:
            w = dict(BASE, **change)
            task, seen = make_task(d, w)
            R, S, P = net.run(task, return_params=True, seen=seen)
            Rs, Ss = rp.sop(task)
            Rt, St = rp.tuned_rule(task, seen=seen)
            o = comps(R, S, task) + [R[t] / task['Qmax'] for t in DAYS] + [P[1000, 1], P[1000, 2]]
            o += [rp.cost_components(Rt, St, task)[k] for k in ('J', 'storage_mean_frac', 'storage_end_minus_start_frac')]
            o += [rp.cost_components(Rs, Ss, task)[k] for k in ('J', 'storage_mean_frac', 'storage_end_minus_start_frac')]
            if with_dp:
                Rd, Sd = dp_run(task); cd = rp.cost_components(Rd, Sd, task)
                o += [cd['J'], cd['storage_mean_frac']]
            reference.append(dict(example=i, variant=label, dam=i, whatif=w, dp=with_dp, outputs=[float(v) for v in o]))
    medians = {p: {k: exam[p][k] for k in ('J', 'S_mean_frac', 'S_end_minus_start_frac')} for p in exam}
    write_pack(out / 'dams', arrays, dtype='f32', meta={'dams': infos, 'exam': medians})
    for stale in ('examples.json', 'examples.bin'):
        (out / stale).unlink(missing_ok=True)
    write_json(out / 'reference.json', dict(cases=reference, tolerance=1e-6))
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f'{len(dams)} dams, {len(reference)} parity cases, {size / 1e6:.2f} MB')
