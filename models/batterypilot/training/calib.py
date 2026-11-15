"""Calibrate the price simulator's scarcity and noise parameters on the DEV years only (ERCOT 2019, 2021).

Stage A (2019, and 2021 outside the Uri window): ORDC mu/sigma, real-time noise scale and ramp coupling, DA
uncertainty scale -- matched to real HB_HUBAVG statistics (daily max/min quantiles, hours above 100/500/1000 $/MWh,
median |RT-DA|, DA/RT mean, 2 h perfect-foresight value).  Averaged over 2 outage seeds.
Stage B (Feb 2021, Winter Storm Uri): the cold-weather outage curve f_cold(T) so that the simulated reserve falls
below the ORDC minimum on the same days as the real price sat at the cap.  EIA-930 demand during Uri is served load
(ERCOT shed up to ~20 GW), so for this one calibration the demand is max(served, ERCOT's day-ahead forecast).
Writes data/calibration.json (GridSim physics 1.0) or data/calibration_v1_1.json (physics 1.1, $GRIDSIM_PHYSICS).
"""
import itertools, json, os, sys, time
import numpy as np, pandas as pd
import _paths  # noqa: E402  (this folder, GridSim and GridData on sys.path; outputs to WORK)
from pricesim import Engine, load_inputs, fundamentals, prices, DEFAULTS
from realdata import node_frame
from battery import Battery
from strategies import pf_daily

from simbank import calib_file
OUT = calib_file()   # per GridSim physics version: WORK/data/calibration.json (1.0), calibration_v1_1.json (1.1)
URI = ('2021-02-08', '2021-02-24')


def target_stats(rt, da, day):
    rt = np.asarray(rt, float); da = np.asarray(da, float)
    g = pd.Series(rt).groupby(day)
    mx, mn = g.max(), g.min()
    b = Battery(1, 2, degradation=False)
    c, d = pf_daily(rt, day, b)
    return dict(max_q25=mx.quantile(.25), max_q50=mx.median(), max_q75=mx.quantile(.75), max_q90=mx.quantile(.9),
                min_q50=mn.median(), min_q10=mn.quantile(.1), h100=(rt > 100).sum(), h300=(rt > 300).sum(),
                h1000=(rt > 1000).sum(), rtda=np.nanmedian(np.abs(rt - da)), damean=np.nanmean(da),
                rtmean=np.mean(rt), pf=(rt * (d - c)).sum() / 1000)


WEIGHTS = dict(max_q25=1, max_q50=2, max_q75=2, max_q90=1, min_q50=1, min_q10=0.5, h100=1, h300=1, h1000=1,
               rtda=1, damean=0.5, rtmean=1, pf=2)


def loss(s, r):
    l = 0.0
    for k, w in WEIGHTS.items():
        c = 5.0 if k.startswith('h') else 2.0
        l += w * np.log((abs(s[k]) + c) / (abs(r[k]) + c)) ** 2
    return l


def mask_uri(index):
    return ~((index >= pd.Timestamp(URI[0], tz='UTC')) & (index < pd.Timestamp(URI[1], tz='UTC')))


if __name__ == '__main__':
    t0 = time.time()
    res = {'stageA': [], 'stageB': []}
    base = dict(floor_share=0.8)
    real, Fs = {}, {}
    for year in (2019, 2021):
        f = node_frame('ERCOT', 'HB_HUBAVG', f'{year}-01-01', f'{year + 1}-01-01')
        m = mask_uri(f.index)
        f = f[m]
        real[year] = target_stats(f.rt.interpolate().values, f.da.interpolate().values, f.day.values)
        eng = Engine('ERCO', year); inp = load_inputs('ERCO', year)
        Fs[year] = []
        for seed in (1, 2):
            F = fundamentals(eng, inp, {}, base, seed=seed)
            keep = mask_uri(F['times'])
            F = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in F.items()}
            F['times'] = F['times'][keep]
            Fs[year].append(F)
        print(year, 'real', {k: round(float(v), 1) for k, v in real[year].items()}, round(time.time() - t0), flush=True)
    # a first coarse grid (mu -2000..0, sigma 1500..3500) ended on its edges (mu -2000, sigma 3500); this grid
    # extends it
    if True:
        grid = dict(mu=[-4000, -3000, -2000], sigma=[3500, 5000, 6500], rt_noise_sd=[0.1, 0.2], ramp_k=[1.5, 3.0],
                    da_sd_scale=[0.7, 1.0])
    if 'wide' in sys.argv:
        # GridSim physics 1.1 (14 Nov 2026): on the grid above the best point sat on the opposite edges (mu -2000,
        # sigma 6500); this grid contains it and extends both edges, as the release did once
        grid = dict(mu=[-4000, -3000, -2000, -1000, 0], sigma=[3500, 5000, 6500, 8000, 9500], rt_noise_sd=[0.1, 0.2],
                    ramp_k=[1.5, 3.0], da_sd_scale=[0.7, 1.0])
        OUT = OUT.replace('.json', '_wide.json')
    best = (1e9, None)
    for vals in itertools.product(*grid.values()):
        P = dict(base, **dict(zip(grid.keys(), vals)))
        L = 0.0; ss = {}
        for year in (2019, 2021):
            st = []
            for i, F in enumerate(Fs[year]):
                o = prices(F, P, seed=i)
                st.append(target_stats(o.rt.values, o.da.values, o.day.values))
            s = {k: float(np.mean([x[k] for x in st])) for k in st[0]}
            ss[year] = s
            L += loss(s, real[year])
        res['stageA'].append(dict(P=P, loss=L))
        if L < best[0]:
            best = (L, P, ss)
            print('best', round(L, 3), P, {y: {k: round(v, 1) for k, v in s.items()} for y, s in ss.items()}, flush=True)
    PA = best[1]
    res['stageA_best'] = dict(P=PA, loss=best[0], sim=best[2], real={y: {k: float(v) for k, v in r.items()} for y, r in real.items()})
    print('stage A done', round(time.time() - t0), flush=True)
    if 'refine' in sys.argv and os.path.exists(OUT):
        old = json.load(open(OUT))
        old['stageA_v1'] = {k: old.get(k) for k in ('stageA_best',)}
        old['stageA_best'] = res['stageA_best']; old['stageA_refine'] = res['stageA']
        cold = {k: old['calibrated'][k] for k in ('cold_fmax', 'cold_T50', 'cold_w', 'cold_tau_h')}
        old['calibrated'] = dict(PA, **cold)
        old['note_refine'] = 'stage A refined with the reserve-conditioned DA expectation; stage B (cold) kept'
        json.dump(old, open(OUT, 'w'), indent=1, default=float)
        print('refined', old['calibrated'], flush=True)
        sys.exit(0)
    # ---------------- stage B: Uri
    f = node_frame('ERCOT', 'HB_HUBAVG', '2021-01-01', '2022-01-01')
    eng = Engine('ERCO', 2021); inp = load_inputs('ERCO', 2021)
    win = (inp.index >= pd.Timestamp('2021-02-10', tz='UTC')) & (inp.index < pd.Timestamp('2021-02-21', tz='UTC'))
    inp2 = inp.copy(); inp2.loc[win, 'd'] = np.maximum(inp.d[win], inp.df[win])
    fr = f[(f.index >= pd.Timestamp(URI[0], tz='UTC')) & (f.index < pd.Timestamp(URI[1], tz='UTC'))]
    real_cap_h = int((fr.rt > 1000).sum()); real_cap_days = sorted(set(fr.index[fr.rt > 1000].strftime('%m-%d')))
    print('Uri real hours >1000:', real_cap_h, real_cap_days, flush=True)
    bestB = (1e9, None)
    for T50, w, fmax in itertools.product([-3, -5, -7], [1.5, 3.0], [0.25, 0.4, 0.55]):
        P = dict(PA, cold_fmax=fmax, cold_T50=T50, cold_w=w, cold_tau_h=36.0)
        hs = []; full = []
        for seed in (1, 2):
            F = fundamentals(eng, inp2, {}, P, seed=seed)
            o = prices(F, P, seed=seed)
            ow = o[(o.index >= pd.Timestamp(URI[0], tz='UTC')) & (o.index < pd.Timestamp(URI[1], tz='UTC'))]
            hs.append(int((ow.rt > 1000).sum()))
            # overlap of >1000 hours with the real ones
            full.append(float(((ow.rt > 1000).values & (fr.rt.values > 1000)).sum() / max(real_cap_h, 1)))
        L = (np.log((np.mean(hs) + 5) / (real_cap_h + 5))) ** 2 + (1 - np.mean(full)) ** 2
        res['stageB'].append(dict(T50=T50, w=w, fmax=fmax, hours=hs, overlap=full, loss=L))
        print('B', T50, w, fmax, hs, np.round(full, 2), round(L, 3), flush=True)
        if L < bestB[0]:
            bestB = (L, dict(cold_fmax=fmax, cold_T50=T50, cold_w=w, cold_tau_h=36.0), hs, full)
    res['stageB_best'] = dict(P=bestB[1], loss=bestB[0], sim_hours=bestB[2], overlap=bestB[3], real_hours=real_cap_h,
                              real_days=real_cap_days)
    cal = dict(PA, **bestB[1])
    res['calibrated'] = cal
    res['note'] = 'fitted on dev years 2019 and 2021 (HB_HUBAVG RT/DA) only'
    res['gridsim_physics'] = eng.physics; res['coal_adder'] = eng.coal_adder
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    print('done', cal, round(time.time() - t0), flush=True)
