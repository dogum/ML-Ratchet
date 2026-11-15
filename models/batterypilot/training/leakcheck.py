"""How much did GridSim's exam-period outage defaults (physics 1.0) move BatteryPilot's simulator?  (HANDOFF P0.5)

    GRIDSIM_PHYSICS=1.0 python leakcheck.py <tag> [--calib <calibration json>] [--rung r4_v3]
    GRIDSIM_PHYSICS=1.1 python leakcheck.py <tag> --calib checks/calibration.json

Dev years only (ERCOT 2019, 2021).  With one scarcity calibration (default: the release's) and the active GridSim
physics version, it recomputes:
 - calib.py's stage-A target statistics on simulated 2019 and 2021 (Uri masked, outage seeds 1 and 2, the
   calibrated scarcity / noise parameters) next to the real HB_HUBAVG ones, and the calibration loss;
 - calib.py's stage-B Uri numbers (hours above $1,000 and overlap with the real ones, seeds 1 and 2);
 - twin.py's test: a rung's capture on full-year twins of 2019 and 2021 (seeds 1, 2) and on real HB_HUBAVG;
 - the thermal capacity the random outages remove (mean MW out, ERCOT 2019 fleet, seeds 1 and 2).
Writes WORK/data/leak/<tag>.json.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import _paths  # noqa: E402
from pricesim import Engine, load_inputs, fundamentals, prices, availability
from simbank import calibrated, calib_file
from calib import target_stats, loss, mask_uri, URI
from evaluate import evaluate_path
from model import PolicyNet
from realdata import node_frame, fill

ROOT = _paths.WORK
COLS = ['rt', 'da', 'df', 'T', 'hour', 'day', 'dow', 'doy']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag'); ap.add_argument('--calib', default=None); ap.add_argument('--rung', default='r4_v3')
    a = ap.parse_args()
    t0 = time.time()
    cfile = a.calib or calib_file()
    cal = calibrated('v2', cfile)
    net = PolicyNet(f'{ROOT}/ckpt/{a.rung}.npz')
    out = dict(tag=a.tag, calib=os.path.basename(cfile), rung=a.rung, stats={}, real={}, twin={}, outages={})
    stageA_P = {k: cal[k] for k in ('floor_share', 'mu', 'sigma', 'rt_noise_sd', 'ramp_k', 'da_sd_scale')}
    L = 0.0
    for year in (2019, 2021):
        eng = Engine('ERCO', year); inp = load_inputs('ERCO', year)
        out['physics'] = eng.physics; out['coal_adder'] = eng.coal_adder
        f = node_frame('ERCOT', 'HB_HUBAVG', f'{year}-01-01', f'{year + 1}-01-01')
        fm = f[mask_uri(f.index)]
        real = target_stats(fm.rt.interpolate().values, fm.da.interpolate().values, fm.day.values)
        st = []
        for seed in (1, 2):
            # stage A exactly as calib.py: fundamentals with the base parameters (no cold outages), Uri masked
            F = fundamentals(eng, inp, {}, dict(floor_share=0.8), seed=seed)
            keep = mask_uri(F['times'])
            F = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in F.items()}
            F['times'] = F['times'][keep]
            o = prices(F, stageA_P, seed=seed - 1)
            st.append(target_stats(o.rt.values, o.da.values, o.day.values))
            if year == 2019:
                navail, _ = availability(eng.cf, inp.index, mode='random', seed=seed)   # as in fundamentals
                n_on = eng.cf.online_count(inp.index)
                pm = eng.cf.pmax                                                        # MW per unit, per cluster
                out['outages'][f'seed{seed}'] = dict(
                    mean_MW_out=float(((n_on - navail) * pm[None, :]).sum(1).mean()),
                    fleet_MW=float((n_on * pm[None, :]).sum(1).mean()))
        s = {k: float(np.mean([x[k] for x in st])) for k in st[0]}
        out['stats'][year] = s; out['real'][year] = {k: float(v) for k, v in real.items()}
        L += loss(s, real)
        # twin test as twin.py (full year, calibrated simulator incl. cold outages)
        fr, _ = fill(f)
        r, _ = evaluate_path({c: fr[c].values for c in COLS}, (1, 2, 4), ('pf', 'rolling', 'bp'), net=net)
        out['twin'][f'{year}_real'] = r.groupby('strategy').net.sum().to_dict()
        for seed in (1, 2):
            o = prices(fundamentals(eng, inp, {}, cal, seed=seed), cal, {}, seed=seed)
            r, _ = evaluate_path({c: o[c].values for c in COLS}, (1, 2, 4), ('pf', 'rolling', 'bp'), net=net)
            out['twin'][f'{year}_twin{seed}'] = r.groupby('strategy').net.sum().to_dict()
        print(year, round(time.time() - t0), flush=True)
    out['stageA_loss'] = L
    for v in out['twin'].values():
        v['capture_bp'] = v['bp'] / v['pf']; v['capture_rolling'] = v['rolling'] / v['pf']
    # stage B (Uri) as calib.py, with the calibration's cold-outage curve
    f = node_frame('ERCOT', 'HB_HUBAVG', '2021-01-01', '2022-01-01')
    eng = Engine('ERCO', 2021); inp = load_inputs('ERCO', 2021)
    win = (inp.index >= pd.Timestamp('2021-02-10', tz='UTC')) & (inp.index < pd.Timestamp('2021-02-21', tz='UTC'))
    inp2 = inp.copy(); inp2.loc[win, 'd'] = np.maximum(inp.d[win], inp.df[win])
    fr = f[(f.index >= pd.Timestamp(URI[0], tz='UTC')) & (f.index < pd.Timestamp(URI[1], tz='UTC'))]
    real_h = int((fr.rt > 1000).sum())
    hs, ov = [], []
    for seed in (1, 2):
        o = prices(fundamentals(eng, inp2, {}, cal, seed=seed), cal, seed=seed)
        ow = o[(o.index >= pd.Timestamp(URI[0], tz='UTC')) & (o.index < pd.Timestamp(URI[1], tz='UTC'))]
        hs.append(int((ow.rt > 1000).sum()))
        ov.append(float(((ow.rt > 1000).values & (fr.rt.values > 1000)).sum() / max(real_h, 1)))
    out['uri'] = dict(sim_hours=hs, overlap=ov, real_hours=real_h)
    out['secs'] = round(time.time() - t0)
    os.makedirs(f'{ROOT}/data/leak', exist_ok=True)
    json.dump(out, open(f'{ROOT}/data/leak/{a.tag}.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
