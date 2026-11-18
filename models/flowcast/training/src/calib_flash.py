"""Rung p2 physics measurement (dev only): is the generated hydrograph as flashy as the real one?

WaterSim's generator applies HALF of the dev-calibrated median parameter shift (SHRINK = 0.5, chosen in Phase 1 on
the NSE of a single prior run). For training a network what matters is that the DISTRIBUTION of simulated
hydrographs matches real ones. For each dev basin (WY2001-2014, its measured water multiplier m* from
calib_water.py so volume is matched) we draw 3 random parameter sets x 4 models and compare the Richards-Baker
flashiness and the top-2 % flow share with the observed record, for total correction s = 0.5 (shipped), 0.75, 1.0.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import waterdata as wd
import watersim as ws
from watersim.generator import _bounds

REP = json.load(open(f'{WATER}/checks/prior_fit.json'))['report']


def extra_correction(prm, model, s):
    """Apply (s - 0.5) x the dev median log/additive shift on top of the generator's half correction."""
    rep = REP.get(model, {})
    b = _bounds(model)
    out = dict(prm)
    for k, r in rep.items():
        if k not in out:
            continue
        d = (s - 0.5) * r['median_diff']
        out[k] = out[k] * math.exp(d) if r['mode'] == 'mult' else out[k] + d
        if k in b:
            out[k] = min(max(out[k], b[k][0]), b[k][1])
    return out


def sigs(q):
    q = np.asarray(q, float); q = q[np.isfinite(q)]
    rb = np.sum(np.abs(np.diff(q))) / max(np.sum(q[1:]), 1e-9)
    top = np.sort(q)[::-1][:max(1, int(0.02 * len(q)))].sum() / max(q.sum(), 1e-9)
    return rb, top


if __name__ == '__main__':
    cw = pd.read_parquet(f'{ROOT}/results/calib_water.parquet').set_index('basin')
    rows = []
    for k, b in enumerate(wd.basins('camels', split='dev')):
        spec = wd.spec(b); df = wd.forcing(b, '1999-10-01', DEVWIN[1]); meta = wd.forcing_meta(b)
        obs = wd.streamflow(b, 'caravan').Q_mm.loc[DEVWIN[0]:DEVWIN[1]]
        rb_o, top_o = sigs(obs.values)
        m = float(cw.loc[b, 'm_star'])
        d2 = df.copy(); d2['P'] *= m
        F = forcing_array(df.loc[DEVWIN[0]:]); clim = climate_stats(F[:, 0], F[:, 1], F[:, 2])
        sp = dict(spec); sp['p_mean'] = clim['p_mean'] * m; sp['aridity'] = clim['pet_mean'] / sp['p_mean']
        rng = np.random.default_rng([5, k])
        for model in ('hbv', 'gr4j', 'sacsma', 'vic_lite'):
            for rep in range(3):
                lat = {kk: float(rng.normal()) for kk in ('storage', 'perm', 'quick', 'snow')}
                seed = int(rng.integers(2**31))
                for s in (0.5, 0.75, 1.0):
                    prm = ws.params_from_attributes(sp, model, 'degree_day', lat, np.random.default_rng(seed))
                    prm = extra_correction(prm, model, s)
                    o = ws.simulate(sp, (d2, meta), DEVWIN[0], DEVWIN[1], model=model, snow='degree_day', params=prm,
                                    labels=False)
                    q = o['ts']['Q'].reindex(obs.index)[obs.notna()]
                    rb, top = sigs(q.values)
                    rows.append({'basin': b, 'model': model, 'rep': rep, 's': s, 'rb_ratio': rb / rb_o,
                                 'top2_ratio': top / top_o, 'nse': ws.metrics.nse(obs[obs.notna()].values, q.values)})
        print(k, b, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(f'{ROOT}/results/calib_flash.parquet')
    g = d.groupby('s')
    summ = pd.DataFrame({'median_log_rb_ratio': g.rb_ratio.apply(lambda x: np.median(np.log(x))),
                         'median_log_top2_ratio': g.top2_ratio.apply(lambda x: np.median(np.log(x))),
                         'median_nse_single_run': g.nse.median()})
    print(summ.round(3))
    print(d.groupby(['s', 'model']).rb_ratio.median().unstack().round(2))
    summ.to_json(f'{ROOT}/results/calib_flash_summary.json', indent=1)
