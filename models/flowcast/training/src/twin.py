"""Twin tests on the 52 dev basins (WY2001-2014, all non-exam).

For each dev basin the network sees exactly the same input (real ERA5-Land forcing + real attributes), so its
prediction is fixed; what changes is the target:
  real      the USGS/Caravan record
  ctwin     the basin's calibrated twin: WaterSim with the dev-calibrated parameters (checks/dev_calibrated_params.json,
            HBV/GR4J/SAC-SMA/VIC-lite + degree-day), gauge noise added
  ptwin     the "ungauged" twin: WaterSim with random generator parameters for this basin's real attributes under
            the physics version given (4 models, latent draws), gauge noise added
If the network reads twin and real alike, the simulator is good enough where it matters; where they differ
(volume, flashiness, timing, low flows, coverage) is the next rung.
usage: python3 twin.py <ckpt> <phys_ver>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import waterdata as wd
import watersim as ws
from watersim import noise
from waterdata import metrics as M
import evalreal as ER
import simbank as SB
from calib_flash import extra_correction

CAL = json.load(open(f'{WATER}/checks/dev_calibrated_params.json'))['params']


def diag(obs, idx, mean, q):
    o = obs.reindex(idx)
    r = ER.score(obs, idx, mean, q)
    v = o.notna().values
    ov, pv = o.values[v], mean[v]
    out = {k: r[k] for k in ('nse', 'kge', 'beta', 'alpha', 'r', 'fhv', 'flv', 'cov90', 'cov50', 'peak_bias_pct')}
    # timing: best mixing fraction f of the next day's prediction (local-day vs UTC-day)
    best = (0.0, out['nse'])
    pn = np.r_[mean[1:], mean[-1]][v]
    for f in (0.1, 0.2, 0.3, 0.4, 0.5):
        n = M.nse(ov, (1 - f) * pv + f * pn)
        if n > best[1]:
            best = (f, n)
    out['best_f'] = best[0]; out['nse_bestf'] = best[1]
    rb = lambda x: np.sum(np.abs(np.diff(x))) / max(np.sum(x[1:]), 1e-9)
    out['flash_ratio'] = rb(pv) / max(rb(ov), 1e-9)
    out['zero_obs'] = float(np.mean(ov < 0.005)); out['zero_pred'] = float(np.mean(pv < 0.005))
    return out


def main(name, ver):
    net, cfg = ER.load_net(name)
    nd = cfg.get('nextday', False)
    phys = SB.load_phys(ver)
    rows = []
    for k, b in enumerate(wd.basins('camels', split='dev')):
        rng = np.random.default_rng([77, k])
        idx, mean, q = ER.run_basin(net, nd, b, DEVWIN[0], DEVWIN[1])
        obs = wd.streamflow(b, 'caravan').Q_mm
        rows.append({'basin': b, 'target': 'real', **diag(obs, idx, mean, q)})
        spec = wd.spec(b); df = wd.forcing(b, '1999-10-01', DEVWIN[1]); meta = wd.forcing_meta(b)
        F = forcing_array(df.loc[DEVWIN[0]:]); clim = climate_stats(F[:, 0], F[:, 1], F[:, 2])
        for model in ('hbv', 'gr4j', 'sacsma', 'vic_lite'):
            key = f'{model}+degree_day'
            if key in CAL.get(b, {}):
                o = ws.simulate(spec, (df, meta), DEVWIN[0], DEVWIN[1], model=model, snow='degree_day',
                                params=CAL[b][key], labels=False)
                qt = pd.Series(noise.streamflow_obs(o['ts']['Q'].values, rng), index=o['ts'].index)
                rows.append({'basin': b, 'target': 'ctwin', 'model': model, **diag(qt, idx, mean, q)})
            # prior twin under physics `ver`
            sp = dict(spec); lat = {kk: float(rng.normal()) for kk in ('storage', 'perm', 'quick', 'snow')}
            m = SB.p_multiplier(phys, spec_attrs(spec), clim, rng)
            d2 = df.copy(); d2['P'] = d2['P'] * m
            sp['p_mean'] = clim['p_mean'] * m; sp['aridity'] = clim['pet_mean'] / sp['p_mean']
            prm = ws.params_from_attributes(sp, model, 'degree_day', lat, rng)
            if phys.get('shrink_total', 0.5) != 0.5:
                prm = extra_correction(prm, model, phys['shrink_total'])
            o = ws.simulate(sp, (d2, meta), DEVWIN[0], DEVWIN[1], model=model, snow='degree_day', params=prm,
                            labels=False)
            qt = pd.Series(noise.streamflow_obs(o['ts']['Q'].values, rng), index=o['ts'].index)
            rows.append({'basin': b, 'target': 'ptwin', 'model': model, 'pmult': m, **diag(qt, idx, mean, q)})
        print(k, b, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(f'{ROOT}/results/twin_{name}_{ver}.parquet')
    g = d.groupby('target')
    summ = pd.DataFrame({'nse': g.nse.median(), 'kge': g.kge.median(), 'abs_log_beta': g.beta.apply(lambda x: np.median(np.abs(np.log(x.clip(1e-3))))),
                         'beta': g.beta.median(), 'fhv': g.fhv.median(), 'flash_ratio': g.flash_ratio.median(),
                         'cov90': g.cov90.median(), 'cov50': g.cov50.median(), 'best_f': g.best_f.median(),
                         'nse_gain_f': (g.nse_bestf.median() - g.nse.median())})
    print(summ.round(3).to_string())
    summ.to_json(f'{ROOT}/results/twin_{name}_{ver}_summary.json', indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
