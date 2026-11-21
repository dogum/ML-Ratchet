"""Dev-only calibration of the released rung: (1) a 2-parameter linear map from the network's simulated crown
fraction to the MTBS high-severity fraction (and the same map for each classical index), (2) widen-only sigma factors
per output so the 80 % interval covers >= 80 % of dev truths. Output models/calib_<rung>.json.

python3 calibrate.py <rung>
"""
from common import *
from scipy.stats import spearmanr
import model as M
from ratchet import load_net
from train import real_predict
from real import load_records

Z80 = 1.2816


def linfit(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    A = np.c_[np.ones(ok.sum()), x[ok]]
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    res = y[ok] - A @ coef
    return [float(coef[0]), float(coef[1])], float(np.std(res))


def widen(z):
    z = np.abs(z[np.isfinite(z)])
    if len(z) < 3:
        return 1.0
    return float(max(1.0, np.quantile(z, 0.8) / Z80))


def calibrate(rung, recs=None, base=None):
    net, ns = load_net(rung)
    recs = recs or [r for r in load_records(f'{REAL}/dev_records_package.pkl') if r is not None]
    p = real_predict(net, ns, recs)
    hs = np.array([r['hs'] for r in recs]); wind = np.array([r['wind'] for r in recs]); lf = np.array([r['lfmc'] for r in recs])
    out = dict(rung=rung, n_dev=len(recs))
    # severity: HS ~ a + b * crown_pred ; sigma via delta method on the logit-like transform
    (a, b), rsd = linfit(p['crown'], hs)
    dcdy = (1.02 + 0.02) * np.exp(p['crown_y']) / (1 + np.exp(p['crown_y'])) ** 2
    sd_hs = np.sqrt((abs(b) * dcdy * p['crown_sd']) ** 2)
    pred_hs = a + b * p['crown']
    out['hs_map'] = dict(a=a, b=b, resid_sd=rsd, n=int(np.isfinite(hs).sum()))
    out['hs_widen'] = widen((pred_hs - hs) / sd_hs)
    out['wind_widen'] = widen((p['wind_y'] - np.log(np.maximum(wind, 0.3))) / p['wind_sd'])
    out['lfmc_widen'] = widen((p['lfmc'] - lf) / p['lfmc_sd'])
    out['lfmc_n'] = int(np.isfinite(lf).sum())
    out['dev_rho'] = dict(hs=float(spearmanr(p['crown'][np.isfinite(hs)], hs[np.isfinite(hs)])[0]),
                          wind=float(spearmanr(p['wind'], wind)[0]))
    # classical maps (dev): HS ~ index, wind ~ back-calculated wind
    if base is None:
        base = json.load(open(latest('out/baselines_dev.json')))
    for k in ('cfp', 'erc_max', 'bi_max', 'fwi_max'):
        x = np.array([base[r['fid']][k] for r in recs], float)
        out[f'map_{k}'] = linfit(x, hs)[0]
    x = np.array([base[r['fid']]['wind_bc'] for r in recs], float)
    out['map_wind_bc'] = linfit(x, wind)[0]
    out['dev_median_hs'] = float(np.nanmedian(hs)); out['dev_median_wind'] = float(np.nanmedian(wind))
    return out


if __name__ == '__main__':
    rung = sys.argv[1]
    c = calibrate(rung)
    jdump(c, f'{ROOT}/models/calib_{rung}.json')
    print(json.dumps(c, indent=1))
