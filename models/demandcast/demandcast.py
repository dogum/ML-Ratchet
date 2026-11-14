"""DemandCast v1.0: day-ahead hourly demand, wind and solar for a balancing authority, with error bars (numpy only).

Three small MLPs (two hidden layers, SiLU) read feature vectors built by `features.py` from
  - the building-stock physics forecast for tomorrow (components: base, lighting, heating, cooling, rooftop PV),
  - the wind- and solar-farm physics forecast (real fleet, pvlib / farm power curves),
  - the last 168 h of real output against the physics, weather and calendar,
and return 24 hourly means and sigmas:
  demand: mu = log(demand / physics prior)  ->  demand_MW = prior_MW * exp(mu)
  wind / solar: capacity factor (fraction of the fleet online at issue); solar is 0 at night.
Sigmas are multiplied by one widen-only factor per output fitted on dev years 2019 and 2021 (`sigma_scale`).
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR = {'d': 0.004, 'w': 0.005, 's': 0.003}


def _forward(W, x, kind, prior=None):
    x = np.asarray(x, np.float64)[..., :W['xm'].shape[0]]       # v1.0 nets read the first (base) features
    z = (x - W['xm']) / W['xs']
    i = 0
    while f'body.{i}.weight' in W:
        z = z @ W[f'body.{i}.weight'].T + W[f'body.{i}.bias']
        z = z / (1 + np.exp(-z))                         # SiLU
        i += 2
    mu = z @ W['mu.weight'].T + W['mu.bias']
    sig = np.logaddexp(0, z @ W['ls.weight'].T + W['ls.bias']) + FLOOR[kind]
    if kind != 'd':
        mu = mu + prior
    return mu, sig


class DemandCast:
    def __init__(self, path=os.path.join(HERE, 'demandcast_weights.npz')):
        z = np.load(path)
        self.W = {k: {n[2:]: z[n].astype(np.float64) for n in z.files if n.startswith(k + '.')} for k in ('d', 'w', 's')}
        meta = json.load(open(os.path.join(HERE, 'demandcast_meta.json')))
        self.sigma_scale = meta['sigma_scale']
        self.meta = meta

    def predict_demand(self, xd, prior_total):
        """xd [n, 329] features; prior_total [n, 24] physics demand (MW). Returns mean MW, sigma of log demand
        (already dev-scaled), and the 95 % band in MW."""
        mu, sig = _forward(self.W['d'], xd, 'd')
        sig = sig * self.sigma_scale['d']
        p = prior_total * np.exp(mu)
        return dict(mw=p, log_sigma=sig, lo95=prior_total * np.exp(mu - 1.96 * sig), hi95=prior_total * np.exp(mu + 1.96 * sig))

    def predict_cf(self, kind, x, prior_cf, night=None):
        """kind 'w' or 's'; x [n, 164|156]; prior_cf [n, 24] physics capacity factor. Returns CF mean and sigma."""
        mu, sig = _forward(self.W[kind], x, kind, prior_cf)
        mu = np.clip(mu, 0, 1.2); sig = sig * self.sigma_scale[kind]
        if kind == 's' and night is not None:
            mu = np.where(night > 0.5, 0.0, mu); sig = np.where(night > 0.5, FLOOR['s'], sig)
        return dict(cf=mu, sigma=sig, lo95=np.clip(mu - 1.96 * sig, 0, None), hi95=mu + 1.96 * sig)
