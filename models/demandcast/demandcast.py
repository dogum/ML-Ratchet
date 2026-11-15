"""DemandCast v1.1: day-ahead hourly demand, wind and solar for a balancing authority, with error bars (numpy only).

Three small MLPs (two hidden layers, SiLU) read feature vectors built by `features.py` from
  - the building-stock physics forecast for tomorrow (components: base, lighting, heating, cooling, rooftop PV),
  - the wind- and solar-farm physics forecast (real fleet, pvlib / farm power curves),
  - the last 168 h of real output against the physics, weather and calendar,
and return 24 hourly means and sigmas:
  demand: mu = log(demand / physics prior)  ->  demand_MW = prior_MW * exp(mu)
  wind / solar: capacity factor (fraction of the fleet online at issue); solar is 0 at night.
Sigmas are multiplied by one widen-only factor per output fitted on dev years 2019 and 2021 (`sigma_scale`).
v1.1 (15 Nov 2026): solar reads the reporting offset of the 168 h history at issue (`solar_offset`), runs the same
network on the history put back on the physics clock, and moves the forecast onto the reported clock. EIA-930 solar
timestamps move between years; v1.0 relied on one fixed shift learned on the dev years. Pass align=False for v1.0.
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


# Solar feature layout (training/features.py solar_x): reported CF of the last 24 h before issue, the physics CF of
# the same hours, and the full 168 h history of reported minus physics CF. History index j (0..167) is LST hour
# (10 + j) % 24, so the last 24 h start at 10:00 LST two days before the target day.
_S_A, _S_P, _S_E = slice(56, 80), slice(80, 104), slice(156, 324)


def solar_offset(xs, ref=0.0, thresh=0.5):
    """Reporting offset of the solar history, read at issue time from the inputs alone: (r, r_hat) per row.

    If the reported output lags the physics by r hours, a(t) = kappa * p(t - r), so the reported-minus-physics error
    is e(t) ~ (kappa - 1) p(t) - kappa r p'(t). A least-squares fit of the 168 h error history on the physics diurnal
    profile p and its slope p' (both from the last 24 h, by LST hour) gives r_hat = -beta / (1 + alpha); r is r_hat
    rounded to -1, 0 or +1 (|r_hat| <= thresh -> 0). `ref` is the grid's r_hat on the dev years (meta
    `solar_offset_ref`), where the fixed dev shift was learned: r then counts whole hours of change since dev.
    Rows without the 168 h history (older feature vectors) get 0."""
    xs = np.atleast_2d(np.asarray(xs, np.float64))
    n = len(xs)
    r = np.zeros(n, int); rh = np.zeros(n)
    if xs.shape[1] < _S_E.stop:
        return r, rh
    h = (10 + np.arange(168)) % 24
    prof = np.zeros((n, 24)); prof[:, (10 + np.arange(24)) % 24] = xs[:, _S_P]
    dprof = (np.roll(prof, -1, 1) - np.roll(prof, 1, 1)) / 2              # slope per hour, by LST hour
    P, D, e = prof[:, h], dprof[:, h], xs[:, _S_E]
    day = P > 0.01
    for i in range(n):
        m = day[i]
        if m.sum() < 12 or not np.any(D[i, m] != 0):
            continue
        A = np.c_[P[i, m], D[i, m]]
        (al, be), *_ = np.linalg.lstsq(A, e[i, m], rcond=None)
        rh[i] = -be / max(1 + al, 0.2) - np.broadcast_to(ref, (n,))[i]
        r[i] = int(np.sign(rh[i])) if abs(rh[i]) > thresh else 0
    return r, rh


def _shift(y, r, fill):
    """y [n, 24] -> out[:, h] = y[:, h - r] (r per row in -1..1), `fill` where h - r falls outside the day."""
    out = y.copy()
    for k in (-1, 1):
        s = r == k
        if s.any():
            out[s] = np.roll(y[s], k, 1)
            out[s, 0 if k == 1 else -1] = fill
    return out


def align_solar_x(xs, r):
    """Put the last 24 h of reported solar back on the physics clock: a_al[j] = a[j + r]; the hour that falls outside
    the history is filled with the physics CF times the week's reported / physics ratio."""
    xs = np.array(xs, np.float64, ndmin=2)
    a, p = xs[:, _S_A].copy(), xs[:, _S_P]
    kap = xs[:, 55:56]
    al = _shift(a, -np.asarray(r), 0.0)
    for k, j in ((1, 23), (-1, 0)):
        s = np.asarray(r) == k
        al[s, j] = kap[s, 0] * p[s, j]
    xs[:, _S_A] = al
    return xs


class DemandCast:
    def __init__(self, path=os.path.join(HERE, 'demandcast_weights.npz')):
        z = np.load(path)
        self.W = {k: {n[2:]: z[n].astype(np.float64) for n in z.files if n.startswith(k + '.')} for k in ('d', 'w', 's')}
        meta = json.load(open(os.path.join(HERE, 'demandcast_meta.json')))
        self.sigma_scale = meta['sigma_scale']
        self.meta = meta

    def offset_ref(self, ba, n):
        """Dev reference of the solar offset estimate per row (0 for an unknown grid or ba=None)."""
        ref = self.meta.get('solar_offset_ref', {})
        if ba is None:
            return np.zeros(n)
        return np.array([ref.get(b, 0.0) for b in np.broadcast_to(np.asarray(ba), (n,))], float)

    def predict_demand(self, xd, prior_total):
        """xd [n, 329] features; prior_total [n, 24] physics demand (MW). Returns mean MW, sigma of log demand
        (already dev-scaled), and the 95 % band in MW."""
        mu, sig = _forward(self.W['d'], xd, 'd')
        sig = sig * self.sigma_scale['d']
        p = prior_total * np.exp(mu)
        return dict(mw=p, log_sigma=sig, lo95=prior_total * np.exp(mu - 1.96 * sig), hi95=prior_total * np.exp(mu + 1.96 * sig))

    def predict_cf(self, kind, x, prior_cf, night=None, align=True, ba=None):
        """kind 'w' or 's'; x [n, 164|156]; prior_cf [n, 24] physics capacity factor. Returns CF mean and sigma.
        align (solar only): read the reporting offset of the solar history from x (solar_offset), put the history back
        on the physics clock, run the network, then move the forecast by the same offset onto the reported clock.
        ba: the grid (or one per row), for its dev reference offset; None = 0."""
        x = np.atleast_2d(x); r = np.zeros(len(x), int)
        if kind == 's' and align:
            r, _ = solar_offset(x, self.offset_ref(ba, len(x))); x = align_solar_x(x, r)
        mu, sig = _forward(self.W[kind], x, kind, prior_cf)
        mu = np.clip(mu, 0, 1.2); sig = sig * self.sigma_scale[kind]
        if kind == 's' and night is not None:
            mu = np.where(night > 0.5, 0.0, mu); sig = np.where(night > 0.5, FLOOR['s'], sig)
        if r.any():
            mu = _shift(mu, r, 0.0); sig = _shift(sig, r, FLOOR['s'])
        out = dict(cf=mu, sigma=sig, lo95=np.clip(mu - 1.96 * sig, 0, None), hi95=mu + 1.96 * sig)
        if kind == 's':
            out['offset_h'] = r
        return out
