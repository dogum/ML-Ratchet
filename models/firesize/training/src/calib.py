"""Calibrate the suppression model (and the sim-level tau0 / ROS-adjustment distributions, by importance weights) on
dev twins: dev ignitions of the 'calib' half simulated in their own real landscape and weather (simbank twin runs).
Objective: weighted log-likelihood of each twin's REAL reported size class under the simulated distribution.
usage: calib.py <physics version> [fold]  -> data/calib_<ver>.json
These are ~13 physical parameters fitted to ~900 dev ignitions; no network sees a real label."""
import sys, json, glob, time
from common import *
import supp
from sim import TG

S = 24          # suppression draws per sim (common random numbers)


def load_bank(ver, kind):
    fs_ = sorted(glob.glob(f'{DATA}/bank/{ver}/{kind}_*.npz'))
    parts = [dict(np.load(f)) for f in fs_]
    B = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return B


# bounded parameterisation: each free parameter p in (lo, hi) via p = lo + (hi - lo) * sigmoid(v)
BOUNDS = dict(mu_resp=(np.log(0.1), np.log(6.0)), mu_resp_fed=(-1.5, 1.5), sd_resp=(0.3, 1.5),
              r0_mult=(np.log(0.1), np.log(30.0)), sd_r0=(0.2, 1.2), tau_e=(np.log(1.0), np.log(48.0)),
              f_perim=(np.log(0.3), np.log(1.5)), ros_mu=(np.log(0.25), np.log(1.2)), ros_sd=(0.3, 1.0),
              q01=(0.0, 0.5), mu_resp_nat=(-1.0, 2.0), b_urb=(-4.0, 2.0), b_agr=(-3.0, 2.0),
              b_slope=(-1.5, 0.5),
              **{f'ros_{g}': (np.log(0.22), np.log(1.4)) for g in ['GR', 'GS', 'SH', 'TU', 'TL']})
PNAMES = list(BOUNDS)
LIT = dict(mu_resp=np.log(1.0), mu_resp_fed=0.0, sd_resp=0.8, r0_mult=0.0, sd_r0=0.7, tau_e=np.log(4.0),
           f_perim=0.0, ros_mu=np.log(1.0), ros_sd=0.3, q01=0.02, mu_resp_nat=0.0, b_urb=0.0, b_agr=0.0,
           b_slope=0.0, **{f'reg_{r}': 0.0 for r in REGIONS[1:]}, **{f'ros_{g}': np.log(1.0) for g in ['GR', 'GS', 'SH', 'TU', 'TL']})
REGIONAL = True


def _to_v(p):
    return np.array([np.log((p[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - p[k])) for k in PNAMES])


def theta_from(v):
    p = {k: BOUNDS[k][0] + (BOUNDS[k][1] - BOUNDS[k][0]) / (1 + np.exp(-x)) for k, x in zip(PNAMES, v)}
    th = json.loads(json.dumps(supp.THETA0))
    th.update(mu_resp=p['mu_resp'], mu_resp_fed=p['mu_resp_fed'], sd_resp=p['sd_resp'],
              r0_mult=float(np.exp(p['r0_mult'])), sd_r0=p['sd_r0'], tau_e=float(np.exp(p['tau_e'])),
              f_perim=float(np.exp(p['f_perim'])), tau0=dict(mu=0.0, sd=1.0, mu_nat=0.0, mu_miss=0.0),
              ros=dict(mu=p['ros_mu'], sd=p['ros_sd']), q01=p['q01'],
              mu_resp_nat=p['mu_resp_nat'], b_urb=p['b_urb'], b_agr=p['b_agr'], b_slope=p['b_slope'],
              reg={r: p.get(f'reg_{r}', 0.0) for r in REGIONS[1:]},
              rep={r: p.get(f'rep_{r}', 0.0) for r in REGIONS},
              r0_fed=p.get('r0_fed', 0.0), p_man=p.get('p_man', 0.0), p_man_fed=p.get('p_man_fed', 0.0),
              ros_g={g: p[f'ros_{g}'] for g in ['GR', 'GS', 'SH', 'TU', 'TL'] if f'ros_{g}' in p}, params={k: float(x) for k, x in p.items()})
    return th


import os as _os
CALMODE = _os.environ.get('CALMODE', 'reg')      # 'reg': regional IA response offsets; 'rep': regional reporting floors
if CALMODE in ('reg', 'regrep'):
    for r in REGIONS[1:]:
        BOUNDS[f'reg_{r}'] = (-2.5, 2.5)
if CALMODE in ('rep', 'regrep'):
    for r in REGIONS:
        BOUNDS[f'rep_{r}'] = (0.0, 0.8); LIT[f'rep_{r}'] = 0.02
if CALMODE in ('rep2',):
    for r in REGIONS:
        BOUNDS[f'rep_{r}'] = (0.0, 0.8); LIT[f'rep_{r}'] = 0.02
    BOUNDS.update(r0_fed=(-3.0, 1.0), p_man=(0.0, 0.5), p_man_fed=(0.0, 0.8))
    LIT.update(r0_fed=0.0, p_man=0.01, p_man_fed=0.01)
PNAMES = list(BOUNDS)
for k in PNAMES:
    LIT.setdefault(k, 0.0)
V0 = _to_v(LIT)       # literature / rung-0 values (ROS adjustment centred on Phase 1's dev factors ~0.35-1)
ESC_LAMBDA = 1.0
ESC_BERN = float(__import__('os').environ.get('ESC_BERN', '0'))


class Twins:
    def __init__(self, ver, sel=None, kind='twin', B=None):
        B = load_bank(ver, kind) if B is None else B
        I = pd.read_parquet(f'{DATA}/ignitions.parquet').set_index('ign')
        if sel is not None:
            m = np.isin(B['ign'], sel)
            B = {k: v[m] for k, v in B.items()}
        self.B = B
        ig = I.loc[B['ign']]
        self.fed = ig.fed.values
        self.cause = np.asarray(ig.cause.astype(str).tolist(), dtype=str)
        self.nat = (self.cause == 'Natural').astype(float)
        self.region = np.asarray(ig.region.astype(str).tolist(), dtype=str)
        FX = pd.read_parquet(f'{DATA}/feat_sim.parquet', columns=['ign', 'f13_NBU', 'f13_NBA', 'slope5']).set_index('ign').loc[B['ign']]
        self.f_urb = FX.f13_NBU.values; self.f_agr = FX.f13_NBA.values; self.slope = FX.slope5.values
        self.wt = ig.weight.values
        self.bin_obs = np.searchsorted(supp.EDGES, np.log10(ig.size_ha.values), side='right')
        self.fg = np.array([supp.fgroup(f) for f in B['fuel0']])
        self.r0lit = np.array([supp.THETA0['r0'][g] for g in self.fg])
        self.uign, self.inv = np.unique(B['ign'], return_inverse=True)
        # class weights = calib-half class totals / twins kept per class (the guard skipped more escapes than non-escapes)
        if kind == 'twin':
            tw = I[I.kind == 'twin']
            dv = fod_records('dev')
            dv = dv[[dev_half(i) == 'calib' for i in dv.FOD_ID]]
            w_e = dv.escape.sum() / max((tw.escape == 1).sum(), 1)
            w_n = (dv.escape == 0).sum() / max((tw.escape == 0).sum(), 1)
            e = I.loc[self.uign].escape.values
            self.wt_ign = np.where(e == 1, w_e, w_n)
        else:
            self.wt_ign = I.loc[self.uign].weight.values
        rs = np.random.default_rng(7)
        n = len(B['ign'])
        self.z1 = rs.normal(size=(n, S))
        self.z2 = rs.normal(size=(n, S))
        self.u = rs.uniform(size=(n, S))
        self.u2 = rs.uniform(size=(n, S))
        self.u3 = rs.uniform(size=(n, S))
        self.A = B['A'].astype(float)
        self.lw = np.clip(B['lw'].astype(float), 1, 8)
        self.logA = np.log(np.maximum(self.A, 1e-6))

    def sizes(self, th):
        """n x S reported sizes (ha) under theta (common random numbers)."""
        n = len(self.A)
        mu = (th['mu_resp'] + th['mu_resp_fed'] * self.fed + th.get('mu_resp_nat', 0) * self.nat +
              th.get('b_urb', 0) * self.f_urb + th.get('b_agr', 0) * self.f_agr +
              np.array([th.get('reg', {}).get(r, 0.0) for r in self.region]))
        t_resp = np.exp(mu[:, None] + th['sd_resp'] * self.z1)
        r0 = (self.r0lit * th['r0_mult'] * np.exp(th.get('b_slope', 0) * self.slope / 10.0 + th.get('r0_fed', 0.0) * self.fed))[:, None] * np.exp(th['sd_r0'] * self.z2)
        pm = (th.get('p_man', 0.0) + (th.get('p_man_fed', 0.0) - th.get('p_man', 0.0)) * self.fed)[:, None]
        r0 = np.where(self.u3 < pm, 0.02 * r0, r0)     # managed / confine fires: little line in the first days
        t_a = self.B['tau0'][:, None] + t_resp                          # n x S
        P = th['f_perim'] * np.pi * np.sqrt(2.0 * self.A * 1e4 / np.pi * (self.lw + 1.0 / self.lw))   # n x T
        out = np.empty((n, S))
        te2 = 2.0 * th['tau_e']
        for j in range(S):
            dt = np.maximum(TG[None, :] - t_a[:, j:j + 1], 0.0)
            Lc = r0[:, j:j + 1] * (dt + dt * dt / te2)
            ok = (Lc >= P) & (TG[None, :] >= t_a[:, j:j + 1])
            hit = ok.any(1)
            k = np.argmax(ok, 1)
            k0 = np.maximum(k - 1, 0)
            ii = np.arange(n)
            g0 = Lc[ii, k0] - P[ii, k0]
            g1 = Lc[ii, k] - P[ii, k]
            f = np.where(g1 - g0 > 0, np.clip(-g0 / np.maximum(g1 - g0, 1e-12), 0, 1), 1.0)
            f = np.where(k == 0, 1.0, f)
            ls = self.logA[ii, k0] + f * (self.logA[ii, k] - self.logA[ii, k0])
            out[:, j] = np.where(hit, np.exp(ls), self.A[:, -1])
        p1 = np.array([th.get('rep', {}).get(r, 0.0) for r in self.region])[:, None]
        floor = np.where(self.u < th.get('q01', 0.0), 0.01, np.where(self.u2 < p1, 1.0, th['floor_ac'])) * ACRE_HA
        return np.maximum(out, floor)

    def sim_probs(self, th):
        """Per-sim soft label (n x NB), importance weight (n), one hard draw (bin index, n)."""
        sz = self.sizes(th)
        b = np.searchsorted(supp.EDGES, np.log10(sz), side='right')
        P = np.zeros((len(sz), supp.NB))
        for j in range(S):
            P[np.arange(len(sz)), b[:, j]] += 1.0 / S
        w = supp.imp_weight(self.B['tau0'], self.B['ros_mult'], self.cause, th, self.fg)
        w = np.where(np.isfinite(w), w, 0)
        return P, w, b[:, 0]

    def probs(self, th):
        P, w, _ = self.sim_probs(th)
        # mixture over each twin's sims
        num = np.zeros((len(self.uign), supp.NB))
        den = np.zeros(len(self.uign))
        np.add.at(num, self.inv, w[:, None] * P)
        np.add.at(den, self.inv, w)
        Pi = num / np.maximum(den, 1e-300)[:, None]
        ess = den ** 2 / np.maximum(np.bincount(self.inv, weights=w ** 2), 1e-300)
        return Pi, ess

    def obs_bins(self):
        b = np.zeros(len(self.uign), int)
        b[self.inv] = self.bin_obs
        return b

    def nll(self, v):
        self.nev = getattr(self, 'nev', 0) + 1
        th = theta_from(v)
        Pi, ess = self.probs(th)
        b = self.obs_bins()
        p = 0.98 * Pi[np.arange(len(b)), b] + 0.02 / supp.NB
        w = self.wt_ign
        e_sim = np.sum(w * Pi[:, supp.ESC_BIN:].sum(1)) / w.sum()
        e_obs = np.sum(w * (b >= supp.ESC_BIN)) / w.sum()
        lg = lambda q: np.log(q / (1 - q))
        pen = ESC_LAMBDA * (lg(np.clip(e_sim, 1e-4, 0.5)) - lg(e_obs)) ** 2     # escape prevalence moment
        pe = np.clip(Pi[:, supp.ESC_BIN:].sum(1), 1e-4, 1 - 1e-4)
        ye = (b >= supp.ESC_BIN)
        bern = -np.sum(w * np.where(ye, np.log(pe), np.log(1 - pe))) / w.sum()
        val = -np.sum(w * np.log(p)) / w.sum() + pen + ESC_BERN * bern
        if self.nev % 50 == 0:
            print('  eval', self.nev, round(val, 4), flush=True)
        return val

    def summary(self, th):
        Pi, ess = self.probs(th)
        b = self.obs_bins()
        w = self.wt_ign
        esc_p = Pi[:, supp.ESC_BIN:].sum(1)
        esc_o = (b >= supp.ESC_BIN).astype(float)
        from sklearn.metrics import roc_auc_score
        sim_med = supp.median_log10(Pi)
        cdf_o = np.array([np.sum(w * (b <= k)) / w.sum() for k in range(supp.NB)])
        cdf_s = np.array([np.sum(w * Pi[:, :k + 1].sum(1)) / w.sum() for k in range(supp.NB)])
        return dict(esc_rate_sim=float(np.sum(w * esc_p) / w.sum()), esc_rate_obs=float(np.sum(w * esc_o) / w.sum()),
                    esc_auc=float(roc_auc_score(esc_o, esc_p, sample_weight=w)),
                    cdf_obs=np.round(cdf_o, 3).tolist(), cdf_sim=np.round(cdf_s, 3).tolist(),
                    median_ess=float(np.median(ess)), nll=float(-np.sum(w * np.log(0.98 * Pi[np.arange(len(b)), b] + 0.02 / supp.NB)) / w.sum()))




def fit(T, v0=V0, iters=2000):
    from scipy.optimize import minimize
    t = time.time()
    r = minimize(T.nll, v0, method='Powell', options=dict(maxfev=iters, xtol=1e-3, ftol=1e-5))
    print('fit', r.fun, r.nfev, f'{time.time() - t:.0f}s', flush=True)
    return r.x


def main():
    ver = sys.argv[1]
    I = pd.read_parquet(f'{DATA}/ignitions.parquet')
    tw = I[I.kind == 'twin']
    fold = np.array([h01(i, 'fold') < 0.5 for i in tw.FOD_ID])
    T = Twins(ver)
    print('twins', len(T.uign), 'sims', len(T.A), flush=True)
    s0 = T.summary(theta_from(V0))
    print('rung0 literature', json.dumps(s0), flush=True)
    v = fit(T)
    s1 = T.summary(theta_from(v))
    print('calibrated', json.dumps(s1), flush=True)
    # 2-fold check of over-fitting (fit on one half of the twins, score the other)
    cv = []
    for f in (0, 1):
        Ta = Twins(ver, sel=tw.ign.values[fold == f])
        Tb = Twins(ver, sel=tw.ign.values[fold != f])
        va = fit(Ta, v, iters=400)
        cv.append(dict(train_nll=Ta.nll(va), test_nll=Tb.summary(theta_from(va))['nll'],
                       test_nll_lit=Tb.summary(theta_from(V0))['nll'], test_nll_full=Tb.summary(theta_from(v))['nll']))
    print('cv', cv, flush=True)
    th = theta_from(v)
    tag = sys.argv[2] if len(sys.argv) > 2 else ver
    json.dump(dict(ver=ver, v=v.tolist(), names=PNAMES, theta=th, lit=s0, cal=s1, cv=cv),
              open(f'{DATA}/calib_{tag}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
