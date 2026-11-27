"""FireSize v1.0.1 inference (numpy only). v1.0.1 (27 Nov 2026) clips standardised inputs at 8 sd by default.

FireSize predicts, from an ignition record plus landscape and weather summaries, the probability that a US wildfire
escapes to >= 100 acres and the distribution of its final size over 12 FPA-FOD size classes. It was trained only on
simulated ignitions (FireSim spread in real LANDFIRE windows under real gridMET weather + a dev-calibrated
initial-attack model) and graded once on 655,869 real FPA-FOD ignitions (2012-2020).

    import firesize
    m = firesize.FireSize()                     # loads firesize_weights.npz next to this file
    out = m.predict(features)                   # pandas DataFrame / dict with the 75 feature columns, or an n x 75 array
    out['p_escape'], out['p_large'], out['median_acres'], out['interval80_acres'], out['probs'] (n x 12)

Features (m.features, 75 columns): see MODEL_CARD.md. `make_features(...)` below builds them from 13 x 13 patches of
the 250 m LANDFIRE 2016 grids around the ignition (fuel model code, canopy cover %, canopy height, slope), gridMET daily
weather of the discovery day and the day before, and the record (cause, agency, discovery hour, day of year, region).
Missing values (NaN) are filled with the training medians.
"""
import os
import warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACRE_HA = 0.404686
# size classes (acres): <0.1, 0.1-0.25, 0.25-1, 1-3, 3-10, 10-30, 30-100, 100-300, 300-1000, 1000-5000, 5000-20000, >=20000
EDGES_AC = np.array([0.095, 0.255, 0.995, 2.995, 9.995, 29.995, 99.995, 299.995, 999.995, 4999.995, 19999.995])
CLASS_LABELS = ['<0.1', '0.1-0.25', '0.25-1', '1-3', '3-10', '10-30', '30-100', '100-300', '300-1k', '1k-5k',
                '5k-20k', '>=20k']
ESC_BIN, LARGE_BIN = 7, 9
_LE = np.log10(EDGES_AC)
_E = np.r_[_LE[0] - 0.6, _LE, _LE[-1] + 0.6]
REGIONS = ['CA', 'NW', 'NR', 'GB', 'SW', 'RM', 'SP', 'SE', 'NE_MW']
CAUSES = ['Human', 'Natural', 'Missing data/not specified/undetermined']
FCLS = ['GR', 'GS', 'SH', 'TU', 'TL', 'SB', 'NBU', 'NBA', 'NBO']
WXF = ['erc', 'bi', 'fm100', 'fm1000', 'vs', 'rmin', 'rmax', 'tmmx', 'tmmn', 'pr', 'vpd']


class FireSize:
    def __init__(self, weights=os.path.join(HERE, 'firesize_weights.npz')):
        z = np.load(weights, allow_pickle=False)
        self.W = {k: z[k] for k in z.files}
        self.features = [str(f) for f in self.W['feats']]
        self.temp = float(self.W['temp'])

    def matrix(self, X):
        if hasattr(X, 'columns'):
            return X[self.features].to_numpy(float)
        if isinstance(X, dict):
            return np.column_stack([np.atleast_1d(np.asarray(X[f], float)) for f in self.features])
        return np.atleast_2d(np.asarray(X, float))

    z_clip = 8.0    # v1.0.1 (27 Nov 2026): clip standardised inputs at +-8 sd; None restores v1.0 (exam re-score in NOTEBOOK)

    def probs(self, X, temp=None):
        """n x 12 class probabilities (ensemble mean of 3 MLPs; temp = widen-only temperature fitted on dev).

        Audit, 25 Nov 2026: no simulated ignition landed on slash-blowdown fuel (LANDFIRE 201-204), so pt_SB has sd 1e-6
        and a real one enters at z = 1e6: escape probability exactly 0 and a 1.7263-acre median whatever the weather.
        91 of the 655,869 exam ignitions were scored that way. A warning names such rows. Since v1.0.1 z_clip = 8 clips
        them (dated exam re-score, 27 Nov 2026: on those 91 the size MAE fell from 1.237 to 0.750 log10; pooled escape
        AUC 0.7266 -> 0.7264). Set z_clip = None for v1.0's outputs."""
        W = self.W
        Z = self.matrix(X)
        Z = np.where(np.isfinite(Z), Z, W['fill'][None, :])
        z = (Z - W['mu']) / W['sd']
        big = np.abs(z) > 8
        if big.any():
            cols = sorted({self.features[j] for j in np.nonzero(big)[1]})[:6]
            warnings.warn(f'FireSize: {int(big.any(1).sum())} row(s) have inputs beyond 8 sd of the simulated training '
                          f'data ({cols}; max {np.abs(z).max():.3g}); the network was never trained there', stacklevel=2)
        if self.z_clip:                      # independent of the warning, so a row does not depend on its batch (Codex)
            z = np.clip(z, -self.z_clip, self.z_clip)
        T = self.temp if temp is None else temp
        out = 0
        for e in range(int(W['n_ens'])):
            h, k = z, 0
            while f'e{e}_L{k}_w' in W:
                h = h @ W[f'e{e}_L{k}_w'].T + W[f'e{e}_L{k}_b']
                if f'e{e}_L{k + 1}_w' in W:
                    h = np.maximum(h, 0.0)
                k += 1
            h = h / T
            h = h - h.max(1, keepdims=True)
            p = np.exp(h)
            out = out + p / p.sum(1, keepdims=True)
        return out / int(W['n_ens'])

    @staticmethod
    def quantile(p, level):
        c = np.cumsum(p, 1)
        k = np.argmax(c >= level, 1)
        cprev = np.where(k > 0, c[np.arange(len(k)), np.maximum(k - 1, 0)], 0.0)
        f = np.clip((level - cprev) / np.maximum(p[np.arange(len(k)), k], 1e-12), 0, 1)
        return 10 ** (_E[k] + f * (_E[k + 1] - _E[k]))

    def predict(self, X):
        p = self.probs(X)                       # widened (calibrated on dev): probabilities and the 80 % interval
        p1 = self.probs(X, temp=1.0)            # sharp: point estimate (median), as graded in the exam
        return dict(probs=p, p_escape=p[:, ESC_BIN:].sum(1), p_large=p[:, LARGE_BIN:].sum(1),
                    median_acres=self.quantile(p1, 0.5),
                    interval80_acres=np.column_stack([self.quantile(p, 0.1), self.quantile(p, 0.9)]))


# ---------------------------------------------------------------- feature code (vendored from training/features.py)
def fuel_class(code):
    c = np.asarray(code, int)
    out = np.full(c.shape, 8, int)
    for k, (lo, hi) in enumerate([(101, 109), (121, 124), (141, 149), (161, 165), (181, 189), (201, 204)]):
        out[(c >= lo) & (c <= hi)] = k
    out[c == 91] = 6
    out[c == 93] = 7
    return out


def simard_emc(t_c, rh):
    T = np.asarray(t_c, float) * 1.8 + 32.0
    H = np.clip(np.asarray(rh, float), 0, 100)
    return np.where(H < 10, 0.03229 + 0.281073 * H - 0.000578 * H * T,
                    np.where(H < 50, 2.22749 + 0.160107 * H - 0.01478 * T,
                             21.0606 + 0.005565 * H * H - 0.00035 * H * T - 0.483199 * H))


_LUT = None
M1 = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35.0])
U10 = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20.0])


def ros_mmin(code, m1, u10):
    """Rothermel head ROS (m/min) from FireSim's behave() lookup table (ros_lut.npz): fuel code x 1-h moisture x 10 m wind."""
    global _LUT
    if _LUT is None:
        _LUT = np.load(os.path.join(HERE, 'ros_lut.npz'))['lut']
    code = np.asarray(code, int)
    i = np.interp(m1, M1, np.arange(len(M1))); j = np.interp(u10, U10, np.arange(len(U10)))
    i0 = np.floor(i).astype(int); j0 = np.floor(j).astype(int)
    i1 = np.minimum(i0 + 1, len(M1) - 1); j1 = np.minimum(j0 + 1, len(U10) - 1)
    fi, fj = i - i0, j - j0
    L = _LUT
    return ((1 - fi) * (1 - fj) * L[code, i0, j0] + fi * (1 - fj) * L[code, i1, j0] + (1 - fi) * fj * L[code, i0, j1]
            + fi * fj * L[code, i1, j1])


def make_features(fuel13, cc13, ch13, slope13, wx0, wx1, cause, federal, hour, doy, region):
    """One ignition. fuel13/cc13/ch13/slope13: 13 x 13 patches (250 m cells, ignition at [6, 6]) of LANDFIRE 2016 FBFM40,
    CC (%), CH, slope (deg). wx0 / wx1: dicts of gridMET daily values (WXF keys; tmmx, tmmn in deg C, vs in m/s, rmin /
    rmax in %) for the discovery day and the day before. cause: one of CAUSES; federal: 0/1; hour: local discovery hour
    (float or None); doy: day of year; region: one of REGIONS. Returns a dict of the 75 feature columns."""
    f = {}
    q = fuel_class(fuel13)
    for k, nm in enumerate(FCLS):
        f[f'pt_{nm}'] = float(q[6, 6] == k); f[f'f5_{nm}'] = float((q[4:9, 4:9] == k).mean())
        f[f'f13_{nm}'] = float((q == k).mean())
    cc = np.where(np.asarray(cc13, float) > 100, 0, np.asarray(cc13, float))
    f['cc5'] = float(cc[4:9, 4:9].mean()); f['ch5'] = float(np.asarray(ch13, float)[4:9, 4:9].mean())
    f['slope5'] = float(np.asarray(slope13, float)[4:9, 4:9].mean()); f['slope_pt'] = float(np.asarray(slope13)[6, 6])
    for v in WXF:
        f[v] = float(wx0.get(v, np.nan)); f[v + '_m1'] = float(wx1.get(v, np.nan))
    miss = hour is None or not np.isfinite(hour)
    h = 12.0 if miss else float(hour)
    f.update(hr_sin=0.0 if miss else np.sin(2 * np.pi * h / 24), hr_cos=0.0 if miss else np.cos(2 * np.pi * h / 24),
             hr_miss=float(miss), doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25),
             fed=float(federal))
    for k, nm in zip(CAUSES, ['c_hum', 'c_nat', 'c_miss']):
        f[nm] = float(cause == k)
    for rg in REGIONS:
        f[f'r_{rg}'] = float(region == rg)
    # Missing weather: the same fallbacks as training (training/src/physfeat.py): 1-h moisture 10 %, 10 m wind 3 m/s.
    m1 = float(np.nan_to_num(np.clip(simard_emc(f['tmmx'], f['rmin']), 1, 35), nan=10.0))
    u = float(np.nan_to_num(f['vs'] * 1.45, nan=3.0))
    R = np.log1p(ros_mmin(np.asarray(fuel13, int), np.full((13, 13), m1), np.full((13, 13), u)))
    f['ros_pt'] = float(R[6, 6]); f['ros5'] = float(R[4:9, 4:9].mean()); f['ros13'] = float(R.mean())
    m1b = float(np.nan_to_num(np.clip(simard_emc(f['tmmx_m1'], f['rmin_m1']), 1, 35), nan=10.0))
    f['ros_pt_m1'] = float(np.log1p(ros_mmin(int(np.asarray(fuel13)[6, 6]), m1b,
                                             float(np.nan_to_num(f['vs_m1'] * 1.45, nan=3.0)))))
    return f
