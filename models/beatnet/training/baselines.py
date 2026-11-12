"""Classical baselines on the same beats.
  majority  : every beat is N.
  rules     : textbook thresholds, not tuned: a beat is V if its QRS is wide (> 120 ms) or its shape departs from the
              record's median beat (correlation < 0.7) while it is not narrow-and-premature-only; S if it is premature
              (pre-RR < 0.85 x local mean RR and the RR before is shorter than the one after) with a normal shape;
              otherwise N. F cannot be told apart by rules.
  LD-sim    : de Chazal et al. 2004 style linear discriminant (RR intervals, QRS width, 10 QRS and 8 T-wave samples)
              trained on the simulated bank, never on real labels."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import prep
C0 = prep.HALF; FS = prep.FS

def qrs_width(w):
    """classical QRS width (ms) from a 180 Hz window: span around the fiducial where the smoothed |slope| exceeds 15% of its max"""
    d = np.abs(np.diff(w, axis=-1)); d = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, 'same'), -1, d)
    lo, hi = C0 - int(0.15 * FS), C0 + int(0.15 * FS); seg = d[:, lo:hi]; thr = 0.15 * seg.max(1, keepdims=True); out = []
    for i, r in enumerate(seg > thr):
        k = np.nonzero(r)[0]; c = C0 - lo; inside = k[(k > c - int(0.1 * FS)) & (k < c + int(0.12 * FS))]
        out.append((inside[-1] - inside[0] + 1) / FS * 1000 if len(inside) else 0.0)
    return np.array(out)

def shape_corr(w, tpl):
    a = w[:, C0 - 18:C0 + 27]; b = tpl[:, C0 - 18:C0 + 27]; a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.maximum(np.sqrt((a * a).sum(1) * (b * b).sum(1)), 1e-9)

def rules(Z, idx):
    w = Z['w'][idx].astype(np.float32); t = Z['tpl'][idx].astype(np.float32); rr = Z['rr'][idx]; qw = qrs_width(w); cr = shape_corr(w, t)
    prem = (rr[:, 0] < np.log(0.85)) & (rr[:, 3] < 0); wide = qw > 120; odd = cr < 0.7
    return np.where(wide | (odd & ~prem), 2, np.where(prem, 1, 0))

def ld_features(Z, idx):
    w = Z['w'][idx].astype(np.float32); rr = np.nan_to_num(Z['rr'][idx]); qw = qrs_width(w)
    q = w[:, C0 + np.linspace(-0.05 * FS, 0.1 * FS, 10).astype(int)]; tt = w[:, C0 + np.linspace(0.15 * FS, 0.5 * FS, 8).astype(int)]
    return np.concatenate([rr[:, :4], qw[:, None] / 100, q, tt], 1)

def fit_ld(bank):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    X = ld_features(bank, np.arange(len(bank['y']))); m = LDA(priors=[0.25] * 4).fit(X, bank['y']); return m
