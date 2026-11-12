"""BeatNet cardiac source model.

A voxel heart (2 mm grid): a thick-walled half-ellipsoid left ventricle, a crescent right ventricle wrapped around it,
and two thin-walled atria above the valve plane (electrically insulated from the ventricles, as the fibrous AV ring is).

Activation: the eikonal equation |grad tau| = 1/v solved as shortest paths (Dijkstra) on a 26-neighbour voxel graph.
  Ventricular working myocardium ~0.5 mm/ms; a fast endocardial layer (the Purkinje network) ~2 mm/ms that the His
  bundle reaches at a few root sites; atrial myocardium ~0.9 mm/ms.

Electrical source: each voxel's transmembrane potential is Vm(x,t) = f(t - tau(x)) - g(t - r(x)), with f a fast upstroke
and g a slower phase-3 repolarisation, r = tau + APD(x) the repolarisation time. The impressed-current dipole of the
whole heart is H(t) = -sigma * integral(grad Vm) dV = sigma * integral[f'(t-tau) grad tau - g'(t-r) grad r] dV
(volume form of the equivalent double layer). So the QRS is the activation-time gradient field swept by the upstroke, and
the T wave is minus the repolarisation-time gradient field swept by the downstroke.

Body frame (Frank convention): x = to the patient's left, y = toward the feet, z = toward the back.
Lead voltages: v = L . H with L the Dower lead vectors (measured on Frank's torso model)."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import distance_transform_edt

HV = 2.0          # voxel size, mm
DT = 1.0          # dipole time step, ms
# Dower 1980/1988 transformation: 12-lead from Frank X, Y, Z (rows: lead, cols: X Y Z)
DOWER = {'I': (0.632, -0.235, 0.059), 'II': (0.235, 1.066, -0.132), 'V1': (-0.515, 0.157, -0.917), 'V2': (0.044, 0.164, -1.387),
         'V3': (0.882, 0.098, -1.277), 'V4': (1.213, 0.127, -0.601), 'V5': (1.125, 0.127, -0.086), 'V6': (0.831, 0.076, 0.230)}
DOWER['III'] = tuple(np.subtract(DOWER['II'], DOWER['I'])); DOWER['aVF'] = tuple(np.subtract(DOWER['II'], 0.5 * np.array(DOWER['I'])))
OFFS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])
MYO, LVF, RVF, ATR = 0, 1, 2, 3     # voxel classes (atrial voxels: 3)


def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)


def heart_basis():
    """unit vectors of the heart frame (p: toward the RV, q, w: apex->base) in body coordinates, standard anatomy"""
    w = -unit([0.62, 0.55, -0.56])                           # apex points left, down and forward
    p = unit([-0.55, 0.2, -0.8]); p = unit(p - (p @ w) * w)   # the RV lies to the right and in front of the LV
    q = np.cross(w, p); return np.stack([p, q, w])            # rows


def rot(ax, ang):
    ax = unit(ax); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class Heart:
    def __init__(self, g=None, seed=0):
        """g: geometry parameters (mm). Builds the voxel model, the graph, transmural and apico-basal coordinates."""
        rng = np.random.default_rng(seed); G = dict(Rc=24.0, Lc=52.0, T=10.0, d=20.0, Rp=26.0, Rq=34.0, Lr=42.0, Trv=4.0, Ra=19.0, Ta=2.5, purk_top=0.2)
        G.update(g or {}); self.G = G; B = heart_basis(); self.B = B
        post = 1.0 if B[1] @ np.array([0, 0, 1.0]) > 0 else -1.0     # +q is posterior if post > 0
        lo = np.array([-G['Rc'] - G['T'] - 4, -max(G['Rq'], G['Rc']) - G['T'] - 6, -G['Lc'] - G['T'] - 4]); hi = np.array([G['d'] + G['Rp'] + G['Trv'] + 6, max(G['Rq'], G['Rc']) + G['T'] + 6, 2 * G['Ra'] + 34])
        ax = [np.arange(lo[i], hi[i], HV) for i in range(3)]; P = np.stack(np.meshgrid(*ax, indexing='ij'), -1); self.shape = P.shape[:3]
        E = lambda c, r: (((P - np.array(c)) / np.array(r)) ** 2).sum(-1)
        vent = P[..., 2] <= 0
        lv_c = (E((0, 0, 0), (G['Rc'], G['Rc'], G['Lc'])) <= 1) & vent; lv_e = (E((0, 0, 0), (G['Rc'] + G['T'],) * 2 + (G['Lc'] + G['T'],)) <= 1) & vent
        rv_c = (E((G['d'], 0, 0), (G['Rp'], G['Rq'], G['Lr'])) <= 1) & vent & ~lv_e
        rv_e = (E((G['d'], 0, 0), (G['Rp'] + G['Trv'], G['Rq'] + G['Trv'], G['Lr'] + G['Trv'])) <= 1) & vent
        lv = lv_e & ~lv_c; rv = rv_e & ~rv_c & ~lv_e
        # atria: two thin spherical shells above the valve plane, sharing a septum
        ca = (G['d'] - 2, -5 * post, G['Ra'] + 4); cl = (-6, 10 * post, G['Ra'] + 4)
        ra_o = E(ca, (G['Ra'] + G['Ta'],) * 3) <= 1; ra_i = E(ca, (G['Ra'],) * 3) <= 1; la_o = E(cl, (G['Ra'] - 1 + G['Ta'],) * 3) <= 1; la_i = E(cl, (G['Ra'] - 1,) * 3) <= 1
        atr = (ra_o | la_o) & ~ra_i & ~la_i & (P[..., 2] > HV)
        tissue = lv | rv | atr; cav = (lv_c | rv_c | ra_i | la_i)
        self.idx = -np.ones(self.shape, int); ijk = np.argwhere(tissue); self.ijk = ijk; n = len(ijk); self.idx[tuple(ijk.T)] = np.arange(n); self.n = n
        self.pos = P[tuple(ijk.T)]                                   # heart-frame mm
        self.atrial = atr[tuple(ijk.T)]; self.lv = lv[tuple(ijk.T)]; self.rv = rv[tuple(ijk.T)]
        # endocardium = tissue next to a cavity; epicardium = tissue next to the outside
        d_cav = distance_transform_edt(~cav, HV)[tuple(ijk.T)]; d_out = distance_transform_edt(tissue | cav, HV)[tuple(ijk.T)]
        self.e = np.clip((d_cav - HV) / np.maximum(d_cav + d_out - 2 * HV, 1e-6), 0, 1)        # 0 endocardium .. 1 epicardium
        endo_lv = (d_cav <= 1.5 * HV) & ~self.atrial & (distance_transform_edt(~lv_c, HV)[tuple(ijk.T)] <= 1.5 * HV)
        endo_rv = (d_cav <= 1.5 * HV) & ~self.atrial & (distance_transform_edt(~rv_c, HV)[tuple(ijk.T)] <= 1.5 * HV) & ~endo_lv
        wv = self.pos[:, 2]; self.apex = wv.min(); self.b = np.clip((wv - self.apex) / -self.apex, 0, 1)      # 0 apex .. 1 base (ventricles)
        purk = self.b < 1 - G['purk_top']
        self.cls = np.full(n, MYO); self.cls[endo_lv & purk] = LVF; self.cls[endo_rv & purk] = RVF; self.cls[self.atrial] = ATR
        self.endo_lv, self.endo_rv = endo_lv, endo_rv
        # 26-neighbour graph; no edges between atria and ventricles (insulating AV ring)
        I, J, Lng = [], [], []
        for o in OFFS:
            nb = ijk + o; ok = np.all((nb >= 0) & (nb < np.array(self.shape)), 1); j = -np.ones(n, int); j[ok] = self.idx[tuple(nb[ok].T)]
            k = (j >= 0) & (self.atrial == self.atrial[np.maximum(j, 0)]); I.append(np.nonzero(k)[0]); J.append(j[k]); Lng.append(np.full(k.sum(), HV * np.linalg.norm(o)))
        I, J, Lng = map(np.concatenate, (I, J, Lng)); ci, cj = self.cls[I], self.cls[J]
        self.etype = np.where(ci == cj, ci, np.where(self.atrial[I], ATR, MYO))
        # CSR with one extra (virtual) node n that feeds the stimulus sites; order the edge data like the CSR
        A = sp.csr_matrix((np.arange(1, len(I) + 1, dtype=float), (I, J)), shape=(n + 1, n + 1)); A.sort_indices()
        self.perm = A.data.astype(int) - 1; self.A = A; self.Lng = Lng; self.I, self.J = I, J
        self.nbr = self._nbr()

    def _nbr(self):
        """neighbour indices along +/- each heart axis for finite-difference gradients (-1 = none)"""
        out = []
        for a in range(3):
            for s in (1, -1):
                o = np.zeros(3, int); o[a] = s; nb = self.ijk + o; ok = np.all((nb >= 0) & (nb < np.array(self.shape)), 1)
                j = -np.ones(self.n, int); j[ok] = self.idx[tuple(nb[ok].T)]; j[(j >= 0) & (self.atrial != self.atrial[np.maximum(j, 0)])] = -1; out.append(j)
        return np.stack(out)

    def nearest(self, pt, mask):
        k = np.nonzero(mask)[0]; return k[np.argmin(((self.pos[k] - np.asarray(pt)) ** 2).sum(1))]

    def activate(self, sites, t0, cv):
        """sites: voxel indices; t0: their stimulus times (ms); cv: speeds (mm/ms) per edge class [MYO, LVF, RVF, ATR] -> tau (ms)"""
        A = self.A.copy(); w = self.Lng / np.asarray(cv, float)[self.etype]; A.data = w[self.perm]
        V = sp.csr_matrix((np.asarray(t0, float) + 1e-3, (np.full(len(sites), self.n), np.asarray(sites))), shape=A.shape)
        tau = dijkstra(A + V, directed=True, indices=self.n)[:self.n] - 1e-3; tau[~np.isfinite(tau)] = np.nan; return tau

    def grad(self, f):
        """finite-difference gradient (heart frame, per mm) of a voxel field, central where possible, one-sided at walls"""
        g = np.zeros((self.n, 3))
        for a in range(3):
            jp, jm = self.nbr[2 * a], self.nbr[2 * a + 1]; fp = np.where(jp >= 0, f[np.maximum(jp, 0)], np.nan); fm = np.where(jm >= 0, f[np.maximum(jm, 0)], np.nan)
            c = (fp - fm) / (2 * HV); fw = (fp - f) / HV; bw = (f - fm) / HV
            g[:, a] = np.where(np.isfinite(c), c, np.where(np.isfinite(fw), fw, np.where(np.isfinite(bw), bw, 0)))
        g[~np.isfinite(g)] = 0; return g


def kernel(width, n=None):
    """derivative of a logistic step with the given width (ms), sampled at DT, unit area"""
    n = n or int(np.ceil(8 * width / DT)); t = np.arange(-n, n + 1) * DT; s = 1 / (1 + np.exp(-t / width)); k = s * (1 - s) / width; return k / k.sum()


def sweep(times, vec, L, width):
    """sum over voxels of vec_i * k(t - times_i) on a DT grid of length L (linear interpolation into bins, then the kernel)"""
    out = np.zeros((L, 3)); ok = np.isfinite(times) & (times >= 0) & (times < L - 2); t = times[ok] / DT; i0 = np.floor(t).astype(int); fr = t - i0
    for c in range(3):
        out[:, c] = np.bincount(i0, vec[ok, c] * (1 - fr), L) + np.bincount(i0 + 1, vec[ok, c] * fr, L)[:L]
    k = kernel(width); h = len(k) // 2; return np.stack([np.convolve(out[:, c], k)[h:h + L] for c in range(3)], 1)


def dipole(heart, tau, r, mask, L, wd=1.5, wr=15.0):
    """heart dipole H(t), heart frame, arbitrary units, sampled at DT for L steps after time 0"""
    gt = heart.grad(np.where(mask, tau, np.nan)); gr = heart.grad(np.where(mask, r, np.nan)); m = mask
    vol = HV ** 3; return vol * (sweep(tau[m], gt[m], L, wd) - sweep(r[m], gr[m], L, wr))
