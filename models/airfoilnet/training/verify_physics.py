"""Independent checks of the simulator (mfoil_fast). -> work/verify_physics.json

1. Inviscid panel method vs the EXACT Joukowski airfoil (conformal map): lift coefficient and Cp.
2. Inviscid lift vs thin-airfoil theory for a thin cambered NACA 2404 (zero-lift angle, slope).
3. Viscous polars vs published XFOIL 6.99 polars (airfoiltools.com, Ncrit 9): NACA 0012, 2412, 4412, SD7034.
   NeuralFoil (xxxlarge, numpy port) is scored on the same XFOIL polars as a check of that port.
4. NACA 2412 at Re 3e6 vs the classic NACA values quoted in Abbott & von Doenhoff (1959) (approximate).
"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import xsolver as xs  # noqa
import neuralfoil_np as nfn  # noqa
mf = xs.mf
ROOT = os.path.join(HERE, '..')
from _paths import WORK   # models/airfoilnet/work or $ML_RATCHET_WORK/airfoilnet (training/_paths.py)
OUT = {}


def inviscid(Y, alpha, npanel=160):
    M = xs.make_model(Y, npanel)
    M.setoper(alpha=alpha); M.oper.viscous = False
    mf.solve_inviscid(M)
    return M


# ---------------------------------------------------------------- 1. Joukowski
def joukowski(mux=-0.08, muy=0.08, n=400):
    z0 = complex(mux, muy); a = abs(1 - z0)
    th = np.linspace(0, 2 * np.pi, n + 1)
    beta = np.arcsin(muy / a)
    th0 = -beta                       # circle point mapping to the TE (zeta = 1)
    zeta = z0 + a * np.exp(1j * (th0 + th))
    z = zeta + 1 / zeta
    return z, zeta, z0, a, beta


z, zeta, z0, a, beta = joukowski()
X = np.c_[z.real, z.imag]
# Selig order: start at TE, go over the upper surface first
if X[len(X) // 4, 1] < X[3 * len(X) // 4, 1]:
    X = X[::-1]
xmin, xmax = X[:, 0].min(), X[:, 0].max(); c = xmax - xmin
Yj = (X - [xmin, 0]) / c                     # scale only (no rotation): alpha is the freestream angle
res = []
for al in (0.0, 4.0, 8.0):
    M = inviscid(Yj, al)
    cl_ex = 8 * np.pi * a * np.sin(np.radians(al) + beta) / (c)
    # exact Cp on the surface
    zz = z0 + a * np.exp(1j * np.linspace(-beta + 1e-4, -beta + 2 * np.pi - 1e-4, 4000))
    alr = np.radians(al); G = 4 * np.pi * a * np.sin(alr + beta)
    dW = np.exp(-1j * alr) - a ** 2 * np.exp(1j * alr) / (zz - z0) ** 2 + 1j * G / (2 * np.pi * (zz - z0))
    V = np.abs(dW / (1 - 1 / zz ** 2)); cpx = 1 - V ** 2
    zx = zz + 1 / zz; xe = (zx.real - xmin) / c; ye = zx.imag / c
    # compare at panel nodes away from the cusp (x < 0.95)
    Xn = M.foil.x; cpn = np.asarray(M.post.cp)[:Xn.shape[1]]
    errs = []
    for k in range(Xn.shape[1]):
        if Xn[0, k] > 0.95:
            continue
        d = (xe - Xn[0, k]) ** 2 + (ye - Xn[1, k]) ** 2
        errs.append(abs(cpn[k] - cpx[np.argmin(d)]))
    res.append(dict(alpha=al, cl_panel=float(M.post.cl), cl_exact=float(cl_ex),
                    cl_rel_err=float(M.post.cl / cl_ex - 1), cp_mae=float(np.mean(errs)), cp_max=float(np.max(errs))))
OUT['joukowski'] = res
print('joukowski', res, flush=True)


# ---------------------------------------------------------------- 2. thin airfoil theory
def naca4(m, p, t, n=161):
    b = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yt = 5 * t * (0.2969 * np.sqrt(b) - 0.1260 * b - 0.3516 * b ** 2 + 0.2843 * b ** 3 - 0.1036 * b ** 4)
    yc = np.where(b < p, m / p ** 2 * (2 * p * b - b ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * b - b ** 2))
    dy = np.where(b < p, 2 * m / p ** 2 * (p - b), 2 * m / (1 - p) ** 2 * (p - b))
    th = np.arctan(dy)
    xu, yu = b - yt * np.sin(th), yc + yt * np.cos(th); xl, yl = b + yt * np.sin(th), yc - yt * np.cos(th)
    return np.vstack([np.c_[xu, yu][::-1], np.c_[xl, yl][1:]])


m, p, t = 0.02, 0.4, 0.04
th = np.linspace(1e-6, np.pi, 20001); xx = 0.5 * (1 - np.cos(th))
dz = np.where(xx < p, 2 * m / p ** 2 * (p - xx), 2 * m / (1 - p) ** 2 * (p - xx))
aL0 = -np.degrees(np.trapezoid(dz * (np.cos(th) - 1), th) / np.pi)
Y = xs.normalize(naca4(m, p, t))
cls = [inviscid(Y, al).post.cl for al in (-2.0, 0.0, 2.0, 4.0)]
slope = np.polyfit([-2, 0, 2, 4], cls, 1)
OUT['thin_airfoil'] = dict(airfoil='NACA 2404 (closed TE)', aL0_theory_deg=float(aL0),
                           aL0_panel_deg=float(-slope[1] / slope[0]), slope_theory_per_rad=2 * np.pi,
                           slope_panel_per_rad=float(slope[0] * 180 / np.pi),
                           slope_ratio=float(slope[0] * 180 / np.pi / (2 * np.pi)),
                           note='a 4% thick section is expected ~3% above 2*pi (thickness effect ~ 1 + 0.77 t/c)')
print('thin', OUT['thin_airfoil'], flush=True)

# ---------------------------------------------------------------- 3. published XFOIL
nf = nfn.NeuralFoil('xxxlarge')
DB = os.path.join(ROOT, 'data', 'raw', 'coord_seligFmt', 'coord_seligFmt')
cases = [('naca2412', 'naca2412', 1e6), ('naca2412', 'naca2412', 2e5), ('n0012', 'n0012', 1e6),
         ('n0012', 'n0012', 2e5), ('naca4412', 'naca4412', 5e5), ('sd7034', 'sd7034', 2e5), ('sd7034', 'sd7034', 1e5)]
xf_rows = []
for dbname, xfname, Re in cases:
    csv = os.path.join(ROOT, 'data', 'xfoil_published', f'xf-{xfname}-il-{int(Re)}.csv')
    L = open(csv).read().splitlines()
    k = [i for i, l in enumerate(L) if l.startswith('Alpha,')][0]
    A = np.array([[float(v) for v in l.split(',')] for l in L[k + 1:] if l.strip()])
    Y = xs.normalize(xs.read_dat(os.path.join(DB, dbname + '.dat')))
    sw = xs.polar_sweep(Y, Re, alphas_up=np.arange(0, 13, 2.), alphas_dn=np.arange(-2, -5, -2.), npanel=160)
    kp = nfn.kulfan_fit(Y)
    for r in sw:
        j = np.argmin(np.abs(A[:, 0] - r['alpha']))
        if abs(A[j, 0] - r['alpha']) > 0.01:
            continue
        o = nf(kp, r['alpha'], Re)
        xf_rows.append(dict(case=f'{xfname} Re={Re:.0e}', alpha=r['alpha'], xf_cl=A[j, 1], xf_cd=A[j, 2], xf_cm=A[j, 4],
                            mf_cl=r['cl'], mf_cd=r['cd'], mf_cm=r['cm'], loose=r['loose'],
                            nf_cl=float(o['CL'][0]), nf_cd=float(o['CD'][0]), nf_cm=float(o['CM'][0])))
    print(dbname, Re, len(sw), flush=True)
R = {k: np.array([r[k] for r in xf_rows]) for k in ('xf_cl', 'xf_cd', 'xf_cm', 'mf_cl', 'mf_cd', 'mf_cm', 'nf_cl', 'nf_cd', 'nf_cm')}
OUT['vs_xfoil'] = dict(n_points=len(xf_rows), rows=xf_rows,
                       mfoil=dict(cl_mae=float(np.mean(np.abs(R['mf_cl'] - R['xf_cl']))),
                                  cd_median_rel=float(np.median(np.abs(R['mf_cd'] / R['xf_cd'] - 1))),
                                  cm_mae=float(np.mean(np.abs(R['mf_cm'] - R['xf_cm'])))),
                       neuralfoil=dict(cl_mae=float(np.mean(np.abs(R['nf_cl'] - R['xf_cl']))),
                                       cd_median_rel=float(np.median(np.abs(R['nf_cd'] / R['xf_cd'] - 1))),
                                       cm_mae=float(np.mean(np.abs(R['nf_cm'] - R['xf_cm'])))))
print('vs xfoil', {k: v for k, v in OUT['vs_xfoil'].items() if k != 'rows'}, flush=True)

# ---------------------------------------------------------------- 4. NACA 2412, Re 3e6
Y = xs.normalize(xs.read_dat(os.path.join(DB, 'naca2412.dat')))
sw = xs.polar_sweep(Y, 3e6, alphas_up=np.arange(-2, 19, 1.), alphas_dn=np.arange(-3, -6, -1.), npanel=160)
al = np.array([r['alpha'] for r in sw]); cl = np.array([r['cl'] for r in sw]); cd = np.array([r['cd'] for r in sw])
cm = np.array([r['cm'] for r in sw])
lin = (al >= -2) & (al <= 6)
s = np.polyfit(al[lin], cl[lin], 1)
OUT['naca2412_Re3e6'] = dict(aL0_deg=float(-s[1] / s[0]), slope_per_deg=float(s[0]), clmax=float(cl.max()),
                             alpha_clmax=float(al[np.argmax(cl)]), cdmin=float(cd.min()),
                             cm_at_cl0p4=float(np.interp(0.4, cl[lin], cm[lin])),
                             reference='Abbott & von Doenhoff (1959), NACA 2412 smooth, Re ~3e6: aL0 ~ -2.0 deg, '
                                       'slope ~0.105/deg, clmax ~1.6, cdmin ~0.006, cm,c/4 ~ -0.045 (approximate values)')
print('naca2412', OUT['naca2412_Re3e6'], flush=True)
json.dump(OUT, open(os.path.join(WORK, 'verify_physics.json'), 'w'), indent=1, default=float)
