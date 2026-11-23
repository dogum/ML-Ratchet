"""mfoil_fast: numba acceleration of Fidkowski's mfoil.py (MIT), incompressible only.

The physics is unchanged. We replace the hot spots of the vendored mfoil.py with numba kernels that do
the same arithmetic in the same order:
  * panel influence coefficients (build_gamma, inviscid_velocity, build_wake, calc_ue_m)
  * the boundary-layer closure and station residual (residual_station and the get_* closures), Mach 0 only
  * sparse assembly of the global Newton system (COO triplets instead of lil_matrix element writes)
`verify_fast.py` checks every patched function against the original to ~1e-10 on real airfoils.

Usage:  import mfoil_fast; M = mfoil_fast.mf.mfoil(coords=X, npanel=...)   (patches are applied on import)
"""
import os
import numpy as np
from numba import njit
from scipy import sparse
import scipy.sparse.linalg  # noqa: F401
import mfoil as mf

PI = np.pi

# ---------------------------------------------------------------- panel kernels
@njit(cache=True)
def _pinfo(x1, z1, x2, z2, px, pz):
    tx = x2 - x1; tz = z2 - z1
    nrm = np.sqrt(tx * tx + tz * tz)
    tx /= nrm; tz /= nrm
    nx = -tz; nz = tx
    dx = px - x1; dz = pz - z1
    x = dx * tx + dz * tz
    z = dx * nx + dz * nz
    d = np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)
    r1 = np.sqrt(x ** 2 + z ** 2)
    r2 = np.sqrt((x - d) ** 2 + z ** 2)
    th1 = np.arctan2(z, x)
    th2 = np.arctan2(z, x - d)
    return tx, tz, nx, nz, x, z, d, r1, r2, th1, th2


@njit(cache=True)
def _linvortex_vel(x1, z1, x2, z2, px, pz):
    tx, tz, nx, nz, x, z, d, r1, r2, th1, th2 = _pinfo(x1, z1, x2, z2, px, pz)
    temp1 = (th2 - th1) / (2 * PI)
    temp2 = (2 * z * np.log(r1 / r2) - 2 * x * (th2 - th1)) / (4 * PI * d)
    ug1 = temp1 + temp2
    ug2 = -temp2
    temp1 = np.log(r2 / r1) / (2 * PI)
    temp2 = (x * np.log(r1 / r2) - d + z * (th2 - th1)) / (2 * PI * d)
    wg1 = temp1 + temp2
    wg2 = -temp2
    return (ug1 * tx + wg1 * nx, ug1 * tz + wg1 * nz, ug2 * tx + wg2 * nx, ug2 * tz + wg2 * nz)


@njit(cache=True)
def _linvortex_stream(x1, z1, x2, z2, px, pz):
    tx, tz, nx, nz, x, z, d, r1, r2, th1, th2 = _pinfo(x1, z1, x2, z2, px, pz)
    ep = 1e-9
    logr1 = 0. if (r1 < ep) else np.log(r1)
    logr2 = 0. if (r2 < ep) else np.log(r2)
    P1 = (0.5 / PI) * (z * (th2 - th1) - d + x * logr1 - (x - d) * logr2)
    P2 = x * P1 + (0.5 / PI) * (0.5 * r2 ** 2 * logr2 - 0.5 * r1 ** 2 * logr1 - r2 ** 2 / 4 + r1 ** 2 / 4)
    return P1 - P2 / d, P2 / d


@njit(cache=True)
def _constsource_vel(x1, z1, x2, z2, px, pz):
    tx, tz, nx, nz, x, z, d, r1, r2, th1, th2 = _pinfo(x1, z1, x2, z2, px, pz)
    ep = 1e-9
    if r1 < ep:
        logr1 = 0.; th1 = PI; th2 = PI
    else:
        logr1 = np.log(r1)
    if r2 < ep:
        logr2 = 0.; th1 = 0.; th2 = 0.
    else:
        logr2 = np.log(r2)
    u = (0.5 / PI) * (logr1 - logr2)
    w = (0.5 / PI) * (th2 - th1)
    return u * tx + w * nx, u * tz + w * nz


@njit(cache=True)
def _constsource_stream(x1, z1, x2, z2, px, pz):
    tx, tz, nx, nz, x, z, d, r1, r2, th1, th2 = _pinfo(x1, z1, x2, z2, px, pz)
    ep = 1e-9
    if r1 < ep:
        logr1 = 0.; th1 = PI; th2 = PI
    else:
        logr1 = np.log(r1)
    if r2 < ep:
        logr2 = 0.; th1 = 0.; th2 = 0.
    else:
        logr2 = np.log(r2)
    P = (x * (th1 - th2) + d * th2 + z * logr1 - z * logr2) / (2 * PI)
    dP = d
    if (th1 + th2) > PI:
        P = P - 0.25 * dP
    else:
        P = P + 0.75 * dP
    return P


@njit(cache=True)
def _linsource_vel(x1, z1, x2, z2, px, pz):
    tx, tz, nx, nz, x, z, d, r1, r2, th1, th2 = _pinfo(x1, z1, x2, z2, px, pz)
    temp1 = np.log(r1 / r2) / (2 * PI)
    temp2 = (x * np.log(r1 / r2) - d + z * (th2 - th1)) / (2 * PI * d)
    ug1 = temp1 - temp2
    ug2 = temp2
    temp1 = (th2 - th1) / (2 * PI)
    temp2 = (-z * np.log(r1 / r2) + x * (th2 - th1)) / (2 * PI * d)
    wg1 = temp1 - temp2
    wg2 = temp2
    return (ug1 * tx + wg1 * nx, ug1 * tz + wg1 * nz, ug2 * tx + wg2 * nx, ug2 * tz + wg2 * nz)


@njit(cache=True)
def _linsource_stream(x1, z1, x2, z2, px, pz):
    tx, tz, nx, nz, x, z, d, r1, r2, th1, th2 = _pinfo(x1, z1, x2, z2, px, pz)
    if th1 < 0: th1 = th1 + 2 * PI
    if th2 < 0: th2 = th2 + 2 * PI
    ep = 1e-9
    if r1 < ep:
        logr1 = 0.; th1 = PI; th2 = PI
    else:
        logr1 = np.log(r1)
    if r2 < ep:
        logr2 = 0.; th1 = 0.; th2 = 0.
    else:
        logr2 = np.log(r2)
    P1 = (0.5 / PI) * (x * (th1 - th2) + th2 * d + z * logr1 - z * logr2)
    P2 = x * P1 + (0.5 / PI) * (0.5 * r2 ** 2 * th2 - 0.5 * r1 ** 2 * th1 - 0.5 * z * d)
    return P1 - P2 / d, P2 / d


@njit(cache=True)
def _build_A(X, tcp, tdp):
    N = X.shape[1]
    A = np.zeros((N + 1, N + 1))
    for i in range(N):
        px = X[0, i]; pz = X[1, i]
        for j in range(N - 1):
            a, b = _linvortex_stream(X[0, j], X[1, j], X[0, j + 1], X[1, j + 1], px, pz)
            A[i, j] += a
            A[i, j + 1] += b
        A[i, N] = -1
        if tcp == 0. and tdp == 0.:   # sharp TE: zero-length TE panel carries nothing
            continue
        a = _constsource_stream(X[0, N - 1], X[1, N - 1], X[0, 0], X[1, 0], px, pz)
        A[i, 0] += -a * (0.5 * tcp)
        A[i, N - 1] += a * (0.5 * tcp)
        a, b = _linvortex_stream(X[0, N - 1], X[1, N - 1], X[0, 0], X[1, 0], px, pz)
        A[i, 0] += -(a + b) * (-0.5 * tdp)
        A[i, N - 1] += (a + b) * (-0.5 * tdp)
    return A


@njit(cache=True)
def _invvel(X, G, tcp, tdp, px, pz, VG):
    """velocity at (px,pz) from airfoil vortex panels + TE panels (no freestream); fills VG (2xN)"""
    N = X.shape[1]
    V0 = 0.; V1 = 0.
    for k in range(N):
        VG[0, k] = 0.; VG[1, k] = 0.
    for j in range(N - 1):
        a0, a1, b0, b1 = _linvortex_vel(X[0, j], X[1, j], X[0, j + 1], X[1, j + 1], px, pz)
        V0 += a0 * G[j] + b0 * G[j + 1]
        V1 += a1 * G[j] + b1 * G[j + 1]
        VG[0, j] += a0; VG[1, j] += a1
        VG[0, j + 1] += b0; VG[1, j + 1] += b1
    if tcp == 0. and tdp == 0.:   # sharp TE
        return V0, V1
    a0, a1 = _constsource_vel(X[0, N - 1], X[1, N - 1], X[0, 0], X[1, 0], px, pz)
    f10 = a0 * (-0.5 * tcp); f11 = a1 * (-0.5 * tcp)
    f20 = a0 * 0.5 * tcp; f21 = a1 * 0.5 * tcp
    V0 += f10 * G[0] + f20 * G[N - 1]
    V1 += f11 * G[0] + f21 * G[N - 1]
    VG[0, 0] += f10; VG[1, 0] += f11
    VG[0, N - 1] += f20; VG[1, N - 1] += f21
    a0, a1, b0, b1 = _linvortex_vel(X[0, N - 1], X[1, N - 1], X[0, 0], X[1, 0], px, pz)
    f10 = (a0 + b0) * (0.5 * tdp); f11 = (a1 + b1) * (0.5 * tdp)
    f20 = (a0 + b0) * (-0.5 * tdp); f21 = (a1 + b1) * (-0.5 * tdp)
    V0 += f10 * G[0] + f20 * G[N - 1]
    V1 += f11 * G[0] + f21 * G[N - 1]
    VG[0, 0] += f10; VG[1, 0] += f11
    VG[0, N - 1] += f20; VG[1, N - 1] += f21
    return V0, V1


@njit(cache=True)
def _ue_m_mats(Xf, Xw, Tw, G, tcp, tdp):
    N = Xf.shape[1]; Nw = Xw.shape[1]
    Cgam = np.zeros((Nw, N))
    VG = np.zeros((2, N))
    for i in range(Nw):
        _invvel(Xf, G, tcp, tdp, Xw[0, i], Xw[1, i], VG)
        for k in range(N):
            Cgam[i, k] = VG[0, k] * Tw[0, i] + VG[1, k] * Tw[1, i]
    B = np.zeros((N + 1, N + Nw - 2))
    Xj = np.zeros((2, 3))
    for i in range(N):
        px = Xf[0, i]; pz = Xf[1, i]
        for j in range(N - 1):
            B[i, j] = _constsource_stream(Xf[0, j], Xf[1, j], Xf[0, j + 1], Xf[1, j + 1], px, pz)
        for j in range(Nw - 1):
            Xj[0, 0] = Xw[0, j]; Xj[1, 0] = Xw[1, j]
            Xj[0, 2] = Xw[0, j + 1]; Xj[1, 2] = Xw[1, j + 1]
            Xj[0, 1] = 0.5 * (Xw[0, j] + Xw[0, j + 1]); Xj[1, 1] = 0.5 * (Xw[1, j] + Xw[1, j + 1])
            if j == Nw - 2:
                Xj[0, 2] = 2 * Xj[0, 2] - Xj[0, 1]; Xj[1, 2] = 2 * Xj[1, 2] - Xj[1, 1]
            a, b = _linsource_stream(Xj[0, 0], Xj[1, 0], Xj[0, 1], Xj[1, 1], px, pz)
            if j > 0:
                B[i, N - 1 + j] += 0.5 * a + b
                B[i, N - 1 + j - 1] += 0.5 * a
            else:
                B[i, N - 1 + j] += b
            a, b = _linsource_stream(Xj[0, 1], Xj[1, 1], Xj[0, 2], Xj[1, 2], px, pz)
            B[i, N - 1 + j] += a + 0.5 * b
            if j < Nw - 2:
                B[i, N - 1 + j + 1] += 0.5 * b
            else:
                B[i, N - 1 + j] += 0.5 * b
    Csig = np.zeros((Nw, N + Nw - 2))
    for i in range(Nw):
        px = Xw[0, i]; pz = Xw[1, i]; t0 = Tw[0, i]; t1 = Tw[1, i]
        jstart = 1 if i == 0 else 0
        jend = N - 2 if i == 0 else N - 1
        for j in range(jstart, jend):
            v0, v1 = _constsource_vel(Xf[0, j], Xf[1, j], Xf[0, j + 1], Xf[1, j + 1], px, pz)
            Csig[i, j] = v0 * t0 + v1 * t1
        for j in range(Nw):
            I0 = max(j - 1, 0); I2 = min(j + 1, Nw - 1)
            Xj[0, 0] = Xw[0, I0]; Xj[1, 0] = Xw[1, I0]
            Xj[0, 1] = Xw[0, j]; Xj[1, 1] = Xw[1, j]
            Xj[0, 2] = Xw[0, I2]; Xj[1, 2] = Xw[1, I2]
            Xj[0, 0] = 0.5 * (Xj[0, 0] + Xj[0, 1]); Xj[1, 0] = 0.5 * (Xj[1, 0] + Xj[1, 1])
            Xj[0, 2] = 0.5 * (Xj[0, 1] + Xj[0, 2]); Xj[1, 2] = 0.5 * (Xj[1, 1] + Xj[1, 2])
            if j == Nw - 1:
                Xj[0, 2] = 2 * Xj[0, 1] - Xj[0, 0]; Xj[1, 2] = 2 * Xj[1, 1] - Xj[1, 0]
            d1 = np.sqrt((Xj[0, 1] - Xj[0, 0]) ** 2 + (Xj[1, 1] - Xj[1, 0]) ** 2)
            d2 = np.sqrt((Xj[0, 2] - Xj[0, 1]) ** 2 + (Xj[1, 2] - Xj[1, 1]) ** 2)
            if i == j:
                if j == 0:
                    dl = np.sqrt((Xf[0, 1] - Xf[0, 0]) ** 2 + (Xf[1, 1] - Xf[1, 0]) ** 2)
                    du = np.sqrt((Xf[0, N - 1] - Xf[0, N - 2]) ** 2 + (Xf[1, N - 1] - Xf[1, N - 2]) ** 2)
                    Csig[i, 0] += (0.5 / PI) * (np.log(dl / d2) + 1)
                    Csig[i, N - 2] += (0.5 / PI) * (np.log(du / d2) + 1)
                    Csig[i, N - 1] += -0.5 / PI
                elif j == Nw - 1:
                    Csig[i, N - 1 + j - 1] += 0
                else:
                    aa = (0.25 / PI) * np.log(d1 / d2)
                    Csig[i, N - 1 + j - 1] += aa + 0.5 / PI
                    Csig[i, N - 1 + j] += aa - 0.5 / PI
            else:
                if j == 0:
                    a0, a1, b0, b1 = _linsource_vel(Xj[0, 1], Xj[1, 1], Xj[0, 2], Xj[1, 2], px, pz)
                    a = a0 * t0 + a1 * t1; b = b0 * t0 + b1 * t1
                    Csig[i, N - 1] += b
                    Csig[i, 0] += a
                    Csig[i, N - 2] += a
                elif j == Nw - 1:
                    v0, v1 = _constsource_vel(Xj[0, 0], Xj[1, 0], Xj[0, 2], Xj[1, 2], px, pz)
                    Csig[i, N + Nw - 3] += v0 * t0 + v1 * t1
                else:
                    a0, a1, b0, b1 = _linsource_vel(Xj[0, 0], Xj[1, 0], Xj[0, 1], Xj[1, 1], px, pz)
                    a1_ = a0 * t0 + a1 * t1; b1_ = b0 * t0 + b1 * t1
                    a0, a1, b0, b1 = _linsource_vel(Xj[0, 1], Xj[1, 1], Xj[0, 2], Xj[1, 2], px, pz)
                    a2_ = a0 * t0 + a1 * t1; b2_ = b0 * t0 + b1 * t1
                    Csig[i, N - 1 + j - 1] += a1_ + 0.5 * b1_
                    Csig[i, N - 1 + j] += 0.5 * a2_ + b2_
    return Cgam, B, Csig


# ------------------------------------------------------------- patched inviscid functions
def build_gamma(M, alpha):
    N = M.foil.N
    t, hTE, dtdx, tcp, tdp = mf.TE_info(M.foil.x)
    nogap = (abs(hTE) < 1e-10 * M.geom.chord)
    X = np.ascontiguousarray(M.foil.x, dtype=np.float64)
    A = _build_A(X, tcp, tdp)
    rhs = np.zeros([N + 1, 2])
    rhs[:N, 0] = -X[1, :]
    rhs[:N, 1] = X[0, :]
    if nogap:
        A[N - 1, :] = 0
        A[N - 1, [0, 1, 2, N - 3, N - 2, N - 1]] = [1, -2, 1, -1, 2, -1]
    A[N, 0] = 1
    A[N, N - 1] = 1
    M.isol.AIC = A
    g = np.linalg.solve(M.isol.AIC, rhs)
    M.isol.gamref = g[0:N, :]
    M.isol.gam = M.isol.gamref[:, 0] * mf.cosd(alpha) + M.isol.gamref[:, 1] * mf.sind(alpha)


def inviscid_velocity(X, G, Vinf, alpha, x, dolin):
    N = X.shape[1]
    t, hTE, dtdx, tcp, tdp = mf.TE_info(X)
    VG = np.zeros((2, N))
    V0, V1 = _invvel(np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(G, dtype=np.float64),
                     tcp, tdp, float(x[0]), float(x[1]), VG)
    V = np.array([V0, V1]) + Vinf * np.array([mf.cosd(alpha), mf.sind(alpha)])
    if dolin:
        return V, VG
    return V


def calc_ue_m(M):
    assert len(M.isol.gam) > 0, 'No inviscid solution'
    N, Nw = M.foil.N, M.wake.N
    t, hTE, dtdx, tcp, tdp = mf.TE_info(M.foil.x)
    Cgam, B, Csig = _ue_m_mats(np.ascontiguousarray(M.foil.x, dtype=np.float64),
                               np.ascontiguousarray(M.wake.x, dtype=np.float64),
                               np.ascontiguousarray(M.wake.t, dtype=np.float64),
                               np.ascontiguousarray(M.isol.gam, dtype=np.float64), tcp, tdp)
    Bp = -np.linalg.solve(M.isol.AIC, B)
    Bp = Bp[:-1, :]
    Dw = np.dot(Cgam, Bp) + Csig
    Dw[0, :] = Bp[-1, :]
    M.vsol.ue_sigma = np.vstack((Bp, Dw))
    mf.rebuild_ue_m(M)


# ------------------------------------------------------------- BL closures (Mach 0)
# parameter vector layout
NCRIT, CUQ, DLR, SLAGK, CTAUC, CTAUE, GA, GB, GC, RHO0, MU0, VINF, SIMI, TURB, WAKE = range(15)


def pack(param):
    assert param.Minf == 0, 'mfoil_fast supports Mach 0 only'
    return np.array([param.ncrit, param.Cuq, param.Dlr, param.SlagK, param.CtauC, param.CtauE,
                     param.GA, param.GB, param.GC, param.rho0, param.mu0, param.Vinf,
                     1.0 if param.simi else 0.0, 1.0 if param.turb else 0.0, 1.0 if param.wake else 0.0])


@njit(cache=True)
def _H(U):
    H = U[1] / U[0]
    g = np.zeros(4); g[0] = -H / U[0]; g[1] = 1 / U[0]
    return H, g


@njit(cache=True)
def _Ret(U, P):
    R = P[RHO0] * U[0] * U[3] / P[MU0]
    g = np.zeros(4); g[0] = U[3] / P[MU0]; g[3] = U[0] / P[MU0]
    return R, g


@njit(cache=True)
def _Hw(U, wgap):
    Hw = wgap / U[0]
    g = np.zeros(4); g[0] = -Hw / U[0]
    return Hw, g


@njit(cache=True)
def _Hs(U, P):
    wake = P[WAKE] > 0.5; turb = P[TURB] > 0.5
    Hk, Hk_U = _H(U)
    if wake and Hk < 1.00005:
        Hk = 1.00005; Hk_U = np.zeros(4)
    if (not wake) and Hk < 1.05:
        Hk = 1.05; Hk_U = np.zeros(4)
    if turb:
        Hsmin = 1.5; dHsinf = .015
        Ret, Ret_U = _Ret(U, P)
        Ho = 4.; Ho_U = np.zeros(4)
        if Ret > 400:
            Ho = 3 + 400. / Ret; Ho_U = -400. / Ret ** 2 * Ret_U
        Reb = Ret; Reb_U = Ret_U.copy()
        if Ret < 200:
            Reb = 200.; Reb_U = np.zeros(4)
        if Hk < Ho:
            Hr = (Ho - Hk) / (Ho - 1)
            Hr_U = (Ho_U - Hk_U) / (Ho - 1) - (Ho - Hk) / (Ho - 1) ** 2 * Ho_U
            aa = (2 - Hsmin - 4 / Reb) * Hr ** 2
            aa_U = (4 / Reb ** 2 * Reb_U) * Hr ** 2 + (2 - Hsmin - 4 / Reb) * 2 * Hr * Hr_U
            Hs = Hsmin + 4 / Reb + aa * 1.5 / (Hk + .5)
            Hs_U = -4 / Reb ** 2 * Reb_U + aa_U * 1.5 / (Hk + .5) - aa * 1.5 / (Hk + .5) ** 2 * Hk_U
        else:
            lrb = np.log(Reb); lrb_U = 1 / Reb * Reb_U
            aa = Hk - Ho + 4 / lrb
            aa_U = Hk_U - Ho_U - 4 / lrb ** 2 * lrb_U
            bb = .007 * lrb / aa ** 2 + dHsinf / Hk
            bb_U = .007 * (lrb_U / aa ** 2 - 2 * lrb / aa ** 3 * aa_U) - dHsinf / Hk ** 2 * Hk_U
            Hs = Hsmin + 4 / Reb + (Hk - Ho) ** 2 * bb
            Hs_U = -4 / Reb ** 2 * Reb_U + 2 * (Hk - Ho) * (Hk_U - Ho_U) * bb + (Hk - Ho) ** 2 * bb_U
        # Mach 0: den = 1, no change
    else:
        a = Hk - 4.35
        if Hk < 4.35:
            num = .0111 * a ** 2 - .0278 * a ** 3
            Hs = num / (Hk + 1) + 1.528 - .0002 * (a * Hk) ** 2
            Hs_Hk = (.0111 * 2 * a - .0278 * 3 * a ** 2) / (Hk + 1) - num / (Hk + 1) ** 2 - .0002 * 2 * a * Hk * (Hk + a)
        else:
            Hs = .015 * a ** 2 / Hk + 1.528
            Hs_Hk = .015 * 2 * a / Hk - .015 * a ** 2 / Hk ** 2
        Hs_U = Hs_Hk * Hk_U
    return Hs, Hs_U


@njit(cache=True)
def _Us(U, P):
    wake = P[WAKE] > 0.5
    Hs, Hs_U = _Hs(U, P)
    Hk, Hk_U = _H(U)
    H, H_U = _H(U)
    if wake and Hk < 1.00005:
        Hk = 1.00005; Hk_U = np.zeros(4)
    if (not wake) and Hk < 1.05:
        Hk = 1.05; Hk_U = np.zeros(4)
    beta = P[GB]; bi = 1. / beta
    Us = 0.5 * Hs * (1 - bi * (Hk - 1) / H)
    Us_U = 0.5 * Hs_U * (1 - bi * (Hk - 1) / H) + 0.5 * Hs * (-bi * (Hk_U) / H + bi * (Hk - 1) / H ** 2 * H_U)
    if (not wake) and (Us > 0.95):
        Us = 0.98; Us_U = np.zeros(4)
    if (not wake) and (Us > 0.99995):
        Us = 0.99995; Us_U = np.zeros(4)
    return Us, Us_U


@njit(cache=True)
def _cf(U, P):
    if P[WAKE] > 0.5:
        return 0., np.zeros(4)
    Hk, Hk_U = _H(U)
    Ret, Ret_U = _Ret(U, P)
    if P[TURB] > 0.5:
        aa = -1.33 * Hk; aa_U = -1.33 * Hk_U
        if aa < -17:
            aa = -20 + 3 * np.exp((aa + 17) / 3)
            aa_U = (aa + 20) / 3 * aa_U
        bb = np.log(Ret / 1.0); bb_U = Ret_U / Ret
        if bb < 3:
            bb = 3.; bb_U = np.zeros(4)
        bb /= np.log(10); bb_U = bb_U / np.log(10)
        cc = -1.74 - 0.31 * Hk; cc_U = -0.31 * Hk_U
        dd = np.tanh(4.0 - Hk / 0.875); dd_U = (1 - dd ** 2) * (-Hk_U / 0.875)
        cf0 = 0.3 * np.exp(aa) * bb ** cc
        cf0_U = cf0 * aa_U + 0.3 * np.exp(aa) * cc * bb ** (cc - 1) * bb_U + cf0 * np.log(bb) * cc_U
        cf = (cf0 + 1.1e-4 * (dd - 1)) / 1.0
        cf_U = (cf0_U + 1.1e-4 * dd_U) / 1.0
    else:
        if Hk < 5.5:
            num = .0727 * (5.5 - Hk) ** 3 / (Hk + 1) - .07
            num_Hk = .0727 * (3 * (5.5 - Hk) ** 2 / (Hk + 1) * (-1) - (5.5 - Hk) ** 3 / (Hk + 1) ** 2)
        else:
            num = .015 * (1 - 1. / (Hk - 4.5)) ** 2 - .07
            num_Hk = .015 * 2 * (1 - 1. / (Hk - 4.5)) / (Hk - 4.5) ** 2
        cf = num / Ret
        cf_U = num_Hk / Ret * Hk_U - num / Ret ** 2 * Ret_U
    return cf, cf_U


@njit(cache=True)
def _cfxt(U, x, P):
    cf, cf_U = _cf(U, P)
    cfxt = cf * x / U[0]
    cfxt_U = cf_U * x / U[0]
    cfxt_U[0] = cfxt_U[0] - cfxt / U[0]
    return cfxt, cfxt_U, cf / U[0]


@njit(cache=True)
def _de(U, P):
    Hk, Hk_U = _H(U)
    aa = 3.15 + 1.72 / (Hk - 1); aa_U = -1.72 / (Hk - 1) ** 2 * Hk_U
    de = U[0] * aa + U[1]
    de_U = U[0] * aa_U
    de_U[0] += aa; de_U[1] += 1
    dmx = 12.0
    if de > dmx * U[0]:
        de = dmx * U[0]; de_U = np.zeros(4); de_U[0] = dmx
    return de, de_U


@njit(cache=True)
def _cteq(U, P):
    wake = P[WAKE] > 0.5
    CC = 0.5 / (P[GA] ** 2 * P[GB]); C = P[GC]
    Hk, Hk_U = _H(U)
    Hs, Hs_U = _Hs(U, P)
    H, H_U = _H(U)
    Ret, Ret_U = _Ret(U, P)
    Us, Us_U = _Us(U, P)
    if wake:
        if Hk < 1.00005:
            Hk = 1.00005; Hk_U = np.zeros(4)
        Hkc = Hk - 1.; Hkc_U = Hk_U
    else:
        if Hk < 1.05:
            Hk = 1.05   # (mfoil keeps Hk_U here: "HK_U" typo in the original, reproduced)
        Hkc = Hk - 1. - C / Ret
        Hkc_U = Hk_U + C / Ret ** 2 * Ret_U
        if Hkc < 0.01:
            Hkc = 0.01; Hkc_U = np.zeros(4)
    num = CC * Hs * (Hk - 1) * Hkc ** 2
    num_U = CC * (Hs_U * (Hk - 1) * Hkc ** 2 + Hs * Hk_U * Hkc ** 2 + Hs * (Hk - 1) * 2 * Hkc * Hkc_U)
    den = (1 - Us) * H * Hk ** 2
    den_U = (-Us_U) * H * Hk ** 2 + (1 - Us) * H_U * Hk ** 2 + (1 - Us) * H * 2 * Hk * Hk_U
    cteq = np.sqrt(num / den)
    cteq_U = 0.5 / cteq * (num_U / den - num / den ** 2 * den_U)
    return cteq, cteq_U


@njit(cache=True)
def _uq(ds, ds_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, P):
    wake = P[WAKE] > 0.5
    beta = P[GB]; A = P[GA]; C = P[GC]
    if wake:
        A = A * P[DLR]; C = 0.
    if wake and Hk < 1.00005:
        Hk = 1.00005; Hk_U = Hk_U * 0.
    if (not wake) and Hk < 1.05:
        Hk = 1.05; Hk_U = Hk_U * 0.
    Hkc = Hk - 1. - C / Ret
    Hkc_U = Hk_U + C / Ret ** 2 * Ret_U
    if Hkc < .01:
        Hkc = .01; Hkc_U = Hkc_U * 0.
    ut = 0.5 * cf - (Hkc / (A * Hk)) ** 2
    ut_U = 0.5 * cf_U - 2 * (Hkc / (A * Hk)) * (Hkc_U / (A * Hk) - Hkc / (A * Hk ** 2) * Hk_U)
    uq = ut / (beta * ds)
    uq_U = ut_U / (beta * ds) - uq / ds * ds_U
    return uq, uq_U


@njit(cache=True)
def _cDi_lam(U, P):
    Hk, Hk_U = _H(U)
    Ret, Ret_U = _Ret(U, P)
    if Hk < 4:
        num = .00205 * (4 - Hk) ** 5.5 + .207
        num_Hk = .00205 * 5.5 * (4 - Hk) ** 4.5 * (-1)
    else:
        Hk1 = Hk - 4
        num = -.0016 * Hk1 ** 2 / (1 + .02 * Hk1 ** 2) + .207
        num_Hk = -.0016 * (2 * Hk1 / (1 + .02 * Hk1 ** 2) - Hk1 ** 2 / (1 + .02 * Hk1 ** 2) ** 2 * .02 * 2 * Hk1)
    cDi = num / Ret
    cDi_U = num_Hk / Ret * Hk_U - num / Ret ** 2 * Ret_U
    return cDi, cDi_U


@njit(cache=True)
def _cDi_turbwall(U, P):
    if P[WAKE] > 0.5:
        return 0., np.zeros(4)
    cf, cf_U = _cf(U, P)
    Hk, Hk_U = _H(U)
    Hs, Hs_U = _Hs(U, P)
    Us, Us_U = _Us(U, P)
    Ret, Ret_U = _Ret(U, P)
    lr = np.log(Ret); lr_U = Ret_U / Ret
    Hmin = 1 + 2.1 / lr; Hmin_U = -2.1 / lr ** 2 * lr_U
    aa = np.tanh((Hk - 1) / (Hmin - 1)); fac = 0.5 + 0.5 * aa
    fac_U = 0.5 * (1 - aa ** 2) * (Hk_U / (Hmin - 1) - (Hk - 1) / (Hmin - 1) ** 2 * Hmin_U)
    cDi = 0.5 * cf * Us * (2 / Hs) * fac
    cDi_U = cf_U * Us / Hs * fac + cf * Us_U / Hs * fac - cDi / Hs * Hs_U + cf * Us / Hs * fac_U
    return cDi, cDi_U


@njit(cache=True)
def _cDi_lamwake(U, Pin):
    P = Pin.copy(); P[TURB] = 0.
    Hk, Hk_U = _H(U)
    Hs, Hs_U = _Hs(U, P)
    Ret, Ret_U = _Ret(U, P)
    HsRet = Hs * Ret
    HsRet_U = Hs_U * Ret + Hs * Ret_U
    num = 2 * 1.1 * (1 - 1 / Hk) ** 2 * (1 / Hk)
    num_Hk = 2 * 1.1 * (2 * (1 - 1 / Hk) * (1 / Hk ** 2) * (1 / Hk) + (1 - 1 / Hk) ** 2 * (-1 / Hk ** 2))
    cDi = num / HsRet
    cDi_U = num_Hk * Hk_U / HsRet - num / HsRet ** 2 * HsRet_U
    return cDi, cDi_U


@njit(cache=True)
def _cDi_outer(U, P):
    if not (P[TURB] > 0.5):
        return 0., np.zeros(4)
    Hs, Hs_U = _Hs(U, P)
    Us, Us_U = _Us(U, P)
    ct = U[2] ** 2; ct_U = np.zeros(4); ct_U[2] = 2 * U[2]
    cDi = ct * (0.995 - Us) * 2 / Hs
    cDi_U = ct_U * (0.995 - Us) * 2 / Hs + ct * (-Us_U) * 2 / Hs - ct * (0.995 - Us) * 2 / Hs ** 2 * Hs_U
    return cDi, cDi_U


@njit(cache=True)
def _cDi_lamstress(U, P):
    Hs, Hs_U = _Hs(U, P)
    Us, Us_U = _Us(U, P)
    Ret, Ret_U = _Ret(U, P)
    HsRet = Hs * Ret
    HsRet_U = Hs_U * Ret + Hs * Ret_U
    num = 0.15 * (0.995 - Us) ** 2 * 2
    num_Us = 0.15 * 2 * (0.995 - Us) * (-1) * 2
    cDi = num / HsRet
    cDi_U = num_Us * Us_U / HsRet - num / HsRet ** 2 * HsRet_U
    return cDi, cDi_U


@njit(cache=True)
def _cDi(U, P):
    wake = P[WAKE] > 0.5
    if P[TURB] > 0.5:
        cDi = 0.; cDi_U = np.zeros(4)
        if not wake:
            c0, c0_U = _cDi_turbwall(U, P)
            cDi = cDi + c0; cDi_U = cDi_U + c0_U
            cDil, cDil_U = _cDi_lam(U, P)
        else:
            cDil, cDil_U = _cDi_lamwake(U, P)
        c0, c0_U = _cDi_outer(U, P)
        cDi = cDi + c0; cDi_U = cDi_U + c0_U
        c0, c0_U = _cDi_lamstress(U, P)
        cDi = cDi + c0; cDi_U = cDi_U + c0_U
        if cDil > cDi:
            cDi = cDil; cDi_U = cDil_U
        if wake:
            cDi = 2 * cDi; cDi_U = 2 * cDi_U
    else:
        cDi, cDi_U = _cDi_lam(U, P)
    return cDi, cDi_U


@njit(cache=True)
def _cDixt(U, x, P):
    cDi, cDi_U = _cDi(U, P)
    cDixt = cDi * x / U[0]
    cDixt_U = cDi_U * x / U[0]
    cDixt_U[0] = cDixt_U[0] - cDixt / U[0]
    return cDixt, cDixt_U, cDi / U[0]


@njit(cache=True)
def _damp(U, P):
    Hk, Hk_U = _H(U)
    Ret, Ret_U = _Ret(U, P)
    th = U[0]
    if Hk < 1.05:
        Hk = 1.05; Hk_U = np.zeros(4)
    Hmi = 1. / (Hk - 1); Hmi_U = -Hmi ** 2 * Hk_U
    aa = 2.492 * Hmi ** 0.43; aa_U = 0.43 * aa / Hmi * Hmi_U
    bb = np.tanh(14 * Hmi - 9.24); bb_U = (1 - bb ** 2) * 14 * Hmi_U
    lrc = aa + 0.7 * (bb + 1); lrc_U = aa_U + 0.7 * bb_U
    lten = np.log(10); lr = np.log(Ret) / lten; lr_U = (1 / Ret) * Ret_U / lten
    dl = .1
    damp = 0.; damp_U = np.zeros(4)
    if lr >= lrc - dl:
        rn = (lr - (lrc - dl)) / (2 * dl); rn_U = (lr_U - lrc_U) / (2 * dl)
        if rn >= 1:
            rf = 1.; rf_U = np.zeros(4)
        else:
            rf = 3 * rn ** 2 - 2 * rn ** 3; rf_U = (6 * rn - 6 * rn ** 2) * rn_U
        ar = 3.87 * Hmi - 2.52; ar_U = 3.87 * Hmi_U
        ex = np.exp(-ar ** 2); ex_U = ex * (-2 * ar * ar_U)
        da = 0.028 * (Hk - 1) - 0.0345 * ex; da_U = 0.028 * Hk_U - 0.0345 * ex_U
        af = -0.05 + 2.7 * Hmi - 5.5 * Hmi ** 2 + 3 * Hmi ** 3 + 0.1 * np.exp(-20 * Hmi)
        af_U = (2.7 - 11 * Hmi + 9 * Hmi ** 2 - 1 * np.exp(-20 * Hmi)) * Hmi_U
        damp = rf * af * da / th
        damp_U = (rf_U * af * da + rf * af_U * da + rf * af * da_U) / th
        damp_U[0] -= damp / th
    ncrit = P[NCRIT]
    Cea = 5; nx = Cea * (U[2] - ncrit)
    nx_U = np.zeros(4); nx_U[2] = Cea
    eex = 1 + np.tanh(nx); eex_U = (1 - np.tanh(nx) ** 2) * nx_U
    ed = eex * .001 / th
    ed_U = eex_U * .001 / th
    ed_U[0] -= ed / th
    damp = damp + ed
    damp_U = damp_U + ed_U
    return damp, damp_U


@njit(cache=True)
def _upwind(upw, upw_U, f1, f1_U1, f2, f2_U2):
    f = (1 - upw) * f1 + upw * f2
    f_U = (-upw_U) * f1 + upw_U * f2
    f_U[:4] += (1 - upw) * f1_U1
    f_U[4:] += upw * f2_U2
    return f, f_U


@njit(cache=True)
def _upw(U1, U2, P):
    Hk1, Hk1_U1 = _H(U1)
    Hk2, Hk2_U2 = _H(U2)
    Hut = 1.
    C = 1. if P[WAKE] > 0.5 else 5.
    Huc = C * Hut / Hk2 ** 2
    Huc_U = np.zeros(8); Huc_U[4:] = -2 * Huc / Hk2 * Hk2_U2
    aa = (Hk2 - 1.) / (Hk1 - 1.); sga = np.sign(aa)
    la = np.log(sga * aa)
    la_U = np.zeros(8); la_U[:4] = -1. / (Hk1 - 1.) * Hk1_U1; la_U[4:] = 1. / (Hk2 - 1.) * Hk2_U2
    Hls = la ** 2; Hls_U = 2 * la * la_U
    if Hls > 15:
        Hls = 15.; Hls_U = Hls_U * 0.
    upw = 1. - 0.5 * np.exp(-Hls * Huc)
    upw_U = -0.5 * np.exp(-Hls * Huc) * (-Hls_U * Huc - Hls * Huc_U)
    return upw, upw_U


@njit(cache=True)
def _residual_station(P, x, Uin, Aux):
    U = Uin.copy()
    U[1, :] -= Aux
    U1 = U[:, 0].copy(); U2 = U[:, 1].copy(); Um = 0.5 * (U1 + U2)
    th = U[0, :]; ds = U[1, :]; sa = U[2, :]
    simi = P[SIMI] > 0.5; turb = P[TURB] > 0.5; wake = P[WAKE] > 0.5
    Z8 = np.zeros(8)

    uk1 = U1[3]; uk1_u = 1.
    uk2 = U2[3]; uk2_u = 1.
    thlog = np.log(th[1] / th[0])
    thlog_U = np.zeros(8); thlog_U[0] = -1. / th[0]; thlog_U[4] = 1. / th[1]
    uelog = np.log(uk2 / uk1)
    uelog_U = np.zeros(8); uelog_U[3] = -uk1_u / uk1; uelog_U[7] = uk2_u / uk2
    xlog = np.log(x[1] / x[0]); xlog_x = np.array([-1. / x[0], 1. / x[1]])
    dx = x[1] - x[0]; dx_x = np.array([-1., 1.])

    upw, upw_U = _upw(U1, U2, P)

    H1, H1_U1 = _H(U1)
    H2, H2_U2 = _H(U2)
    H = 0.5 * (H1 + H2)
    H_U = np.zeros(8); H_U[:4] = 0.5 * H1_U1; H_U[4:] = 0.5 * H2_U2

    Hs1, Hs1_U1 = _Hs(U1, P)
    Hs2, Hs2_U2 = _Hs(U2, P)
    Hs, Hs_U = _upwind(0.5, Z8, Hs1, Hs1_U1, Hs2, Hs2_U2)

    Hslog = np.log(Hs2 / Hs1)
    Hslog_U = np.zeros(8); Hslog_U[:4] = -1. / Hs1 * Hs1_U1; Hslog_U[4:] = 1. / Hs2 * Hs2_U2

    if simi:
        thlog = 0.; thlog_U = thlog_U * 0.
        Hslog = 0.; Hslog_U = Hslog_U * 0.
        uelog = 1.; uelog_U = uelog_U * 0.
        xlog = 1.; xlog_x = np.array([0., 0.])
        dx = 0.5 * (x[0] + x[1]); dx_x = np.array([0.5, 0.5])

    Hw1, Hw1_U1 = _Hw(U1, Aux[0])
    Hw2, Hw2_U2 = _Hw(U2, Aux[1])
    Hw = 0.5 * (Hw1 + Hw2)
    Hw_U = np.zeros(8); Hw_U[:4] = 0.5 * Hw1_U1; Hw_U[4:] = 0.5 * Hw2_U2

    if turb:
        salog = np.log(sa[1] / sa[0])
        salog_U = np.zeros(8); salog_U[2] = -1. / sa[0]; salog_U[6] = 1. / sa[1]
        de1, de1_U1 = _de(U1, P)
        de2, de2_U2 = _de(U2, P)
        de, de_U = _upwind(0.5, Z8, de1, de1_U1, de2, de2_U2)
        Us1, Us1_U1 = _Us(U1, P)
        Us2, Us2_U2 = _Us(U2, P)
        Us, Us_U = _upwind(0.5, Z8, Us1, Us1_U1, Us2, Us2_U2)
        Hk1, Hk1_U1 = _H(U1)
        Hk2, Hk2_U2 = _H(U2)
        Hk, Hk_U = _upwind(upw, upw_U, Hk1, Hk1_U1, Hk2, Hk2_U2)
        Ret1, Ret1_U1 = _Ret(U1, P)
        Ret2, Ret2_U2 = _Ret(U2, P)
        Ret, Ret_U = _upwind(0.5, Z8, Ret1, Ret1_U1, Ret2, Ret2_U2)
        cf1, cf1_U1 = _cf(U1, P)
        cf2, cf2_U2 = _cf(U2, P)
        cf, cf_U = _upwind(upw, upw_U, cf1, cf1_U1, cf2, cf2_U2)
        dsa = 0.5 * (ds[0] + ds[1])
        dsa_U = np.zeros(8); dsa_U[1] = 0.5; dsa_U[5] = 0.5
        uq, uq_U = _uq(dsa, dsa_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, P)
        cteq1, cteq1_U1 = _cteq(U1, P)
        cteq2, cteq2_U2 = _cteq(U2, P)
        cteq, cteq_U = _upwind(upw, upw_U, cteq1, cteq1_U1, cteq2, cteq2_U2)
        e3 = np.zeros(4); e3[2] = 1.
        saa, saa_U = _upwind(upw, upw_U, sa[0], e3, sa[1], e3)
        Klag = P[SLAGK]; beta = P[GB]
        Clag = Klag / beta * 1. / (1. + Us)
        Clag_U = -Clag / (1. + Us) * Us_U
        ald = 1.0
        if wake:
            ald = P[DLR]
        Rlag = Clag * (cteq - ald * saa) * dx - 2 * de * salog + 2 * de * (uq * dx - uelog) * P[CUQ]
        Rlag_U = Clag_U * (cteq - ald * saa) * dx + Clag * (cteq_U - ald * saa_U) * dx \
            - 2 * de_U * salog - 2 * de * salog_U \
            + 2 * de_U * (uq * dx - uelog) * P[CUQ] + 2 * de * (uq_U * dx - uelog_U) * P[CUQ]
        Rlag_x = Clag * (cteq - ald * saa) * dx_x + 2 * de * uq * dx_x
    else:
        if simi:
            Rlag = sa[0] + sa[1]
            Rlag_U = np.array([0., 0., 1., 0., 0., 0., 1., 0.])
            Rlag_x = np.array([0., 0.])
        else:
            damp1, damp1_U1 = _damp(U1, P)
            damp2, damp2_U2 = _damp(U2, P)
            damp, damp_U = _upwind(0.5, Z8, damp1, damp1_U1, damp2, damp2_U2)
            Rlag = sa[1] - sa[0] - damp * dx
            Rlag_U = np.array([0., 0., -1., 0., 0., 0., 1., 0.]) - damp_U * dx
            Rlag_x = -damp * dx_x

    # Mach 0: Ms = 0
    cfxt1, cfxt1_U1, cfxt1_x1 = _cfxt(U1, x[0], P)
    cfxt2, cfxt2_U2, cfxt2_x2 = _cfxt(U2, x[1], P)
    cfxtm, cfxtm_Um, cfxtm_xm = _cfxt(Um, 0.5 * (x[0] + x[1]), P)
    cfxt = 0.25 * cfxt1 + 0.5 * cfxtm + 0.25 * cfxt2
    cfxt_U = np.zeros(8); cfxt_U[:4] = 0.25 * (cfxt1_U1 + cfxtm_Um); cfxt_U[4:] = 0.25 * (cfxtm_Um + cfxt2_U2)
    cfxt_x = 0.25 * np.array([cfxt1_x1 + cfxtm_xm, cfxtm_xm + cfxt2_x2])

    Rmom = thlog + (2 + H + Hw - 0.) * uelog - 0.5 * xlog * cfxt
    Rmom_U = thlog_U + (H_U + Hw_U - Z8) * uelog + (2 + H + Hw - 0.) * uelog_U - 0.5 * xlog * cfxt_U
    Rmom_x = -0.5 * xlog_x * cfxt - 0.5 * xlog * cfxt_x

    cDixt1, cDixt1_U1, cDixt1_x1 = _cDixt(U1, x[0], P)
    cDixt2, cDixt2_U2, cDixt2_x2 = _cDixt(U2, x[1], P)
    cDixt, cDixt_U = _upwind(upw, upw_U, cDixt1, cDixt1_U1, cDixt2, cDixt2_U2)
    cDixt_x = np.array([(1. - upw) * cDixt1_x1, upw * cDixt2_x2])

    cfxtu, cfxtu_U = _upwind(upw, upw_U, cfxt1, cfxt1_U1, cfxt2, cfxt2_U2)
    cfxtu_x = np.array([(1. - upw) * cfxt1_x1, upw * cfxt2_x2])

    # Mach 0: Hss = 0
    Rshape = Hslog + (0. + 1 - H - Hw) * uelog + xlog * (0.5 * cfxtu - cDixt)
    Rshape_U = Hslog_U + (Z8 - Z8 - H_U - Hw_U) * uelog \
        + (0. + 1 - H - Hw) * uelog_U + xlog * (0.5 * cfxtu_U - cDixt_U)
    Rshape_x = xlog_x * (0.5 * cfxtu - cDixt) + xlog * (0.5 * cfxtu_x - cDixt_x)

    R = np.array([Rmom, Rshape, Rlag])
    R_U = np.empty((3, 8)); R_U[0] = Rmom_U; R_U[1] = Rshape_U; R_U[2] = Rlag_U
    R_x = np.empty((3, 2)); R_x[0] = Rmom_x; R_x[1] = Rshape_x; R_x[2] = Rlag_x
    return R, R_U, R_x


def residual_station(param, x, Uin, Aux):
    return _residual_station(pack(param), np.asarray(x, dtype=np.float64),
                             np.ascontiguousarray(Uin, dtype=np.float64), np.asarray(Aux, dtype=np.float64))


def get_damp(U, param):
    return _damp(np.asarray(U, dtype=np.float64), pack(param))


# ------------------------------------------------------------- global system assembly (COO)
class _COO:
    """collects dense blocks (rows given by a start index and height, cols by a list of starts)"""
    __slots__ = ('r', 'c', 'v')

    def __init__(self):
        self.r = []; self.c = []; self.v = []

    def add_block(self, rows, cols, vals):
        rows = np.asarray(rows); cols = np.asarray(cols)
        self.r.append(np.repeat(rows, len(cols))); self.c.append(np.tile(cols, len(rows)))
        self.v.append(np.asarray(vals, float).ravel())

    def tocsr(self, shape):
        if not self.r:
            return sparse.csr_matrix(shape)
        return sparse.csr_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))),
                                 shape=shape)


_AR3 = np.arange(3); _AR4 = np.arange(4)


def build_glob_sys(M):
    Nsys = M.glob.Nsys
    R = np.zeros(3 * Nsys)
    RU = _COO(); RX = _COO()
    st_r, st_c1, st_c2, st_U, st_x = [], [], [], [], []
    for si in range(3):
        Is = M.vsol.Is[si]
        xi = M.isol.xi[Is]
        N = len(Is)
        U = M.glob.U[:, Is]
        Aux = np.zeros(N)
        param = mf.build_param(M, si)
        if si == 2:
            Aux[:] = M.vsol.wgap
        i0 = 1 if (si < 2) and (xi[0] < 1e-8 * xi[-1]) else 0
        if si < 2:
            Ip = [i0, i0 + 1]
            Ust, Ust_U, Ust_x, xst = mf.stagnation_state(U[:, Ip], xi[Ip])
            param.turb, param.simi = False, True
            R1, R1_Ut, R1_x = residual_station(param, np.r_[xst, xst], np.stack((Ust, Ust), axis=-1), Aux[[i0, i0]])
            param.simi = False
            R1_Ust = R1_Ut[:, 0:4] + R1_Ut[:, 4:8]
            R1_U = np.dot(R1_Ust, Ust_U)
            R1_x = np.dot(R1_Ust, Ust_x)
            J = [Is[i0], Is[i0 + 1]]
            if i0 == 1:
                rows = np.arange(3 * Is[0], 3 * Is[0] + 3)
                R[rows] = U[0:3, 0] - Ust[0:3]
                RU.add_block(rows, np.arange(4 * Is[0], 4 * Is[0] + 4), np.eye(3, 4))
                RU.add_block(rows, np.arange(4 * J[0], 4 * J[0] + 4), -Ust_U[0:3, 0:4])
                RU.add_block(rows, np.arange(4 * J[1], 4 * J[1] + 4), -Ust_U[0:3, 4:8])
                RX.add_block(rows, np.array(J), -Ust_x[0:3, :])
        else:
            R1, R1_U, J = mf.wake_sys(M, param)
            R1_x = []
            param.turb, param.wake = True, True
        rows = np.arange(3 * Is[i0], 3 * Is[i0] + 3)
        R[rows] = R1
        for j in range(len(J)):
            RU.add_block(rows, np.arange(4 * J[j], 4 * J[j] + 4), R1_U[:, 4 * j:(4 * j + 4)])
            if len(R1_x) > 0:
                RX.add_block(rows, np.array([J[j]]), R1_x[:, j:(j + 1)])
        for i in range(i0 + 1, N):
            Ip = [i - 1, i]
            tran = M.vsol.turb[Is[i - 1]] ^ M.vsol.turb[Is[i]]
            if tran:
                Ri, Ri_U, Ri_x = mf.residual_transition(M, param, xi[Ip], U[:, Ip], Aux[Ip])
                mf.store_transition(M, si, i)
            else:
                Ri, Ri_U, Ri_x = residual_station(param, xi[Ip], U[:, Ip], Aux[Ip])
            r0 = 3 * Is[i]
            R[r0:r0 + 3] += Ri
            st_r.append(r0); st_c1.append(4 * Is[i - 1]); st_c2.append(4 * Is[i])
            st_U.append(Ri_U); st_x.append(Ri_x)
            if tran:
                param.turb = True
    if st_r:
        r0 = np.array(st_r); c1 = np.array(st_c1); c2 = np.array(st_c2)
        UU = np.array(st_U); XX = np.array(st_x)          # (n,3,8), (n,3,2)
        rows = (r0[:, None, None] + _AR3[None, :, None])
        RU.r.append(np.broadcast_to(rows, (len(r0), 3, 4)).ravel())
        RU.c.append(np.broadcast_to(c1[:, None, None] + _AR4[None, None, :], (len(r0), 3, 4)).ravel())
        RU.v.append(UU[:, :, 0:4].ravel())
        RU.r.append(np.broadcast_to(rows, (len(r0), 3, 4)).ravel())
        RU.c.append(np.broadcast_to(c2[:, None, None] + _AR4[None, None, :], (len(r0), 3, 4)).ravel())
        RU.v.append(UU[:, :, 4:8].ravel())
        xc = np.stack([c1 // 4, c2 // 4], axis=1)        # node indices
        RX.r.append(np.broadcast_to(rows, (len(r0), 3, 2)).ravel())
        RX.c.append(np.broadcast_to(xc[:, None, :], (len(r0), 3, 2)).ravel())
        RX.v.append(XX.ravel())
    R_x = RX.tocsr((3 * Nsys, Nsys))
    x_st = -M.isol.sgnue
    x_st = np.concatenate((x_st, -np.ones(M.wake.N)))
    R_st = R_x @ x_st
    Ist, st_ue = M.isol.Istag, M.isol.sstag_ue
    allrows = np.arange(3 * Nsys)
    RU.add_block(allrows, np.array([4 * Ist[0] + 3]), (R_st * st_ue[0])[:, None])
    RU.add_block(allrows, np.array([4 * Ist[1] + 3]), (R_st * st_ue[1])[:, None])
    M.glob.R = R
    M.glob.R_U = RU.tocsr((3 * Nsys, 4 * Nsys))
    M.glob.R_x = R_x


_orig_solve_glob = mf.solve_glob


def solve_glob(M):
    if M.oper.givencl:
        return _orig_solve_glob(M)
    Nsys = M.glob.Nsys
    ue = M.glob.U[3, :]; ds = M.glob.U[1, :]
    uemax = max(abs(ue))
    np.maximum(ue, 1e-10 * uemax, out=ue)   # same as the original per-element loop (in place on U)
    ueinv = mf.get_ueinv(M)
    R = np.concatenate((M.glob.R, ue - (ueinv + M.vsol.ue_m @ (ds * ue))))
    ue_m = M.vsol.ue_m
    ue_m = ue_m.toarray() if sparse.issparse(ue_m) else np.asarray(ue_m)
    bot = np.zeros((Nsys, 4 * Nsys))
    bot[:, 3::4] = np.eye(Nsys) - ue_m * ds[None, :]
    bot[:, 1::4] = -ue_m * ue[None, :]
    RV = sparse.vstack([M.glob.R_U, sparse.csr_matrix(bot)], format='csc')
    dV = -sparse.linalg.spsolve(RV, R)
    M.glob.dU = np.reshape(dV[0:4 * Nsys], (4, Nsys), order='F')


def build_wake(M):
    """mfoil.build_wake, with one fix: for a sharp (closed) TE the TE-segment normal is zero, so the first
    wake direction is taken from the TE bisector (TE_info) instead of failing the 'wrong wake direction' check."""
    assert len(M.isol.gam) > 0, 'No inviscid solution'
    N = M.foil.N
    Vinf = M.oper.Vinf
    Nw = int(np.ceil(N / 10 + 10 * M.geom.wakelen))
    S = M.foil.s
    ds1 = 0.5 * (S[1] - S[0] + S[N - 1] - S[N - 2])
    sv = mf.space_geom(ds1, M.geom.wakelen * M.geom.chord, Nw)
    xyw = np.zeros([2, Nw]); tw = xyw.copy()
    xy1, xyN = M.foil.x[:, 0], M.foil.x[:, N - 1]
    xyte = 0.5 * (xy1 + xyN)
    n = xyN - xy1; t = np.array([n[1], -n[0]])
    if np.hypot(*n) < 1e-9 * M.geom.chord:
        tb = mf.TE_info(M.foil.x)[0]
        t = tb / np.hypot(*tb)
    assert t[0] > 0, 'Wrong wake direction; ensure airfoil points are CCW'
    xyw[:, 0] = xyte + 1e-5 * t * M.geom.chord
    sw = S[N - 1] + sv
    X = np.ascontiguousarray(M.foil.x, dtype=np.float64)
    _, hTE, dtdx, tcp, tdp = mf.TE_info(M.foil.x)
    G = np.ascontiguousarray(M.isol.gam, dtype=np.float64)
    VG = np.zeros((2, N))
    ca, sa = mf.cosd(M.oper.alpha), mf.sind(M.oper.alpha)
    def vel(G_, ca_, sa_, p):
        v0, v1 = _invvel(X, G_, tcp, tdp, float(p[0]), float(p[1]), VG)
        return np.array([v0 + Vinf * ca_, v1 + Vinf * sa_])
    for i in range(Nw - 1):
        v1 = vel(G, ca, sa, xyw[:, i]); v1 = v1 / mf.norm2(v1); tw[:, i] = v1
        xyw[:, i + 1] = xyw[:, i] + (sv[i + 1] - sv[i]) * v1
        v2 = vel(G, ca, sa, xyw[:, i + 1]); v2 = v2 / mf.norm2(v2); tw[:, i + 1] = v2
        xyw[:, i + 1] = xyw[:, i] + (sv[i + 1] - sv[i]) * 0.5 * (v1 + v2)
    uewi = np.zeros([Nw, 1]); uewiref = np.zeros([Nw, 2])
    G0 = np.ascontiguousarray(M.isol.gamref[:, 0]); G1 = np.ascontiguousarray(M.isol.gamref[:, 1])
    for i in range(Nw):
        uewi[i] = np.dot(vel(G, ca, sa, xyw[:, i]), tw[:, i])
        uewiref[i, 0] = np.dot(vel(G0, mf.cosd(0), mf.sind(0), xyw[:, i]), tw[:, i])
        uewiref[i, 1] = np.dot(vel(G1, mf.cosd(90), mf.sind(90), xyw[:, i]), tw[:, i])
    M.wake.N = Nw
    M.wake.x = xyw
    M.wake.s = sw
    M.wake.t = tw
    M.isol.uewi = uewi
    M.isol.uewiref = uewiref


_orig_TE_info = mf.TE_info


def TE_info(X):
    """mfoil.TE_info; for a sharp TE (zero gap) the TE-panel direction p is undefined (0/0); return
    tcp = tdp = 0 so the zero-length TE source/vortex panel contributes nothing (as in XFOIL)."""
    s = X[:, -1] - X[:, 0]
    if np.hypot(s[0], s[1]) < 1e-10:
        t1 = X[:, 0] - X[:, 1]; t1 = t1 / mf.norm2(t1)
        t2 = X[:, -1] - X[:, -2]; t2 = t2 / mf.norm2(t2)
        t = 0.5 * (t1 + t2); t = t / mf.norm2(t)
        dtdx = t1[0] * t2[1] - t2[0] * t1[1]
        return t, 0.0, dtdx, 0.0, 0.0
    return _orig_TE_info(X)


LOOSE = dict(window=10, min_iter=20, dcl=0.01, dcd_rel=0.05, dcm=0.005, rnorm=1.0)


def solve_coupled(M):
    """mfoil.solve_coupled plus a history of (Rnorm, cl, cd, cm) per Newton step.

    Strict convergence is mfoil's own (Rnorm < rtol). mfoil's Newton often ends in a limit cycle when a
    transition point sits in the last panel or flips between two nodes; the forces then change by a few
    1e-4 from step to step. If the last `window` steps keep cl, cd and cm within LOOSE limits, we stop and
    flag the point as loosely converged (M.glob.conv_loose = True) and report the median forces of the window.
    """
    nNewton = M.param.niglob
    M.glob.conv = False; M.glob.conv_loose = False
    M.glob.realloc = True
    hist = []
    for iNewton in range(nNewton):
        mf.build_glob_sys(M)
        _calc_force_light(M)
        Rnorm = mf.norm2(M.glob.R)
        hist.append((Rnorm, M.post.cl, M.post.cd, M.post.cm))
        if Rnorm < M.param.rtol:
            M.glob.conv = True
            break
        if not np.isfinite(Rnorm):
            break
        if iNewton + 1 >= LOOSE['min_iter'] and _loose_ok(hist):
            M.glob.conv_loose = True
            break
        mf.solve_glob(M)
        mf.update_state(M)
        M.glob.realloc = False
        mf.stagpoint_move(M)
        mf.update_transition(M)
    if (not M.glob.conv) and (not M.glob.conv_loose) and len(hist) >= LOOSE['window'] and _loose_ok(hist):
        M.glob.conv_loose = True
    M.glob.hist = np.array(hist)


def _calc_force_light(M):
    """cl, cm (pressure integration, vectorised) and cd (Squire-Young) for the Newton history; the full
    mfoil calc_force (with the skin-friction split) still runs once after convergence in solve_viscous."""
    xref = M.geom.xref; alpha = M.oper.alpha; N = M.foil.N; chord = M.geom.chord
    ue = M.glob.U[3, :N]
    cp = 1 - (ue / M.param.Vinf) ** 2
    X = M.foil.x
    ip = np.r_[N - 1, np.arange(0, N - 1)]; ii = np.r_[0, np.arange(1, N)]
    x1 = X[:, ip]; x2 = X[:, ii]; dxv = x2 - x1
    dx1 = x1 - xref[:, None]; dx2 = x2 - xref[:, None]
    dx1nds = dxv[0] * dx1[0] + dxv[1] * dx1[1]; dx2nds = dxv[0] * dx2[0] + dxv[1] * dx2[1]
    ca, sa = mf.cosd(alpha), mf.sind(alpha)
    dx = -dxv[0] * ca - dxv[1] * sa
    cp1, cp2 = cp[ip], cp[ii]; cpbar = 0.5 * (cp1 + cp2)
    cl = np.sum(dx * cpbar) / chord
    cm = np.sum(cp1 * dx1nds / 3 + cp1 * dx2nds / 6 + cp2 * dx1nds / 6 + cp2 * dx2nds / 3) / chord ** 2
    iw = M.vsol.Is[2][-1]; U = M.glob.U[:, iw]
    H = U[1] / U[0]
    cd = 2.0 * U[0] * (U[3] / M.param.Vinf) ** ((5 + H) / 2.)
    M.post.cl, M.post.cm, M.post.cd = cl, cm, cd


def _loose_ok(hist):
    w = LOOSE['window']
    if len(hist) < w:
        return False
    h = np.array(hist[-w:])
    if not np.all(np.isfinite(h)):
        return False
    rn, cl, cd, cm = h.T
    return (np.ptp(cl) < LOOSE['dcl'] and np.ptp(cd) < LOOSE['dcd_rel'] * np.median(cd) and
            np.ptp(cm) < LOOSE['dcm'] and np.median(rn) < LOOSE['rnorm'] and np.median(cd) > 0)


def apply():
    mf.build_gamma = build_gamma
    mf.inviscid_velocity = inviscid_velocity
    mf.calc_ue_m = calc_ue_m
    mf.build_wake = build_wake
    mf.TE_info = TE_info
    mf.solve_coupled = solve_coupled
    mf.residual_station = residual_station
    mf.get_damp = get_damp
    mf.build_glob_sys = build_glob_sys
    mf.solve_glob = solve_glob


apply()
