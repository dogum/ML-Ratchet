"""Numba version of the MPC dynamic-programming sweep in tp/mpc.py (same equations, checked equal to the numpy
loop in tmp/check_dp.py). Training/teacher code only; the released ThermoPilot does not need it."""
import numpy as np
from numba import njit


@njit(cache=True)
def _plf(f, c0, c1, c2):
    return c0 + f * (c1 + f * c2)


@njit(cache=True)
def part_power(P_on, f, speed, plf, plf2):
    # speed: 0 single, 1 variable, 2 two-stage, 3 other
    if f <= 0.0:
        return 0.0
    if speed == 0:
        return P_on * f / (plf[0] + f * plf[1])
    if f > 1.0:
        f = 1.0
    if speed == 1:
        if f >= 0.3:
            return P_on * f * (0.75 + 0.25 * f)
        f3 = max(f / 0.3, 1e-9)
        return P_on * 0.3 * (0.75 + 0.25 * 0.3) * f3 / _plf(min(f3, 1.0), plf[0], plf[1], plf[2])
    if speed == 2:
        return P_on * f / _plf(f, plf2[0], plf2[1], plf2[2])
    return P_on * f / _plf(f, plf[0], plf[1], plf[2])


@njit(cache=True)
def dp_sweep(grid, tm0, A, At, b, Cc, Ct, Cq, fcont, gx0, P_fan, hsp, csp, To, Qmax, Qh, Ph, Qs, Pc, kind,
             hp_speed, cool_speed, Qheat, eff, base, lo, hi, price, gas_price, lam, drate, frate, pref, fref,
             plf, plf2):
    """kind: 0 hp, 1 furnace, 2 fuel baseboard, 3 electric baseboard, 4 electric furnace / other.
    Returns the first-step action values (NA,)."""
    H, NA = hsp.shape
    NG = grid.shape[0]
    g0 = grid[0]
    dg = grid[1] - grid[0]
    V = np.zeros(NG)
    first = np.zeros(NA)
    for j in range(H - 1, -1, -1):
        npts = NG if j > 0 else 1
        Vn = np.empty(NG)
        fc = fcont[j]
        gx = gx0 if fc else 0.0
        q0 = P_fan if fc else 0.0
        bj = b[j]
        be = bj / (1.0 + bj * gx)
        for gi in range(npts):
            Tm = grid[gi] if j > 0 else tm0
            Aair = A[j] + At[j] * Tm
            Tb = (Aair + bj * (gx * To[j] + q0)) / (1.0 + bj * gx)
            best = 1e300
            for a in range(NA):
                hs = hsp[j, a]
                cs = csp[j, a]
                Qreq_h = (hs - Tb) / be if Tb < hs else 0.0
                Qreq_c = (Tb - cs) / be if (Tb > cs and Qs[j] > 0) else 0.0
                Qhd = min(Qreq_h, Qmax[j])
                Qcd = min(Qreq_c, Qs[j])
                T = Tb + be * (Qhd - Qcd)
                qtot = Qhd - Qcd
                if fc:
                    qtot += gx * (To[j] - T) + q0
                Tm1 = Cc[j] + Ct[j] * Tm + Cq[j] * qtot
                gas = 0.0
                if kind == 0:
                    if Qh[j] > 0:
                        f = min(Qhd, Qh[j]) / Qh[j]
                        run_h = 1.0 if Qhd > Qh[j] else f
                    else:
                        f = 0.0
                        run_h = 1.0 if Qhd > 0 else 0.0
                    Pel = part_power(Ph[j], f, hp_speed, plf, plf2) + max(Qhd - Qh[j], 0.0)
                elif kind == 1:
                    run_h = Qhd / max(Qheat, 1e-9)
                    gas = Qhd / eff
                    Pel = 0.004 * Qheat * run_h
                elif kind == 2:
                    run_h = 0.0
                    gas = Qhd / eff
                    Pel = 0.01 * Qhd
                elif kind == 3:
                    run_h = 0.0
                    Pel = Qhd
                else:
                    run_h = Qhd / max(Qheat, 1e-9)
                    Pel = Qhd
                fcl = Qcd / Qs[j] if Qs[j] > 0 else 0.0
                Pel += part_power(Pc[j], fcl, cool_speed, plf, plf2)
                if fc:
                    Pel += P_fan
                else:
                    Pel += P_fan * min(max(run_h + fcl, 0.0), 1.0)
                elec = (base[j] + Pel) / 1000.0
                dev = max(lo[j] - T, 0.0) + max(T - hi[j], 0.0)
                cost = price[j] * elec + gas_price * gas / 1000.0 + lam * dev
                if drate[j] > 0:
                    cost += drate[j] * max(elec - pref[j], 0.0)
                if frate[j] > 0:
                    cost += frate[j] * max(elec - fref[j], 0.0)
                pos = (Tm1 - g0) / dg
                if pos < 0.0:
                    pos = 0.0
                if pos > NG - 1.000001:
                    pos = NG - 1.000001
                i0 = int(pos)
                w = pos - i0
                Q = cost + V[i0] * (1.0 - w) + V[i0 + 1] * w
                if j > 0:
                    if Q < best:
                        best = Q
                else:
                    first[a] = Q
            if j > 0:
                Vn[gi] = best
        if j > 0:
            V = Vn
    return first
