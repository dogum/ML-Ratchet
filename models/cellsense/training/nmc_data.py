"""CellSense NMC811: the real cells. Pozzato et al. 2022 (LG INR21700-M50T, NMC811 / graphite-SiOx, 4.85 Ah nominal), ten
cells aged at 23 C with UDDS driving discharges and CC-CV charging at C/4 to 3C. Every few weeks each cell had a reference
performance test; its capacity test is a C/20 (0.24 A) discharge from full (CC-CV to 4.2 V, rest) down to 2.5 V, logged
every second. Those curves are what CellSense-NMC reads.

Split by CELL, never by reading: three dev cells drawn by mlratchet.splits.split_units(cells, 3, seed=811) -> W3, W4, W8.
The other seven (W5, W7, W9, W10, G1, V4, V5) are the exam and never choose anything.

Grid: capacity discharged Q (Ah) at 250 voltages from 4.15 V down to 2.50 V (6.6 mV apart). The first few minutes of the
test (4.19 -> 4.15 V) are left out because they mix the rest after charging with the discharge.
"""
import _paths as _P
import os, numpy as np

MAT = os.path.join(_P.WORK, 'data_m50t', 'diagnostic_tests', '_processed_mat')
CELLS = ['W3', 'W4', 'W5', 'W7', 'W8', 'W9', 'W10', 'G1', 'V4', 'V5']
DEV = ['W3', 'W4', 'W8']                         # split_units(CELLS, 3, seed=811)
EXAM = [c for c in CELLS if c not in DEV]
CHARGE_C = dict(W3=3, W4=0.25, W5=0.5, W7=0.25, W8=0.5, W9=1, W10=3, G1=3, V4=0.25, V5=1)   # Table 3 of the data article
# cycle counts at diagnostics 1-9, Table 3 of the data article (later diagnostics have no published count)
CYCLES = dict(W3=[0, 25, 75], W4=[0, 25, 75, 123, 132, 159, 176, 179], W5=[0, 25, 75, 125, 159, 167, 187, 194, 219], W7=[0, 25, 75, 125, 141],
              W8=[0, 25, 75, 125, 148, 150, 151, 157, 185], W9=[0, 25, 75, 122, 144, 145, 146, 150, 179], W10=[0, 25, 75, 122, 146, 148, 151, 159, 188],
              G1=[0, 25, 30, 37, 62], V4=[0, 20, 45, 70, 95], V5=[0, 12, 18, 29])
V_HI, V_LO, NV = 4.15, 2.50, 250
VG = np.linspace(V_HI, V_LO, NV)
I_TEST = 0.24


def load_raw():
    """{cell: [dict(diag, t, V, I, Q), ...]} from the processed capacity_test.mat"""
    import scipy.io as sio
    d = sio.loadmat(os.path.join(MAT, 'capacity_test.mat'), squeeze_me=True); lab = [str(x) for x in d['col_cell_label']]; out = {}
    for j, c in enumerate(lab):
        out[c] = []
        for i in range(d['vcell'].shape[0]):
            v = d['vcell'][i, j]
            if np.ndim(v) == 0: continue
            out[c].append(dict(diag=i + 1, t=np.asarray(d['time'][i, j], float), V=np.asarray(v, float), I=np.asarray(d['curr'][i, j], float),
                               Q=np.asarray(d['cap'][i, j], float)))
    return out


def q_of_v(V, Q, vgrid=VG, tol=0.001):
    """capacity discharged at each grid voltage (V falling, Q rising); NaN below where the curve ends. A test that stopped
    within tol (1 mV) of the 2.5 V cut-off counts as reaching it: the cycler stops at 2.50000-2.50004 V, a hair above the
    grid's last point, and those tests would otherwise look cut short"""
    V, Q = np.asarray(V, float), np.asarray(Q, float)
    if V[-1] - vgrid[-1] < tol: V, Q = np.append(V, vgrid[-1]), np.append(Q, Q[-1])
    Vm = np.minimum.accumulate(V); ok = (vgrid <= Vm[0]) & (vgrid >= Vm[-1]); q = np.full(len(vgrid), np.nan)
    q[ok] = np.interp(-vgrid[ok], -Vm, Q); q[vgrid > Vm[0]] = 0.0; return q


def readings(cells=None):
    """one row per capacity test: the Q(V) curve on the grid, the capacity to 2.5 V (the answer key), and the same cell's
    first test (its reference curve when new)"""
    raw = load_raw(); rows = []
    for c in (cells or CELLS):
        ref = None
        for r in raw[c]:
            m = r['I'] < -0.01; t, V, Q = r['t'][m], r['V'][m], r['Q'][m]
            Q = Q - Q[0] + I_TEST * (t[0] - r['t'][0]) / 3600 if Q[0] > 1e-3 else Q          # counted from the start of the discharge
            q = q_of_v(V, Q); row = dict(cell=c, diag=r['diag'], q=q, q_total=float(Q[-1]), V_start=float(r['V'][0]), V=V, Q=Q,
                                         cycle=(CYCLES[c][r['diag'] - 1] if r['diag'] <= len(CYCLES[c]) else None), charge_c=CHARGE_C[c])
            if ref is None: ref = row
            row['q_ref'] = ref['q']; row['q_total_ref'] = ref['q_total']; rows.append(row)
    return rows


if __name__ == '__main__':
    R = readings()
    for r in R:
        print(r['cell'], r['diag'], r['cycle'], 'V0 %.4f' % r['V_start'], 'Q %.4f' % r['q_total'], 'fade %.2f%%' % (100 * (1 - r['q_total'] / r['q_total_ref'])),
              'Q@3.6V %.3f' % r['q'][np.argmin(np.abs(VG - 3.6))])
