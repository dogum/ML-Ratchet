"""Bundle a few real EXAM tests for demo_nmc811.py: one exam cell drawn by fixed_draw(EXAM, 1, seed=811) -> W9, its first
test (the reference) and five later tests drawn by fixed_draw(tests 2-15, 5, seed=811). Not picked by result.
Data: Pozzato, Allam & Onori 2022 (CC BY 4.0). Writes examples/m50t_<cell>.npz.
"""
import _paths as _P
import os, sys, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, os.path.dirname(os.path.dirname(_P.MODEL)))
import nmc_data as D
from mlratchet.splits import fixed_draw

if __name__ == '__main__':
    cell = fixed_draw(D.EXAM, 1, seed=811)[0]; rows = D.readings([cell]); later = [r['diag'] for r in rows if r['diag'] > 1]
    pick = [1] + fixed_draw(later, 5, seed=811); R = [r for r in rows if r['diag'] in pick]
    out = os.path.join(_P.MODEL, 'examples', f'm50t_{cell}.npz')
    np.savez_compressed(out, cell=cell, diag=np.array([r['diag'] for r in R]), cycle=np.array([-1 if r['cycle'] is None else r['cycle'] for r in R]),
                        Q=np.stack([r['q'] for r in R]).astype(np.float32), q_total=np.array([r['q_total'] for r in R]), vgrid=D.VG,
                        charge_c=D.CHARGE_C[cell], source='Pozzato, Allam & Onori 2022, Data in Brief 41, 107995; https://osf.io/qsabn/ ; CC BY 4.0')
    print(out, pick, os.path.getsize(out))
