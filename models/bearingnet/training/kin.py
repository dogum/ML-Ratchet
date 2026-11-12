"""Rolling-bearing kinematics (Randall & Antoni 2011, eqs. 1-4; also Harris & Kotzalas). Stationary outer race, rotating inner.
   r = (d / D) cos(phi)
   FTF  = fr/2 (1 - r)                      cage (fundamental train)
   BPFO = n fr/2 (1 - r)                    ball pass, outer race
   BPFI = n fr/2 (1 + r)                    ball pass, inner race
   BSF  = D/(2d) fr (1 - r^2)               ball spin; a ball defect strikes the races at 2*BSF
All returned as orders (multiples of shaft speed fr)."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np

BEARINGS = {   # n balls, ball diameter d, pitch diameter D (same units), contact angle deg
    'SKF6205': dict(n=9, d=0.3126, D=1.537, phi=0.0),       # CWRU drive end, inches, from the CWRU bearing-information page
    'SKF6203': dict(n=8, d=0.2656, D=1.122, phi=0.0),       # CWRU fan end, inches, idem
    'PB6203': dict(n=8, d=6.75, D=28.55, phi=0.0),          # Paderborn test bearing 6203, mm (Lessmeier et al. 2016)
}

def orders(n, d, D, phi=0.0):
    r = d / D * np.cos(np.radians(phi))
    return dict(FTF=0.5 * (1 - r), BPFO=n / 2 * (1 - r), BPFI=n / 2 * (1 + r), BSF=D / (2 * d) * (1 - r * r))

def geom(name): return BEARINGS[name]

if __name__ == '__main__':
    # check against the CWRU published defect-frequency table (multiples of running speed). CWRU lists "Rolling Element" = 2*BSF.
    pub = {'SKF6205': dict(BPFI=5.4152, BPFO=3.5848, FTF=0.39828, B2=4.7135), 'SKF6203': dict(BPFI=4.9469, BPFO=3.0530, FTF=0.3817, B2=3.9874)}
    out = {}
    for b, p in pub.items():
        o = orders(**BEARINGS[b]); o['B2'] = 2 * o['BSF']
        out[b] = {k: dict(ours=round(o[k], 5), published=v, rel_err=float(abs(o[k] / v - 1))) for k, v in p.items()}
    print(out); import json; json.dump(out, open(_P.WORK + '/checks/kinematics_check.json', 'w'), indent=1)
    print('PB6203', orders(**BEARINGS['PB6203']))
