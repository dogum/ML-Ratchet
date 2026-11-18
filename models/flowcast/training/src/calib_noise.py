"""Rung p3 measurement (dev only): how large must the simulated forcing error be?

Ceiling seen in reality: per-basin calibrated GR4J (NSE objective) on the 52 dev basins reaches median NSE 0.557
in its own calibration window (checks/dev_calibration_nse.json) -- forcing error + structure error + gauge error.
In the simulator, the same GR4J with the same parameters, driven by forcing_error(ERA5) instead of ERA5 itself,
loses only the forcing part. For each forcing-error setting we report the median NSE(true run, noisy-forcing run).
If that stays near 0.9 while reality allows ~0.56, the sim's forcing is too clean and the network learns to trust
daily rain timing more than real ERA5-Land allows.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import waterdata as wd
import watersim as ws
from watersim import noise

CALP = json.load(open(f'{WATER}/checks/dev_calibrated_params.json'))['params']


def jitter_p(df, rng, jit):
    """Move a random share (0..jit) of each day's precipitation to the previous or next day (timing error)."""
    if jit <= 0:
        return df
    P = df['P'].values.copy()
    a = rng.uniform(0, jit, len(P)) * (P > 0)
    d = rng.choice([-1, 1], len(P))
    mv = a * P
    out = P - mv
    j = np.clip(np.arange(len(P)) + d, 0, len(P) - 1)
    np.add.at(out, j, mv)
    df = df.copy(); df['P'] = out
    return df


SETTINGS = {'default (cv 0.25)': (0.25, 0.08, 0.0), 'cv 0.4 + jitter 0.3': (0.4, 0.1, 0.3),
            'cv 0.5 + jitter 0.5': (0.5, 0.12, 0.5), 'cv 0.7 + jitter 0.6': (0.7, 0.15, 0.6)}

if __name__ == '__main__':
    rows = []
    for k, b in enumerate(wd.basins('camels', split='dev')):
        prm = CALP[b]['gr4j+degree_day']
        spec = wd.spec(b); meta = wd.forcing_meta(b)
        df = wd.forcing(b, '1999-10-01', DEVWIN[1])
        o0 = ws.simulate(spec, (df, meta), DEVWIN[0], DEVWIN[1], model='gr4j', params=prm, labels=False)['ts']['Q'].values
        for name, (cv, pb, jit) in SETTINGS.items():
            rng = np.random.default_rng([3, k])
            for rep in range(2):
                dn = noise.forcing_error(df, rng, p_day_cv=cv, p_bias_sd=pb)
                dn = jitter_p(dn, rng, jit)
                o1 = ws.simulate(spec, (dn, meta), DEVWIN[0], DEVWIN[1], model='gr4j', params=prm, labels=False)['ts']['Q'].values
                rows.append({'basin': b, 'setting': name, 'nse': ws.metrics.nse(o0, o1) if hasattr(ws, 'metrics') else
                             1 - np.sum((o1 - o0) ** 2) / np.sum((o0 - o0.mean()) ** 2)})
        print(k, flush=True)
    d = pd.DataFrame(rows)
    s = d.groupby('setting').nse.median()
    print(s.round(3))
    jdump({'median_nse_true_vs_noisy_forcing': s.to_dict(), 'real_ceiling_calibrated_gr4j_nse': 0.557},
          f'{ROOT}/results/calib_noise.json')
