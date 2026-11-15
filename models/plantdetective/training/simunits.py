"""Random unit parameters for PlantDetective's simulated units (wider than gridsim.generate_fleet, so that the network
sees the real spread).  Ranges: EIA-860 sizes (gridsim.generator.SIZE), EIA-923 operating heat rates (wider tails),
RTS-GMLC ramp / min up-down defaults (x lognormal), part-load shape from a linear fuel curve
F(P) = a + b P  ->  HR(f) = HR_full (1 + beta (1/f - 1)), beta = no-load fuel share at full load, plus a small
convex term.  These are PRIORS for training only; nothing here is fitted to exam data."""
import numpy as np

TECHS = ['coal_steam', 'gas_cc', 'gas_ct', 'gas_steam']
SIZE = {'coal_steam': (150, 900), 'gas_cc': (100, 800), 'gas_ct': (20, 250), 'gas_steam': (40, 800)}
HR_FULL = {'coal_steam': (9.0, 12.0), 'gas_cc': (6.3, 8.8), 'gas_ct': (8.8, 13.5), 'gas_steam': (9.2, 13.5)}
BETA = {'coal_steam': (0.02, 0.14), 'gas_cc': (0.02, 0.16), 'gas_ct': (0.08, 0.45), 'gas_steam': (0.03, 0.18)}
PMIN = {'coal_steam': (0.22, 0.62), 'gas_cc': (0.28, 0.72), 'gas_ct': (0.30, 0.92), 'gas_steam': (0.06, 0.50)}
RAMP = {'coal_steam': 0.015, 'gas_cc': 0.03, 'gas_ct': 0.10, 'gas_steam': 0.015}   # fraction of pmax per minute
UPDN = {'coal_steam': (8, 8), 'gas_cc': (4, 4), 'gas_ct': (1, 1), 'gas_steam': (6, 6)}
START_HEAT = {'coal_steam': 18.0, 'gas_cc': 6.0, 'gas_ct': 1.0, 'gas_steam': 15.0}  # MMBtu per MW (warm), RTS-GMLC
VOM = {'coal_steam': 4.5, 'gas_cc': 2.5, 'gas_ct': 5.0, 'gas_steam': 4.0}


def hr_curve(hr_full, beta, pmax, pmin_frac, curv=0.0, n=5):
    f = np.linspace(max(pmin_frac, 0.05), 1.0, n)
    hr = hr_full * (1 + beta * (1 / f - 1) + curv * (f - 1) ** 2)
    return [[float(a * pmax), float(b)] for a, b in zip(f, hr)]


def random_unit(tech, rng, pmax=None):
    lo, hi = SIZE[tech]
    pmax = float(pmax or np.exp(rng.uniform(np.log(lo), np.log(hi))))
    hr_full = float(rng.uniform(*HR_FULL[tech]))
    beta = float(rng.uniform(*BETA[tech]))
    pmin_frac = float(rng.uniform(*PMIN[tech]))
    curv = float(rng.uniform(0, 0.1))
    ramp = float(RAMP[tech] * np.exp(rng.normal(0, 0.6)))
    ut, dt = UPDN[tech]
    min_up = float(max(1, round(ut * np.exp(rng.normal(0, 0.5)))))
    min_dn = float(max(1, round(dt * np.exp(rng.normal(0, 0.5)))))
    return dict(tech=tech, pmax_MW=pmax, pmin_MW=pmin_frac * pmax, hr_full=hr_full, beta=beta, curv=curv,
                heat_rate_curve=hr_curve(hr_full, beta, pmax, pmin_frac, curv),
                ramp_up_MW_per_min=ramp * pmax, ramp_down_MW_per_min=ramp * pmax, min_up_h=min_up, min_down_h=min_dn,
                start_heat_MMBtu_per_MW=float(START_HEAT[tech] * np.exp(rng.normal(0, 0.4))),
                vom=float(VOM[tech] * np.exp(rng.normal(0, 0.3))))
