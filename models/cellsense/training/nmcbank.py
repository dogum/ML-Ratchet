"""CellSense NMC811 training data: pairs of C/20 discharges of the same simulated M50T cell, once new and once aged, as
capacity Q(V) on the 250-voltage grid (4.15 V to 2.50 V), with the aging written down exactly.

Physics versions (each one is a frozen-set version too, val_<physics>.npz):
  textbook  Chen2020 as published, cell-to-cell spread by hand; degradation modes drawn wide and independent
  cal       new cells drawn around the three dev cells' fits (nmc_calibrate.py fresh); mode ranges from the dev cells'
            voltage-fit trajectories (nmc_calibrate.py modes), widened
  ocp       cal, plus the open-circuit-curve correction fitted on the dev cells (nmc_ocpfit.py), with the fresh and mode
            fits redone on top of it
  ocpj      ocp, with the open-circuit curves jittered per cell: the correction scaled by 0.5-1.5 and a 4 mV-per-coefficient
            spline wiggle added to each electrode (the size of what the dev fits leave), so the network cannot lean on
            exact curve shapes
  ocpj2     ocpj twice as wide: correction scaled 0-2, 8 mV per coefficient
Aging: lithium lost (LLI, fraction of the new cell's cyclable lithium), negative electrode lost (LAM_NE, a random part of it
lithiated), positive lost (LAM_PE) and resistance growth. Labels: capacity to 2.5 V at C/20 (Ah), fade, lli, lam_ne, lam_pe,
r_growth.

    python nmcbank.py <physics> <seed> <n> <out.npz>
"""
import _paths as _P
import os, sys, json, numpy as np
sys.path.insert(0, _P.SRC); import nmcsim as NS, nmc_data as D, nmc_calibrate as CA
OUT = os.path.join(_P.WORK, 'nmc')
LABELS = ['q_total', 'fade', 'lli', 'lam_ne', 'lam_pe', 'r_growth']
NAMES = CA.NAMES


def load_cal(physics):
    tag = '_ocp' if physics == 'ocp' else ''
    F = json.load(open(os.path.join(OUT, f'cal_fresh{tag}.json'))); M = json.load(open(os.path.join(OUT, f'cal_modes{tag}.json')))
    corr = json.load(open(os.path.join(OUT, 'ocp_corr.json'))) if physics == 'ocp' else None
    return F, M, corr


def fresh(rng, B, physics, F=None):
    if physics == 'textbook':
        c = NS.chen2020(); ln = lambda v, sd: v * np.exp(rng.normal(0, sd, B))
        P = dict(Q_p=ln(c['Q_p'], 0.03), Q_n=ln(c['Q_n'], 0.03), Q_Li=ln(c['Q_Li'], 0.03), tau_p=ln(c['tau_p'], 0.3), tau_n=ln(c['tau_n'], 0.3),
                 I0_p=ln(c['I0_p'], 0.3), I0_n=ln(c['I0_n'], 0.3), R_ohm=rng.uniform(0.01, 0.04, B), sig_p=rng.uniform(1e-4, 0.01, B), sig_n=rng.uniform(1e-4, 0.01, B),
                 V_full=rng.uniform(4.183, 4.195, B))
    else:
        Z = np.array([F[k]['z'] for k in sorted(F)]); i = rng.integers(0, len(Z), B); j = rng.integers(0, len(Z), B)
        a = np.where(rng.random(B) < 0.5, rng.random(B), 1.0)[:, None]; z = a * Z[i] + (1 - a) * Z[j]
        sd = np.array([0.012, 0.012, 0.012, 0.15, 0.15, 0.15, 0.15, 0.15, 0.3, 0.3, 0.0])     # capacities vary little cell to cell
        z = z + rng.normal(0, 1, z.shape) * sd; z[:, -1] += rng.normal(0, 0.003, B)
        P = {n: np.exp(z[:, k]) for k, n in enumerate(NAMES)}; P['V_full'] = z[:, -1]
    P['Q_Li'] = np.minimum(P['Q_Li'], 0.999 * P['Q_n'] + 0.02 * P['Q_p'])
    P['R_e'] = np.full(B, 1e-4); P['tau_e'] = np.full(B, 100.0)
    return P


def mode_ranges(physics, M=None):
    """upper ends of the uniform-power mode draws. textbook: wide by hand. cal/ocp: 2.5x the largest dev voltage-fit value
    (the dev cells reach 6-9% fade; 2.5x leaves room for cells that age further), floored; resistance growth capped at 1.5"""
    if physics == 'textbook': return dict(lli=0.20, lam_ne=0.20, lam_pe=0.20, r_growth=1.0)
    mx = {k: max(abs(m['modes'][k]) for m in M.values()) for k in CA.MODES}
    if physics == 'ocp': return dict(lli=max(0.06, 2.5 * mx['lli']), lam_ne=max(0.06, 2.5 * mx['lam_ne']), lam_pe=max(0.06, 2.5 * mx['lam_pe']), r_growth=1.0, r_drop=0.5)
    return dict(lli=max(0.06, 2.5 * mx['lli']), lam_ne=max(0.06, 2.5 * mx['lam_ne']), lam_pe=max(0.06, 2.5 * mx['lam_pe']), r_growth=min(1.5, max(0.5, 2.5 * mx['r_growth'])))


def age(rng, P0, rg, corr=None):
    B = len(P0['Q_p']); young = rng.random(B) < 0.15; sk = lambda hi, p=1.3: np.where(young, 0.05, 1.0) * hi * rng.random(B) ** p
    m = np.stack([sk(rg['lli']), sk(rg['lam_ne']), sk(rg['lam_pe']), sk(rg['r_growth'], 1.0)], 1)
    if rg.get('r_drop'):   # ocp physics: the dev cells' fitted resistance FALLS 10-50% after their first test; allow both signs
        m[:, 3] = np.where(young, 0.05, 1.0) * rng.uniform(-rg['r_drop'], rg['r_growth'], B)
    P = CA.aged(P0, m, f_li=rng.random(B), corr=corr); P['V_full'] = P0['V_full'] + rng.normal(0, 0.002, B)    # rest voltage varies test to test
    return P, m


JITTER = {'ocpj': dict(scale=(0.5, 1.5), sd=0.004),     # the correction scaled per cell, plus 4 mV spline wiggles per electrode
          'ocpj2': dict(scale=(0.0, 2.0), sd=0.008)}    # twice as wide


def jitter(rng, P, K, physics='ocpj'):
    J = JITTER[physics]; B = len(P['Q_p']); P['ocp_scale'] = rng.uniform(*J['scale'], B)
    P['dU_p'] = rng.normal(0, J['sd'], (B, K)); P['dU_n'] = rng.normal(0, J['sd'], (B, K)); return P


def make(physics, seed, B, out=None, chunk=2000):
    rng = np.random.default_rng(seed); F, M, corr = (None, None, None) if physics == 'textbook' else load_cal('ocp' if physics.startswith('ocpj') else physics)
    rg = mode_ranges('ocp' if physics.startswith('ocpj') else physics, M)
    R = {k: [] for k in ('Q0', 'Q1', 'y')}
    for s in range(0, B, chunk):
        n = min(chunk, B - s); P0 = fresh(rng, n, physics, F)
        if physics.startswith('ocpj'): P0 = jitter(rng, P0, NS.N_SPL, physics)
        P1, m = age(rng, P0, rg, corr)
        T = rng.uniform(22.0, 24.0, n)
        o0 = NS.simulate(P0, dt=60.0, T=T, corr=corr); o1 = NS.simulate(P1, dt=60.0, T=T + rng.normal(0, 0.3, n), corr=corr)
        Q0, Q1 = NS.q_of_v(o0, D.VG), NS.q_of_v(o1, D.VG); q0, q1 = o0['Q_total'], o1['Q_total']
        Q_Li0 = P0['Q_Li']; lli_eff = 1 - P1['Q_Li'] / Q_Li0                    # all lithium lost, including what left with lost electrode
        y = np.stack([q1, 1 - q1 / q0, lli_eff, m[:, 1], m[:, 2], m[:, 3]], 1)
        ok = np.isfinite(Q0).sum(1) > 200
        for k, v in (('Q0', Q0[ok]), ('Q1', Q1[ok]), ('y', y[ok])): R[k].append(v.astype(np.float32))
        print(physics, seed, s + n, flush=True)
    R = {k: np.concatenate(v) for k, v in R.items()}
    if out: np.savez_compressed(out, **R, labels=np.array(LABELS), vgrid=D.VG, physics=physics, ranges=json.dumps(rg))
    return R


if __name__ == '__main__':
    make(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
