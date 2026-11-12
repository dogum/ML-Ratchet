"""MODEL_CARD.md and README.md for the TransitNet release, written from the logs."""
import _paths as _P  # portable paths (see _paths.py)
import json, numpy as np
D = _P.WORK + '/'; R = json.load(open(D + 'ratchet.json')); rel = [r for r in R if r['accepted']][-1]; ex = rel['exam']; PA = ex['params_all']; BS = ex['params_by_snr']
PH = json.load(open(D + 'physics_check.json')); T = json.load(open(D + 'web/texts.json')); BL = json.load(open(D + 'data/baseline_fits.json')); data = json.load(open(D + 'web/data.json'))
p = lambda x, d=1: '—' if x is None else f"{100 * x:.{d}f}%"
rows = '\n'.join(f"| {i + 1} | {r['note'].split(':')[0]} | {'kept' if r['accepted'] else 'rejected'} | {r['objective']:.3f} | {r['val']['auc']:.3f} | {r['dev']['logloss']:.3f} | {p(r['dev']['k_frac_med'])} | {r['exam']['auc']:.3f} | {p(r['exam']['params_all']['k_frac']['median'])} | {r['exam']['params_all']['b_abs']['median']:.3f} |" for i, r in enumerate(R))
phys = '\n'.join(f"| {r['k']} | {r['b']} | {r['e']} | {r['depth_ppm']:.0f} | {r['max_err_ppm']:.3f} |" for r in PH['transit'])
big = '\n'.join(f"| {r['z']} | {r['p']} | {r['ours']:.5f} | {r['brute']:.5f} |" for r in PH['big_occulter'])
snr = '\n'.join(f"| {k.replace('-1000000000.0', '+').replace('-', ' to ')} | {v['k_frac']['n']} | {p(v['k_frac']['median'])} | {p(v['k_frac_naive']['median'])} | {v['b_abs']['median']:.3f} | {p(v['aR_frac']['median'])} | {p(v['k_in_1sigma'], 0)} |" for k, v in BS.items())
par = '\n'.join(f"| {r['label']} | {r['n']} | {r['k']} | {r['b']} | {r['a']} |" for r in data['exam']['par'])
cls = '\n'.join(f"| {r['label'].strip('. ')} | {r['n']} | {p(r['called'], 0)} |" for r in data['exam']['cls'])
card = f"""# TransitNet model card ({T['version']})

**What it is.** A 1-D convolutional network (five input branches, about 330,000 weights) that reads a TESS 2-minute light curve of a star with a candidate periodic dip
(period, epoch, duration) and returns what the dip is (planet, eclipsing binary on the star, background eclipsing binary
blended into its light, or nothing real) and, for a planet, the radius ratio Rp/R*, impact parameter b, a/R* and the
duration, each with a 68% interval. Trained on **zero real planets**.

## The physics

Quadratic limb darkening; the light a disk of radius p blocks at separation z, integrated exactly over the covered annuli
(48-point Gauss-Legendre in a variable that removes the square-root endpoints), valid for any p, so the same function
eclipses a star by a bigger one. Keplerian orbits (eccentric beyond 4 days for planets), 2-minute exposures averaged.
Eclipsing binaries: both eclipses, surface-brightness ratio, ellipsoidal variation, third light. Background binaries: the
same, diluted 3 to 3,000 times, moving the flux-weighted centroid by (1 - m)/m x (1 - phi) x separation.

Against batman (Kreidberg 2015, exact Mandel & Agol quadratic limb darkening), instantaneous flux:

| k | b | e | depth (ppm) | max error (ppm) |
|---|---|---|---|---|
{phys}

A star eclipsed by a bigger one, against pixel-by-pixel integration (4001 x 4001):

| z | p | ours | brute force |
|---|---|---|---|
{big}

## The data

Noise: 2,200 ordinary TESS 2-minute targets (no TOI, no known planet), 220 from each of sectors 3, 9, 15, 22, 28, 36, 44, 50, 57
and 64; each simulated reading stitches 1 to 3 of them of similar brightness. Signals: planets (45%: P 0.4-25 d log-uniform,
Rp/R* 0.01-0.3 log-uniform, b mostly 0-1, a/R* from the TIC density by Kepler's law, limb darkening drawn in Kipping's
q1, q2), eclipsing binaries (20%), background binaries (20%), nothing (15%). Catalogue-like candidate ephemerides: period errors
of 1e-7 to 3e-5 of the period (5x the stated error 15% of the time), epoch within 0.15 durations, duration within 15%, binaries
at half their period 15% of the time. 60,000 training readings; 5,000 frozen for the ratchet.

## The ratchet

{T['ratchetSub']}

| Rung | Change | Decision | Sim score | Sim AUC | Dev log-loss | Dev Rp/R* error | Exam AUC | Exam Rp/R* error | Exam b error |
|---|---|---|---|---|---|---|---|---|---|
{rows}

## Final exam

{T['exam']['sub']}

Planet or not (probability above 0.5 called a planet), AUC {ex['auc']:.3f}:

| Real signals | Count | Called a planet |
|---|---|---|
{cls}

Parameters by signal-to-noise (median absolute error; coverage: published value inside TransitNet's 68% interval with the
published error added):

| SNR | Planets | Rp/R* error | sqrt(depth) | b error | a/R* error | Rp/R* coverage |
|---|---|---|---|---|---|---|
{snr}

Against a least-squares transit fit on the same folded data ({BL['sec_per_fit']:.1f} s per star, fixed limb darkening, no error bars, no vetting):

| SNR and method | Planets | Rp/R* | b | a/R* |
|---|---|---|---|---|
{par}

## Limits

{chr(10).join('- ' + x for x in T['limits'])}

Data: NASA TESS (SPOC light curves via MAST), NASA Exoplanet Archive (pscomppars), ExoFOP TOI list.
"""
open(D + 'release/MODEL_CARD.md', 'w').write(card)
readme = f"""# TransitNet {T['version'].split(' ')[0]}

Is that dip in a star's light a planet? If so, how big, how tilted, how far out? A small network that learned from physics alone
(simulated transits and eclipsing binaries in real TESS noise) and was graded on {ex['n_planets']:,} confirmed planets and {ex['n_fp']:,}
false positives.

    pip install numpy scipy astropy
    python demo.py

{T['use']['code']}

MODEL_CARD.md: the physics checks, the data, every rung, the exam and the limits. `training/` holds every script.
"""
open(D + 'release/README.md', 'w').write(readme); print('card', len(card))
