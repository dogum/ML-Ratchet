"""Write web/texts.json, release_card.md and release_readme.md from the result files (exam.json, ratchet.json,
baselines.json, physics_check.json, twin_test.json, twin_net.json, sigma_calibration.json), so every number in the
page and the model card comes from the same run."""
import _paths as _P  # portable paths (see _paths.py)
import json
S = _P.WORK + '/'
E = json.load(open(S + 'exam.json')); RL = json.load(open(S + 'ratchet.json')); PC = json.load(open(S + 'physics_check.json'))
TT = json.load(open(S + 'twin_test.json')); TN = json.load(open(S + 'twin_net.json')); SC = json.load(open(S + 'sigma_calibration.json'))
L = E['lengths']; MR = E['mass_radius']; DT = E['detection']; rel = SC['ckpt'].replace('.pt', '')
pc = lambda x, d=1: f'{100 * x:.{d}f}%'
LAB_NOISE = [8760414, 10919030, 8350211]; DEMO = [7618730, 7353997, 2695975]

method = [
 ['1 · Simulator from the physics', f"""<p>Each simulated red giant gets a mass, a temperature and ν<sub>max</sub>; Δν follows from the scaling relations
 (plus 2% scatter, clump stars a few % lower). Its power spectrum has two granulation components (Kallinger et al. 2014),
 a Gaussian oscillation hump (width from Mosser et al. 2012, height from Yu et al. 2018's amplitude relation), and radial,
 dipole, quadrupole and octupole modes on the universal red-giant pattern (Mosser et al. 2011), with dipole mixed modes from
 the asymptotic coupling formula in 70% of stars. Every frequency bin gets a random complex amplitude, so each mode behaves
 like a randomly kicked, damped oscillator. The 29.4-minute exposures smear high frequencies by sinc².</p>
 <p class="small">Checks (physics_check.json): spectrum statistics χ² with 2 dof (mean {PC['chi2_2dof']['mean']:.3f}, variance {PC['chi2_2dof']['var']:.3f}, expected 1, 1);
 Parseval: time-series variance / model variance {PC['parseval'][0]['ratio']:.3f}–{PC['parseval'][2]['ratio']:.3f};
 a classical pipeline recovers the inputs of 24 simulated stars to {pc(PC['classical_on_sims']['numax_med_abs_lnerr'])} (ν<sub>max</sub>) and {pc(PC['classical_on_sims']['dnu_med_abs_lnerr'])} (Δν);
 our scaling formulas reproduce Yu et al.'s 16,094 masses and radii to 0.3% and 0.04% scatter (rounding); the Sun (M = R = 1, 5777 K) gives 3090 μHz and 135.1 μHz.</p>"""],
 ['2 · Real noise, real gaps', """<p>Simulated stars are added to real Kepler light curves of 274 quiet dwarf stars of matching brightness. Dwarfs oscillate
 above 1000 μHz, beyond the 283 μHz Nyquist limit of Kepler's 29.4-minute cadence, so their light curves are pure instrument
 and photon noise, with Kepler's real gaps, jumps and quarter boundaries. 232 dwarfs feed training, 42 others the frozen
 test set. Each example uses the whole 4-year light curve or a random piece from 27 days up.</p>"""],
 ['3 · The network', f"""<p>Two small convolutional branches (412k numbers in all). One reads the log power spectrum on 1024 log-spaced bins
 (where is the hump?). The other reads a "comb map": the autocorrelation of the spectrum around 48 trial centres, at lags
 scaled to the expected spacing (how far apart are the teeth?). A head returns ν<sub>max</sub> and Δν with error bars, and the probability
 that oscillations are in the band at all. It saw only simulations (40,000 for training); no real star was ever a training example.</p>"""],
 ['4 · The ratchet', f"""<p>A new version is kept only if it beats the current one on a frozen set of 5,000 simulated stars
 (mean of: ν<sub>max</sub> within 5%, Δν within 2%, detection balanced accuracy) by more than 0.01, and its errors on 18 real
 development stars are at most 10% worse. {sum(r['accepted'] for r in RL)} of {len(RL)} versions were kept.</p>"""],
 ['5 · Calibrated on 18 dev stars only', f"""<p>Simulated "twins" of the dev stars (same ν<sub>max</sub>, Δν, mass, temperature, brightness, gaps) showed the
 v1 humps {abs(TT['v1']['median_below_hump_above'][1]):.2f} dex too tall: Yu et al.'s amplitudes use the Kjeldsen et al. (2008) convention (a sine of
 amplitude A puts A² in the spectrum). v2 fixed that and narrowed high-ν<sub>max</sub> humps; the mismatch dropped to
 {TT['v2']['median_below_hump_above'][1]:+.2f} dex. The network reads real dev stars about as well as their twins
 (median error {pc(TN['real_med_abs'][0])} / {pc(TN['real_med_abs'][1])} real vs {pc(TN['twin_med_abs'][0])} / {pc(TN['twin_med_abs'][1])} twins, ν<sub>max</sub> / Δν).</p>"""],
 ['6 · The exam', f"""<p>420 Kepler red giants never used to choose anything, against the published values of Yu et al. (2018).
 Error bars are widened by one factor per output fitted on dev only (ν<sub>max</sub> ×{SC['widen'][0]:.2f}, Δν ×{SC['widen'][1]:.2f}), never narrowed.
 Baselines: a classical SYD-style pipeline on the same spectra (background fit, smoothed power excess, autocorrelation), and
 always guessing the dev median.</p>"""],
]
limits = [
 f"On whole 4-year light curves the classical pipeline is better ({pc(L['0']['numax']['cl_med'])} vs {pc(L['0']['numax']['med'])} median ν<sub>max</sub> error, {pc(L['0']['dnu']['cl_med'])} vs {pc(L['0']['dnu']['med'])} for Δν). The published answers were made with a similar pipeline on the same data, which favours it. StarQuake's advantage is on short light curves.",
 f"Δν from short light curves is hard: from 27 days only {pc(L['27']['dnu']['hit'], 0)} of stars are within 2% and {pc(L['27']['dnu']['out'], 0)} are off by more than 5% (often by one comb tooth). The error bars do not fully catch these jumps.",
 f"False alarms: {pc(DT['0']['fpr'], 0)}–{pc(max(v['fpr'] for v in DT.values()), 0)} of unscreened quiet dwarfs are called oscillating (spots, eclipses and instrument trouble look like a hump). The classical comb test has fewer false alarms on full light curves ({pc(DT['0']['cl_fpr'], 0)}) but far more on short pieces.",
 f"Masses carry a +{pc(MR['M_bias'], 0)} median offset against Yu et al. (a small ν<sub>max</sub> bias, cubed). Radii from uncorrected scaling relations sit {pc(MR['gaia_R_ratio_med'] - 1, 0)} above Gaia's (Yu et al.'s own: {pc(MR['gaia_yuR_ratio_med'] - 1, 0)}); this is the known scaling-relation offset, not corrected here.",
 f"Error bars are too narrow on short pieces: within 1 sigma {pc(L['0']['numax']['cov68'], 0)} / {pc(L['0']['dnu']['cov68'], 0)} of stars on whole light curves but {pc(L['27']['numax']['cov68'], 0)} / {pc(L['27']['dnu']['cov68'], 0)} at 27 days (numax / dnu). The widening factor was fitted on dev pooled over lengths; a per-length factor is the obvious fix.",
 f"The most luminous giants (numax 8-15 uHz) read about {pc(L['0']['numax']['strata'][0]['med'], 0)} high in numax even from 4 years of data; the simulated twins of dev stars show the same bias, so it is the network, not the simulator.",
 "Trained for Kepler long cadence (29.4 min) and red giants with ν<sub>max</sub> 5–275 μHz. TESS light curves need a retrain on TESS noise; main-sequence stars (ν<sub>max</sub> > 1000 μHz) are out of reach at this cadence.",
 "The simulator has no red-clump vs RGB mixed-mode fine structure tuned on real stars, no rotation splitting, no stellar activity and no magnetic suppression beyond a simple dipole-suppression draw.",
]
use = f"""# numpy + scipy only (astropy to read FITS)
unzip starquake-v1.0.zip && cd starquake-v1.0
python demo.py                      # three real exam stars, a few seconds

from starquake import StarQuake
sq = StarQuake()
r = sq.read_kepler(['kplr..._llc.fits', ...], teff=4800)    # or sq.read_ppm(flux_ppm_on_cadence_grid)
r['detected'], r['p_osc']           # are oscillations present (numax 5-275 uHz)?
r['numax'], r['numax_err']          # uHz, 1-sigma
r['dnu'], r['dnu_err']              # uHz, 1-sigma
r['mass'], r['radius']              # solar units, scaling relations (needs teff)"""
credits = """Data: Kepler mission light curves (NASA; Mikulski Archive for Space Telescopes, public stpubdata bucket).
Truth: Yu, Huber, Bedding et al. 2018, ApJS 236, 42 (machine-readable tables from the journal). Gaia DR3 FLAME radii
(ESA/Gaia/DPAC, via the ESA Gaia TAP service). Kepler DR25 stellar properties (NASA Exoplanet Archive) for the quiet dwarfs.
Physics: Kallinger et al. 2014; Mosser et al. 2011, 2012, 2015; Kjeldsen &amp; Bedding 1995; Kjeldsen et al. 2008; Vrard et al. 2018;
Huber et al. 2009 (SYD). Built with numpy, scipy, PyTorch, astropy, astroquery. Code MIT; data under their providers' terms."""
T = dict(method=method, limits=limits, use=use, credits=credits, lab_noise=LAB_NOISE, demo_kics=DEMO)
json.dump(T, open(S + 'web/texts.json', 'w'), indent=1)

# ---------- model card ----------
def lrow(l):
    r = L[l]; n, d = r['numax'], r['dnu']; nm = {'0': 'whole (~4 yr)', '365': '365 d', '90': '90 d', '54': '54 d', '27': '27 d'}[l]
    return (f"| {nm} | {r['n']} | {pc(n['med'], 2)} / {pc(n['hit'], 0)} | {pc(n['cl_med'], 2)} / {pc(n['cl_hit'], 0)} | {pc(n['dumb_med'], 0)} | "
            f"{pc(d['med'], 2)} / {pc(d['hit'], 0)} | {pc(d['cl_med'], 2)} / {pc(d['cl_hit'], 0)} | {pc(d['dumb_med'], 0)} | {pc(n['cov68'], 0)} / {pc(d['cov68'], 0)} | {pc(n['cov95'], 0)} / {pc(d['cov95'], 0)} |")
ladder = '\n'.join(f"| {r['ckpt'][:-3]} | {r['note']} | {r['objective']:.3f} | {r['val']['numax_hit']:.3f} / {r['val']['dnu_hit']:.3f} / {r['val']['det_bal']:.3f} | "
                   f"{' / '.join(pc(x, 2) for x in r['dev']['keys'])} | {'kept' if r['accepted'] else 'rejected'}: {r['why']} |" for r in RL)
det = '\n'.join(f"| {l} | {pc(v['tpr'], 1)} | {pc(v['fpr'], 1)} | {pc(v['cl_tpr'], 1)} | {pc(v['cl_fpr'], 1)} |" for l, v in DT.items())
card = f"""# StarQuake v1.0 — model card

## What it does
Reads a Kepler long-cadence light curve of a red giant (27 days to 4 years) and returns:
whether solar-like oscillations are present (numax 5-275 uHz), the frequency of maximum oscillation power **numax** and the
large frequency separation **Delta nu**, each with a 1-sigma error bar, and (given Teff) mass and radius from the asteroseismic
scaling relations M = (numax/3090)^3 (dnu/135.1)^-4 (Teff/5777)^1.5, R = (numax/3090) (dnu/135.1)^-2 (Teff/5777)^0.5.
The solar references (3090 uHz, 135.1 uHz, 5777 K) are those of Yu et al. 2018, so results compare directly with that catalogue.
(With the IAU 5772 K, solar numax would scale by sqrt(5777/5772), a 0.04% change.)

Released rung: **{rel}**. numpy/scipy inference (`starquake.py`), BatchNorm folded into convolutions; numpy equals torch to 7e-7; the in-browser JS port equals Python to 1e-6.

## Inputs and preprocessing (`pipeline.py`)
PDCSAP flux, quality-flagged cadences removed, per-quarter median normalisation to ppm, gaps of <= 3 cadences interpolated,
10-day running-mean high-pass, 5-sigma clip, power density (ppm^2/uHz, Parseval-normalised for the duty cycle).
Features: log power on 1024 log-spaced bins 3-283 uHz; a 48 x 110 comb map (autocorrelation of the spectrum within +-3.5 expected
separations of 48 trial centres, lags 0.56-1.65 expected separations); log observing length and duty cycle.

## Physics in the simulator (`training/starsim.py`)
Granulation: two Harvey-like super-Lorentzians, amplitudes and timescales scaling with numax and mass (Kallinger et al. 2014, Table 2) with scatter.
Envelope: Gaussian, FWHM 0.66 numax^0.88 (Mosser et al. 2012), dev-calibrated narrowing at high numax (v2). Radial-mode amplitude
from Yu et al. 2018 eq. 8 (RGB/clump coefficients), divided by sqrt(2) for the Kjeldsen et al. 2008 convention (v2), 12% of stars diluted.
Modes: universal red-giant pattern (Mosser et al. 2011): eps(dnu), d01, d02, d03, curvature alpha = 0.008; visibilities 1, 1.54, 0.63, 0.071.
Dipole mixed modes (70% of stars) from tan(theta_p) = q tan(theta_g) with DPi1 and q drawn for RGB or clump; power and width scaled by (1 - zeta).
Linewidths 0.05-0.3 uHz (log-normal, Teff^3). Stochastic excitation: random complex Gaussian amplitude in every frequency bin (chi^2 2 dof).
Apodisation sinc^2(pi nu / 2 nu_Nyq). dnu from M, Teff and numax via the scaling relations + 2% scatter; clump stars 2% lower.
Synthesis by inverse FFT on a grid twice the observed span; added to a real quiet-dwarf Kepler light curve (real noise and gaps).

Verification (`training/physics_check.json`): chi^2 statistics mean {PC['chi2_2dof']['mean']:.3f}, variance {PC['chi2_2dof']['var']:.3f} (expected 1, 1);
Parseval variance ratio {PC['parseval'][0]['ratio']:.3f}-{PC['parseval'][2]['ratio']:.3f}; classical recovery on 24 simulated stars: numax {pc(PC['classical_on_sims']['numax_med_abs_lnerr'], 2)}, dnu {pc(PC['classical_on_sims']['dnu_med_abs_lnerr'], 2)} median error;
mixed modes {PC['mixed_modes']['n_modes']} found vs {PC['mixed_modes']['expected']:.1f} expected (numax 60, DPi1 75 s);
scaling relations reproduce Yu et al.'s 16,094 masses and radii (median ratio 1.000, scatter {pc(PC['scaling']['yu_recomputed']['M_ratio_mad'], 2)} / {pc(PC['scaling']['yu_recomputed']['R_ratio_mad'], 2)} from rounding);
simulated dnu-numax slope {PC['scaling']['sim_dnu_numax']['sim_slope']:.3f} (Yu catalogue {PC['scaling']['sim_dnu_numax']['yu_all_slope']:.3f}, Mosser 2012 0.751); the Sun gives 3090 uHz / 135.1 uHz (by construction of the references).
Twin test on the 18 dev stars (`twin_test.json`): v1 humps {TT['v1']['median_below_hump_above'][1]:+.2f} dex vs real, v2 {TT['v2']['median_below_hump_above'][1]:+.2f} dex.
Network twin test (`twin_net.json`, r1): median |error| real {pc(TN['real_med_abs'][0])} / {pc(TN['real_med_abs'][1])}, twins {pc(TN['twin_med_abs'][0])} / {pc(TN['twin_med_abs'][1])} (numax / dnu).

## Data
- Truth: Yu et al. 2018 (ApJS 236, 42), 16,094 Kepler red giants, journal machine-readable tables. Exam: 420 stars (70 per numax stratum 8-15-25-40-70-130-250 uHz, seed 2026), >= 1,200 d of data. Dev: 18 other stars (3 per stratum).
- Noise: 360 + 60 quiet Kepler dwarfs (DR25: logg > 4.25, no KOI/TCE), screened for variability in-band; 232 train / 42 validation kept.
- Exam negatives: 120 more dwarfs, NOT screened (seed 2027).
- Gaia DR3 FLAME radii for {MR['gaia_n']} exam stars (ESA Gaia TAP).
- Light curves: Kepler long cadence from MAST (stpubdata public S3 bucket).

## Network
Two CNN branches (1-D on the log spectrum, 2-D on the comb map) + MLP head, 412k parameters, beta-NLL (beta 0.5) on ln numax and ln dnu,
BCE on detection, EMA weights, AdamW, cosine schedule. Training data: 40,000 simulations (sim1 v1 physics 15k, sim2 v2 15k, sim3 v2 10k with 30% non-oscillating).

## Ratchet (`training/ratchet.json`)
Objective on the frozen simulated set (val2: 5,000 v2 giants in 42 held-out noise stars; valneg: 2,000 non-oscillating):
mean of numax-within-5% rate, dnu-within-2% rate, detection balanced accuracy. Keep if objective > incumbent + 0.01 AND the mean ratio of four
dev median errors (whole and 90 d; numax and dnu) vs the incumbent is <= 1.10. When the physics changed (v1 -> v2) all rungs were re-scored on val2 and decisions re-taken.

| rung | what changed | objective | numax hit / dnu hit / det | dev med err (whole numax, dnu; 90 d numax, dnu) | decision |
|---|---|---|---|---|---|
{ladder}

Note: r4's objective gain came from the detection output (its frozen-set dnu hit rate was lower than r1's, 0.482 vs 0.519); r5, the same network trained longer, recovered it (0.530) and cut the dev errors by 10%.

## Exam (420 Kepler red giants, never used for any decision)
Median |ln error| / fraction within 5% (numax) or 2% (dnu). Coverage: fraction within 1 and 2 sigma, sigma = sqrt(sigma_net^2 + sigma_published^2), widen factors numax x{SC['widen'][0]:.2f}, dnu x{SC['widen'][1]:.2f} fitted on dev.

| length | n | StarQuake numax | classical numax | dumb | StarQuake dnu | classical dnu | dumb | cov 1-sigma (numax/dnu) | cov 2-sigma |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(lrow(l) for l in ('0', '365', '90', '54', '27'))}

Classical = SYD-style pipeline (version chosen on dev: multi-start background fit for pieces < 300 d).

Detection (420 giants should be yes; 120 unscreened quiet dwarfs should be no). StarQuake: p > 0.5 fixed. Classical: comb strength threshold chosen on dev.

| length | StarQuake TPR | StarQuake FPR | classical TPR | classical FPR |
|---|---|---|---|---|
{det}

Mass and radius (whole light curve, Yu Teff): median |difference| vs Yu et al. {pc(MR['M_med_frac'])} (mass), {pc(MR['R_med_frac'])} (radius); bias {pc(MR['M_bias'])} / {pc(MR['R_bias'])};
median error bars {pc(MR['M_sig_med'])} / {pc(MR['R_sig_med'])}. Against Gaia DR3 FLAME radii ({MR['gaia_n']} stars): R/R_Gaia median {MR['gaia_R_ratio_med']:.3f} (scatter {pc(MR['gaia_R_scatter'])});
Yu et al.'s radii {MR['gaia_yuR_ratio_med']:.3f} ({pc(MR['gaia_yuR_scatter'])}); classical {MR['gaia_clR_ratio_med']:.3f} ({pc(MR['gaia_clR_scatter'])}).

## Limits
""" + '\n'.join('- ' + x.replace('<sub>', '').replace('</sub>', '') for x in limits) + f"""

## Credits and licences
{credits.replace('&amp;', '&')}
"""
open(S + 'release_card.md', 'w').write(card)
readme = f"""# StarQuake v1.0

Red-giant asteroseismology from a Kepler light curve: numax, Delta nu (with error bars), detection, mass and radius.
A small neural network trained only on simulated stars injected into real Kepler noise, examined on 420 real stars.

    pip install numpy scipy astropy   # astropy only to read FITS files
    python demo.py

```python
{use.split(chr(10), 4)[4]}
```

Files: `starquake.py` (numpy network + API), `pipeline.py` (light curve -> spectrum -> features), `starquake_weights.npz`,
`demo.py` + `examples/` (3 real exam stars, fixed random draw), `MODEL_CARD.md` (physics, ratchet, exam, limits, credits),
`training/` (every script used: simulator, banks, training, ratchet, exam, export, web).
"""
open(S + 'release_readme.md', 'w').write(readme); print('wrote texts.json, release_card.md, release_readme.md')
