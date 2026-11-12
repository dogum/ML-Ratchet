import _paths as _P  # portable paths (see _paths.py)
import json, numpy as np
R = json.load(open(_P.WORK + '/ratchet.json')); S = json.load(open(_P.WORK + '/release_summary.json')); D = json.load(open(_P.LAB + '/data.json'))
P = json.load(open(_P.LAB + '/pe.json')); BR = json.load(open(_P.WORK + '/bns_rungs.json')); B = json.load(open(_P.LAB + '/bns.json'))
rows, k = [], 0
for r in R:
    if r.get('rejudged'): num = f'{k} (re-judged)'
    else: k += 1; num = str(k)
    v = r['val']; e = v['eff']; g = lambda s: e.get(str(s), e.get(s))
    sf = v.get('single_fa'); sfs = f"{100 * np.mean([sf[x] for x in sf if int(x) >= 12]):.0f}%" if sf else 'not measured'
    rows.append(f"| {num} | {r['note'].split(':')[0]} | {'kept' if r['accepted'] else 'rejected'} | {100 * g(8):.0f}% | {100 * g(10):.0f}% | {100 * v['mc_relerr_med']:.1f}% | {v['mc_cov68']:.2f} | {100 * r['sens_glitch']:.0f}% | {sfs} | {r['val']['sens'] + r['sens_glitch'] - 2 * r['val']['mc_relerr_med']:.3f} |")
ev = []
for e in D['events']:
    h = e['hit']
    ev.append(f"| {e['label']} | {h['score']:.1f} | {h['coincidence']:.1f} | {h['h1_alone']:.1f} / {h['l1_alone']:.1f} | < {h['far_per_hour']:.2f} | {h['chirp_mass']:.1f} [{h['chirp_mass_68'][0]:.1f}, {h['chirp_mass_68'][1]:.1f}] | {e['mc_det_ref']:.1f} [{e['mc_det_lo']:.1f}, {e['mc_det_hi']:.1f}] | {1000 * (h['gps'] - e['gps']):+.0f} ms |")
pe_rows = []
lab = {'GW150914': 'GW150914', 'LVT151012': 'GW151012', 'GW151226': 'GW151226', 'GW170104': 'GW170104'}
for name, p in P['events'].items():
    s, ref = p['summary'], p['ref']; f = lambda k, d=1: f"{s[k][1]:.{d}f} [{s[k][0]:.{d}f}, {s[k][2]:.{d}f}]"
    pe_rows.append(f"| {lab[name]} | {f('m1_source')} | {ref['m1']} | {f('m2_source')} | {ref['m2']} | {f('chirp_mass_source')} | {ref['mc'][0]} | {f('chi_eff', 2)} | {ref['chi'][0]:+.2f} | {f('luminosity_distance_mpc', 0)} | {ref['dl']} |")
pp = P['pp']['maxdev']
brows = []
for i, r in enumerate(BR):
    v = r['val']; e = v['eff']
    brows.append(f"| {i + 1} | {r['note'].split(':')[0]} | {'kept' if r['accepted'] else 'rejected'} | {100 * e['10']:.0f}% | {100 * e['12']:.0f}% | {100 * e['15']:.0f}% | {100 * e['20']:.0f}% | {100 * v['mc_relerr_med']:.1f}% | {r['objective']:.3f} | {r['real']['score']:.1f} vs {r['real']['bg_max']:.1f} | {r['real']['mc']:.4f} |")
bk = next(r for r in reversed(BR) if r['accepted'])
card = f"""# ChirpNet model card (v2.0)

ChirpNet v2 is four small networks, all trained only on simulations and tested on real LIGO data:

| File | What it does | Input |
|---|---|---|
| `chirpnet_weights.npz` | black-hole **finder** (rung 4): detection statistic, chirp mass ± 68%, mass ratio, merger time | 2 s, 2 × 4096 samples at 2048 Hz |
| `chirpnet_coinc_weights.npz` | **coincidence checker** (rung 8, two-stream): confirms both detectors see the same chirp | same |
| `chirpnet_pe_weights.npz` | **posterior** network (DINGO-style normalizing flow): samples of chirp mass, mass ratio, χ_eff, distance, inclination | same + both noise spectra |
| `chirpnet_bns_weights.npz` | **neutron-star** detector and chirp-mass estimator (time-frequency 2-D CNN) | 16 s, 2 × 16384 samples at 1024 Hz |

Finder plus checker together are the released detector, rung 9 ({D['release']['params']:,} weights). `chirpnet.py` (numpy and scipy only)
runs all four. `demo.py` reproduces the numbers below. `training/` holds every script that produced the weights (needs PyTorch and LALSuite).

## 1. Black-hole detector

**Pipeline.** Raw 4096 Hz strain is whitened with its own Welch PSD (4 s Hann segments, median average), band-limited to
20-900 Hz, resampled to 2048 Hz and normalized to unit-variance noise. Excursions beyond 25σ are gated out, as in LIGO's
own searches. The finder slides over the data in 1/16 s steps. Wherever it scores above 0 the checker also runs, and if
the checker scores below -1.0 the window's statistic drops by 30. That only ever lowers a score, so every value above 0 is exact.

**Training data (all simulated).** 8,000 aligned-spin IMRPhenomD binary black holes. Chirp mass is log-uniform in
5.5-45 M☉ (detector frame), q in 0.2-1, spins in [-0.85, 0.85]. Each is projected onto H1/L1 for a random sky position
with the real light-travel delay, whitened with O1/O2 and design noise spectra, and dropped into fresh band-limited
Gaussian noise. Loud sine-Gaussian glitches go into one detector in 30-35% of windows. From rung 6 on, some negatives
hold a chirp in one detector only. Across all rungs: {D['release']['samples']:,} training windows.

### The ratchet

Every candidate was scored on a frozen validation set (20,000 noise windows to set a false-alarm probability of 1e-3,
7,200 held-out signals at SNR 5-20, a glitchy version of both, and one-detector chirps). Real LIGO events never chose anything.

- **Rule 1** (rungs 1-3): Gaussian sensitivity must rise. It wrongly rejected rung 3, which had gained 55 points of glitch robustness.
- **Rule 2** (rungs 3-5): one objective, Gaussian sensitivity + glitchy sensitivity − 2 × median chirp-mass error, must rise by 0.01.
- **Rule 3** (rung 6): a stress test on time-slid real data showed that a loud chirp in one detector alone fools rung 4.
  A fix must halve one-detector false alarms and may not drop the objective by more than 0.01.
- **Rule 4** (rungs 7-9): as rule 3, but counting only loud (SNR ≥ 12) one-detector chirps, since quiet ones are ambiguous.
  Rungs 6-8 each cut the fakes to a few percent but lost too much sensitivity. Rung 9 splits the job: the rung 4 finder
  proposes and the rung 8 checker confirms. Its veto level (-1.0) was chosen on simulations with a fresh random seed:
  the most sensitive level whose loud-fake rate was at most a quarter of rung 4's.

| Rung | Change | Decision | Found at SNR 8 | Found at SNR 10 | Chirp-mass error (median, SNR ≥ 10) | 68% coverage | Found with glitches | Loud one-detector fakes flagged | Objective |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

### Final exam: real LIGO data

The four events use 32 s GWOSC recordings. For the background, L1 is slid against H1 in 0.25 s steps from 3 to 25 s,
and windows near the event in either detector are removed. That gives {S['bg_n']:,} windows and {S['live_s'] / 3600:.1f} h of livetime;
the loudest statistic was {S['bg_max']:.2f}. The release threshold is {S['threshold']:.1f}. The "alone" scores are the full
pipeline's statistic with the partner detector slid by 5, 9 or 13 s (the maximum of the three).

| Event | Statistic | Coincidence (checker) | H1 alone / L1 alone | FAR per hour (upper limit) | ChirpNet chirp mass, 68% | LIGO GWTC-1, detector frame, 90% | Merger time vs LIGO |
|---|---|---|---|---|---|---|---|
{chr(10).join(ev)}

## 2. Posteriors (DINGO-style neural posterior estimation)

A ChirpNet trunk (started from rung 4) embeds the 2 s window. Each detector's noise spectrum is summarized as its ASD at
10 frequencies from 25 to 560 Hz. Together they condition an 8-layer RealNVP flow over (log ℳ, q, χ_eff, log₁₀ d_L,
cos ι, t_c). Training draws use physical amplitudes: waveforms are scaled from 400 Mpc, distances follow a uniform-in-volume
prior out to each waveform's horizon, PSDs are jittered by 0.6-1.6×, and only draws with network SNR ≥ 6 are kept
(4,000 steps × 128). Source-frame masses use the sampled distance and the Planck 2015 cosmology. For the real events
the window is placed using ChirpNet's own merger time.

**Calibration** (P-P test, 400 held-out injections): largest gap between the rank CDF and the diagonal is
log ℳ {pp['log_mc']}, q {pp['q']}, χ_eff {pp['chi_eff']}, d_L {pp['log10_dL']}, cos ι {pp['cos_iota']}. Chance alone allows about 0.07.
The merger-time posterior ({pp['t_c']}) is too wide to use; it is not reported.

| Event | m₁ (90%) | LIGO | m₂ (90%) | LIGO | ℳ source (90%) | LIGO | χ_eff (90%) | LIGO | d_L Mpc (90%) | LIGO |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(pe_rows)}

Every LIGO median falls inside ChirpNet's 90% interval.

## 3. Neutron stars

**Pipeline.** Minutes of 4096 Hz strain are whitened with an 8 s-segment Welch PSD, band-limited to 25-500 Hz,
resampled to 1024 Hz, and gated at 25σ. The model reads 16 s windows. A fixed front end computes power in 0.375 s Hann
frames every 1/16 s (2.7 Hz resolution, long enough to integrate a slowly sweeping chirp coherently). It averages the power
into 64 log-spaced bands from 24 to 450 Hz and takes log(1 + power). A 2-D residual CNN (6 stages, 16→96 channels) does the rest.

**Training data.** 3,000 IMRPhenomD_NRTidalv2 binary neutron stars: detector-frame chirp mass 0.85-2.0 M☉, q 0.6-1,
component masses 0.95-2.6 M☉, tidal deformabilities 0-1500, aligned spins up to 0.05. The merger is placed 8-15.8 s
into the window. Glitches go into 20% of windows. Validation uses 600 held-out waveforms.

**Ratchet objective:** Gaussian sensitivity (SNR 6-12) − 2 × median chirp-mass error; glitch false alarms ≤ 5%.

| Rung | Change | Decision | Found at SNR 10 | 12 | 15 | 20 | Chirp-mass error | Objective | GW170817 statistic vs loudest background | GW170817 ℳ |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(brows)}

**Final exam: GW170817.** The 2048 s cleaned H1 and L1 data products (LIGO-P1700349) are used. Background comes from
L1 slid circularly against H1 by 12 shifts of 29-503 s, with windows holding the event in either detector removed:
{B['bg_n']:,} windows over {B['bg_live_s'] / 3600:.1f} h, loudest {B['bg_max']:.2f}. The released model scores GW170817 at **{B['peak']['score']:.1f}**
with chirp mass **{B['mc_combined']:.4f} M☉** (LIGO: 1.1975 +0.0001/−0.0001, detector frame) and merger time
{1000 * B['peak']['t_merger_rel']:+.0f} ms from LIGO's. The raw Livingston data (LIGO-P1700337) still carries the famous glitch
1.1 s before merger. With automatic gating it scores {B['raw_glitch']['score']:.1f} (ℳ {B['raw_glitch']['mc']:.4f}); with the glitch left in it scores {B['raw_nogate']:.1f}.

## Limits

- **Coincidence is not perfect.** The checker lets about 1 in 6 loud one-detector chirps through in simulation, and
  GW151226's Hanford signal alone still scores above the threshold. The neutron-star model has no checker at all
  (H1 alone {B['single']['h1_alone']:.1f}, L1 alone {B['single']['l1_alone']:.1f}, both above its background).
- Aligned spins only; no precession or higher-order modes. The black-hole model covers chirp masses of about 6-45 M☉ (detector frame).
- False-alarm rates are bounded by hours of time slides ({S['live_s'] / 3600:.0f} h for black holes, {B['bg_live_s'] / 3600:.0f} h for neutron stars).
  They are upper limits, nowhere near the significance of LIGO's production searches.
- The neutron-star chirp mass is good to about 2% on simulations, versus LIGO's 0.01%, because 16 s windows give up the early
  inspiral. On GW170817 the released rung is 5% low and its 68% interval misses LIGO's value; rung 2 had been within 1.3%.
- All five real events are from O1/O2 (2015-2017). Untested on O3/O4 data.
"""
open(_P.WORK + '/release/MODEL_CARD.md', 'w').write(card)
print('card ok', len(card))
