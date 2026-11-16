<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# ChirpNet <sub>v2.1</sub>

> **Neural networks that never saw a real gravitational wave, then found five.**

*Is there a black-hole or neutron-star merger in this LIGO data, and how heavy was it?*

Four small networks trained only on simulated mergers from general relativity. Pointed at real recordings from the two LIGO observatories they find the black-hole and neutron-star mergers, weigh them, and give calibrated posterior ranges. Since v2.1 a physics template bank (a TaylorF2 matched filter) refines the neutron-star chirp mass after the network finds the merger. Everything runs on plain numpy and scipy.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/chirpnet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/chirpnet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Whitened strain from LIGO Hanford and Livingston (GWOSC HDF5, 4 kHz). |
| **Returns** | Detection statistic and false-alarm rate (upper limit from time slides)<br>Chirp mass ± 68%, mass ratio, merger time<br>Posterior samples of masses, effective spin, distance and inclination (DINGO-style flow)<br>A separate neutron-star detector; its chirp mass refined by a matched filter (v2.1) with a 68% range |
| **Learned from** | Aligned-spin waveforms from general relativity (IMRPhenomD for black holes, IMRPhenomD_NRTidalv2 for neutron stars via LALSuite), projected onto the two detectors with the real light-travel delay and whitened with O1/O2 and design noise spectra. |
| **Noise** | Simulated Gaussian noise coloured by O1/O2 and design spectra; the exam background is time-slid real LIGO data. |
| **Network** | 1-D residual CNN finder + two-stream coincidence checker (882k weights), an 8-layer RealNVP posterior flow, and a 2-D time-frequency CNN for neutron stars, followed (v2.1) by a TaylorF2 matched filter for their chirp mass |
| **Physics also enters** | output: v2.1: the neutron-star chirp mass is a TaylorF2 matched-filter fit seeded by the network's reading |
| **Real data's jobs** | gate, error bars, exam (never training labels) |
| **Dev set** | None of the real events. Rules were set on simulations and time-slid noise. v2.1: real O2 and O3 noise from times away from every catalogued event (seeded GWOSC bulk files), used for injections and background, never the exam files. |
| **Exam** | GW150914, GW151012, GW151226, GW170104 (32 s each) and GW170817 (2,048 s) |

## Run it

```bash
cd models/chirpnet
pip install numpy scipy h5py
python demo.py        # about 70 s
```

or from anywhere in the repo: `mlratchet demo chirpnet`.

```python
import chirpnet
net = chirpnet.ChirpNet("chirpnet_weights.npz")        # loads the coincidence checker too
h1, t0 = chirpnet.load_gwosc("examples/H-H1_LOSC_4_V2-1126259446-32.hdf5")
l1, _  = chirpnet.load_gwosc("examples/L-L1_LOSC_4_V2-1126259446-32.hdf5")
hits = net.scan(h1, l1, gps_start=t0)
post = chirpnet.ChirpPE("chirpnet_pe_weights.npz").weigh(h1, l1, t0, hits[0]["gps"])
```

## The exam

Thirteen real events — five from O1/O2 and eight from O3, the O3 set named in advance — plus hours of time-slid real background from the same files to bound false alarms. Every number here is on real data that never chose anything.

| Metric | ChirpNet | Baselines | |
|---|---|---|---|
| Black-hole events found (O1/O2)<br><sub>Faintest scores 9.6; the loudest of 57,760 time-slid background windows scores 6.2.</sub> | **4 / 4** | — | ✅ wins |
| Chirp mass within 10% of LIGO | **4 / 4** | LIGO GWTC-1 (published): reference | ✅ wins |
| LIGO posterior medians inside ChirpNet's 90% range | **all** | — | ✅ wins |
| GW170817 statistic vs loudest background (6.2 h) | **14.4** | Loudest time-slid background (dumb baseline): 5 | ✅ wins |
| GW170817 chirp mass (detector frame, M☉)<br><sub>v2.1, matched filter after the network: +0.002% from LIGO, 68% range [1.19753, 1.19877] (one-sided: its lower edge sits on the estimate and misses LIGO's median by 0.00003). v2.0 (network alone), re-scored: 1.1365 [1.1285, 1.1444], 5.1% low. On dev injections at chirp mass 1.1–1.3 the network alone was 1.2% off, so dev did not reproduce the 5% miss; the matched filter was chosen on its general dev gain (median error 1.78% → 0.027%).</sub> | **1.19753** | LIGO (published): 1.1975 | ✅ wins |
| Loud one-detector fakes passed | **17%** | Finder alone (rung 4) (classical): 99.6% | ✅ wins |
| O3 black-hole events found, within the training range (7 events)<br><sub>Statistics 9.9–14.1. Named in advance; data first reached on 12 Nov 2026 (GWOSC could not be reached from the shared build machine).</sub> | **7 / 7** | Loudest of 29,879 time-slid O3 background windows (dumb baseline): 7.24 | ✅ wins |
| O3 chirp mass inside ChirpNet's 68% range<br><sub>Within 4% on six events; GW200224_222234 reads 15% low (35.0 vs 41.1).</sub> | **6 / 7** | GWTC-2.1 / GWTC-3 (detector frame) (published): reference | ✅ wins |
| O3 catalogue medians inside ChirpNet's 90% posterior ranges<br><sub>m1, m2, chirp mass (source frame), effective spin and distance for seven events.</sub> | **35 / 35** | — | ✅ wins |
| GW190521 (detector-frame chirp mass 98.7, far above the 45 the model trained on)<br><sub>Statistic −0.6. Outside the training range, named in advance anyway.</sub> | **missed** | LIGO/Virgo search (published): found | ❌ loses |
| O3 time-slid background windows above the O1/O2 threshold (7.0)<br><sub>The loudest O3 background window scores 7.24 in 1.04 h. Dev (10.6 h of non-exam O3 noise, 15 Nov 2026) shows it is worse: 46 windows at or above 7.0, the loudest 10.88. Re-setting the threshold by the same rule would put it at 11.5, a sensitivity cost that cannot be measured without a simulation bank; not released. Unchanged in v2.1.</sub> | **1 of 29,879** | O1/O2 background (57,760 windows) (dumb baseline): 0 | ❌ loses |

### Where it wins

- Finds all four O1/O2 black-hole events with a clear gap above real background.
- Posteriors are calibrated (P-P within chance) and contain every GWTC-1 median.
- The cascade cuts loud single-detector fakes from 99.6% to 17%.
- Unchanged since O1/O2, it finds all seven in-range O3 events named in advance, and every O3 catalogue median lands inside its 90% posteriors.

### Where it loses

- GW170817: the network alone reads the chirp mass 5% low with too narrow an interval; v2.1's matched filter fixes the value, but its 68% range is one-sided and misses LIGO's median by 0.00003.
- GW151226's Hanford signal alone still scores above threshold (so do GW190408's and GW200311's in O3).
- Misses GW190521, a merger heavier than anything in its training set.
- One O3 exam background window (7.24) clears the 7.0 threshold set on O1/O2 noise; on 10.6 h of non-exam O3 noise 46 windows do (loudest 10.88).

## Limits

- Aligned spins only; no precession or higher modes.
- False-alarm rates rest on hours of time slides, so they are upper limits.
- The neutron-star model has no coincidence check yet.
- The neutron-star matched filter has no spin or tidal templates; its 68% range is the raw likelihood range widened ×52 on simulated injections, and it needs minutes of raw strain.
- No network can be retrained on this machine: the LALSuite waveform banks were never kept and LALSuite has no Windows build.
- Real events tested come from O1–O3 (2015–2020); O4 is untested.
- Very heavy mergers (detector-frame chirp mass above ~45) are outside the training range: GW190521 is missed.

## The ratchet

### Black holes

**Objective:** Gaussian sensitivity + glitchy sensitivity − 2 × median chirp-mass error (rule 2 onward)  
**Rule:** The rule changed in the open, three times. Rule 1: Gaussian sensitivity must rise. Rule 2: the single objective must rise by 0.01. Rule 3: halve one-detector false alarms without losing more than 0.01 of objective. Rule 4: as rule 3, counting only loud (SNR ≥ 12) one-detector chirps.

| Rung | Change | Frozen-set score | Decision | Why | Found at SNR 10 | Chirp-mass error | Found with glitches | Loud one-detector fakes passed |
|---|---|---|---|---|---|---|---|---|
| r1 | Baseline 6-layer residual CNN, SNR 6–30, Gaussian noise | 0.199 | ● kept | First rung. | 71% | 11.6% | 0% | — |
| r2 | SNR curriculum: start loud, anneal to the threshold | 0.477 | ● kept | Sensitivity and chirp mass both improved. | 89% | 5.8% | 0% | — |
| r3 | Glitch training: loud glitches in one detector in 35% of windows | 1.029 | ○ rejected | Rule 1 (Gaussian sensitivity must rise) said no. | 88% | 5.3% | 55% | — |
| r3b | Rung 3 re-judged under rule 2 (one objective that also counts glitch robustness) | 1.029 | ● kept | Objective 1.029 vs 0.477. | 88% | 5.3% | 55% | — |
| r4 | Polish: longer, gentler training near the threshold, glitches on | 1.105 | ● kept | Objective 1.105 vs 1.029. | 91% | 4.9% | 58% | 100% |
| r5 | Test-time averaging (3× compute) | 1.109 | ○ rejected | +0.004, short of the margin. | 91% | 4.8% | 59% | — |
| r6 | Coincidence training: one-detector chirps labelled as noise | 0.721 | ○ rejected | Cut one-detector fakes to 3.5% but lost too much sensitivity (rule 3). | 61% | 5.5% | 41% | 2% |
| r7 | Loud singles only as negatives | 0.726 | ○ rejected | Same trade-off (rule 4). | 59% | 5.4% | 42% | 1% |
| r8 | Two-stream network: one encoder per detector plus an interaction stage | 0.953 | ○ rejected | Fakes to 1%, sensitivity still down. | 80% | 5.3% | 52% | 1% |
| r9 | Cascade: rung 4 proposes, rung 8 confirms both detectors agree | 1.116 | **◆ released** | Objective 1.116 vs 1.105, loud one-detector fakes 17% vs 99.6%. | 90% | 4.9% | 60% | 17% |

<sub>Columns after *Why*: Frozen simulated validation set.</sub>

Rule 1 wrongly rejected rung 3, which had gained 55 points of glitch robustness; rule 2 re-judged it.

### Neutron stars

**Objective:** Gaussian sensitivity (SNR 6–12) − 2 × median chirp-mass error, with glitch false alarms ≤ 5%  
**Rule:** Keep if the objective rises by more than 0.01 and glitch false alarms stay at or below 5%.

| Rung | Change | Frozen-set score | Decision | Why | Found at SNR 15 | Chirp-mass error | GW170817 vs background | GW170817 ℳ |
|---|---|---|---|---|---|---|---|---|
| b1 | Plain CNN stretched to 16 s windows at 1024 Hz | -0.286 | ● kept | First rung. | 17% | 14.9% | 10.8 vs 12.9 | 1.460 |
| b2 | Time-frequency front end (0.375 s frames, 64 log bands) + 2-D CNN | 0.054 | ● kept | Objective 0.054 vs −0.286. | 94% | 2.6% | 11.1 vs 5.5 | 1.182 |
| b3 | Polish at lower learning rate | 0.093 | **◆ released** | Objective 0.093 vs 0.054. | 97% | 2.2% | 14.4 vs 5.0 | 1.137 |

<sub>Columns after *Why*: Simulations, plus the GW170817 exam reported after the fact (never used to choose).</sub>

### O3 release threshold (15 Nov 2026, not released)

**Objective:** False alarms on real O3 dev noise (not exam data): time-slid windows at or above the release threshold  
**Rule:** Fixed before scoring (NOTEBOOK.md, 15 Nov 2026): if any O3 dev background window scores at or above 7.0, re-set the threshold by the rule that set 7.0 (loudest background + 0.5, rounded up to the next 0.5) over the O1/O2 and O3 dev backgrounds. A release must also beat the current one on dev; the network is unchanged and its sensitivity at a new threshold cannot be measured here (no simulation bank), so that part cannot be shown.

| Rung | Change | Frozen-set score | Decision | Why | Threshold | Dev windows ≥ threshold | Loudest dev window |
|---|---|---|---|---|---|---|---|
| t1 | Released threshold 7.0 (set on O1/O2 time slides), scored on O3 dev noise | — | ● kept | The failure shows on dev: 46 of 306,176 windows (10.6 h) at or above 7.0; the loudest 10.9. | 7.0 | 46 | 10.88 |
| t2 | Threshold re-set by the 7.0 rule over O1/O2 + O3 dev background | — | ○ rejected | The rule gives 11.5 (loudest dev window 10.88 + 0.5). Its cost in sensitivity cannot be measured without a simulation bank, so it cannot be shown to beat 7.0 on dev; nothing released. | 11.5 | 0 | 10.88 |

<sub>Columns after *Why*: O3 dev background (306,176 time-slid windows, 10.6 h).</sub>

### Neutron-star chirp mass, matched filter (15 Nov 2026)

**Objective:** Neutron-star ladder objective (sensitivity − 2 × median chirp-mass error); detection is unchanged, so only the error term moves. Scored on 120 TaylorF2 injections (spins, tides) in real O2 noise that is not exam data, because bns_val cannot be rebuilt without LALSuite  
**Rule:** Fixed before scoring (NOTEBOOK.md, 15 Nov 2026): keep if, on the even-numbered injections, the matched filter's median chirp-mass error is lower than the network's by more than 0.005 (0.01 of objective) AND its 68% range, after one widen-only factor fitted on the odd-numbered injections, covers 60–80%.

| Rung | Change | Frozen-set score | Decision | Why | Median chirp-mass error | 68% coverage | Median half-width |
|---|---|---|---|---|---|---|---|
| b3 | Released network alone (v2.0), re-scored on the dev injections | — | ● kept | Incumbent: 108 of 120 found. | 1.78% | 45% | 1.36% |
| m1 | Network finds the merger; a TaylorF2 matched filter (30–1000 Hz, 128 s, profiled over η) refines the chirp mass; range widened ×52.0 (fitted on the odd injections) | — | **◆ released** | Error 1.78% → 0.027% (objective +0.035); coverage 73%. The GW170817-specific dev condition (network error > 2% or coverage < 50% at chirp mass 1.1–1.3) was not met: 1.16%, 58%. | 0.027% | 73% | 0.10% |

<sub>Columns after *Why*: Dev injections in real O2 noise (training/bns_mf_dev.py); check half (55 found).</sub>

Raw per-rung logs are in [`training/`](training/).

## Physics

Aligned-spin waveforms from general relativity (IMRPhenomD for black holes, IMRPhenomD_NRTidalv2 for neutron stars via LALSuite), projected onto the two detectors with the real light-travel delay and whitened with O1/O2 and design noise spectra.

**Simulator**

- 8,000 binary black holes, chirp mass 5.5–45 M☉ (detector frame), q 0.2–1, spins ±0.85
- 3,000 binary neutron stars with tides, chirp mass 0.85–2.0 M☉
- Loud sine-Gaussian glitches in one detector in 30–35% of windows
- From rung 6, negatives that hold a chirp in one detector only

**Checked against independent references**

- Posterior P-P test on 400 held-out injections — largest gaps 0.025–0.06 (chance allows about 0.07)
- Matched filter (v2.1) on 108 found TaylorF2 injections with spins and tides in real O2 noise: median chirp-mass error 0.034% (network alone 1.91%)

## Next rungs

- Extend the training bank to heavier mergers (GW190521-like) and re-run the ratchet (needs LALSuite on Linux or macOS).
- O3: ship the O3 dev background so scan() reports O3 false-alarm rates from O3 noise, then retrain with O3-like glitches next to gates.
- A coincidence checker for the neutron-star model.
- Spin-aligned templates for the matched filter, and a range from a likelihood over spin instead of a widening factor.
- Live posterior sampling in the browser.

## Files

| Path | What |
|---|---|
| `chirpnet.py` | Inference — numpy/scipy only |
| `chirpnet_weights.npz`, `chirpnet_coinc_weights.npz`, `chirpnet_pe_weights.npz`, `chirpnet_bns_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Gravitational Wave Open Science Center (LIGO/Virgo strain)](https://gwosc.org) | exam | CC BY 4.0 |
| [GWOSC O2 and O3 bulk strain (4 kHz, 4096 s files away from every catalogued event)](https://gwosc.org/data/) | dev | CC BY 4.0 |
| [GW170817 data products (LIGO-P1700349, LIGO-P1700337)](https://dcc.ligo.org/LIGO-P1700349/public) | exam | CC BY 4.0 |
| [GWTC-1 published parameters](https://arxiv.org/abs/1811.12907) | truth | Citation |
| [LALSuite waveforms (IMRPhenomD, IMRPhenomD_NRTidalv2)](https://git.ligo.org/lscsoft/lalsuite) | physics | GPL-2.0 (used for training only; not a runtime dependency) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
