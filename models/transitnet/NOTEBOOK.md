> **Lab notebook.** The build notes as recorded during the original work in a cloud sandbox (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that sandbox: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# TransitNet — status (3 Nov 2026, v1.0)

- **What it is:** a network that decides whether a periodic dip in a TESS light curve is a planet, and if so measures it. It was trained on zero real planets. Fourth project in Greg's physics-ML collection, after ChirpNet, SunDoctor and CellSense.
- **Page:** https://claude.ai/artifact/1hdMeCpp4qDXBN3a8B9Ff5
  - Hero: TOI-270 c.
  - Real-star explorer with 10 picks: HD 209458 b, KELT-9 b, pi Men c, LHS 3844 b, HD 219134 b, TOI-700 b, and the false positives TOI-617.01, TOI-236.01, TOI-570.01, TOI-523.01.
  - "Make a dip" lab: four real pool stars; inject a planet, a binary or a background binary. The transit model, folding and network all run in JS; parity with Python is about 1e-7.
  - Exam, ratchet, how it works, use it, limits.
- **Release:** `transitnet-v1.0.zip` (2.9 MB), containing:
  - `transitnet.py` (numpy; `TransitNet().read(...)` and `load_spoc`)
  - `pipeline.py`
  - weights
  - `demo.py` with 3 real stars
  - `MODEL_CARD.md`, `README.md`
  - `training/`
- The cloud workspace (`/home/claude/transit`) is ephemeral. The zip and the page are the durable copies.

## Data (all reachable from the sandbox)

- NASA Exoplanet Archive pscomppars, via the TAP API.
- ExoFOP TOI list (CSV).
- MAST: astroquery for product lists; downloads redirect to the stpubdata S3 bucket.
- 7,778 SPOC 2-minute light curves, about 2 GB, compacted to npz.
- Not reachable: VizieR/CDS (connection reset), so the Claret limb-darkening tables could not be used.
- Real set:
  - 1,625 confirmed planets with P ≤ 25 d, Tmag ≤ 13.5 and a detectable depth, up to 3 sectors each
  - 992 TOIs that follow-up marked false positive (FP), up to 2 sectors each
  - 77 TOIs marked false alarm (FA)
- Dev set: 20 planets and 20 FPs (seed 2026). Their whole host stars are excluded from the exam.
- Noise pool: 2,200 ordinary 2-minute targets, 220 from each of 10 sectors spread across the mission.

## Physics and pipeline

- Exact quadratic limb-darkened occultation (Gauss-Legendre, any p). Matches batman to ≤ 0.94 ppm.
- Eclipsing binaries, background binaries (third light 3–3,000×), Keplerian eccentric orbits, centroid-shift physics.
- Pipeline:
  - running-median detrend that masks the transit
  - per-sector transit-time search, then a linear ephemeris through it (catalogue periods drift; e.g. TOI-270 d's composite period is off by 0.0025 d)
  - fine refit
  - views: local 201, secondary 101, global 401, odd/even 101, centroid 101
  - sibling planets and TOIs cut out
- Limb-darkening prior recalibrated on free fits to the dev planets: median u1+u2 is 0.54, so q1 was changed to 0.15–0.55.

## Ratchet

- **Objective:** 0.5 × (2·AUC − 1) + 0.5 × mean parameter skill, on a frozen 5,000-reading set (`val2`, v2 physics).
- **Dev gate:** log-loss may not be more than 0.05 worse, and the radius-ratio (k) error may not be more than 1 point worse.

| Rung | Change | Result |
|---|---|---|
| 1 | Baseline | Kept |
| 2 | LD prior from dev planets, plus centroids in the data | Rejected: the absolute-k head failed to learn (k error 34%) |
| 3 | Parameters as corrections to the model-free estimates (k from √depth, a/R* from P and T14), on the rung 2 data | **Kept, released** |
| 4 | Centroid branch | Rejected: +0.002 only, needed +0.01 |

## Exam (released rung 3)

Classification:

- 1,196 planets and 832 FPs with SNR ≥ 7.
- AUC 0.835.
- 98% of planets called planet, but 66% of FPs also called planet.
- Breakdown of FPs called planet:

| FP type | Called planet |
|---|---|
| Nearby EB / centroid offset (NEB) | 64% |
| Eclipsing or spectroscopic binary (EB/SB) | 62% |
| Other | 71% |

- Some real planets get scored low: TOI-2109 b at 0.00, TOI-1518 b at 0.30.
- TOI-617.01, a nearby-EB false positive, is called a background binary at 98%.

Parameters (median error) by SNR:

| SNR | Rp/R* | √depth baseline | b | a/R* | Rp/R* in 68% interval (pub. error added) |
|---|---|---|---|---|---|
| 100+ | 1.6% | 4.4% | 0.07 | 4.2% | — |
| 20–100 | 3.5% | 5.7% | — | — | — |
| 7–20 | 6.7% | 6.8% | — | — | — |
| All | — | — | — | — | 70% (b 79%, a/R* 71%) |

- Least-squares fit on the same folded data (0.8 s per star): Rp/R* 2.3% / 4.7% / 10.7% for SNR 100+ / 20–100 / 7–20. TransitNet beats it everywhere except a/R* at high SNR (3.7% vs 4.2%).

## Lessons

- Calibrating the simulator on dev units worked again: the limb-darkening prior came from free fits to the dev planets.
- Predicting corrections to model-free estimates beat predicting absolute values. It removed a 3–5% k bias seen even on simulated data.
- Catalogue ephemerides need per-sector alignment.
- Photometry alone cannot rule out most TOI false positives: they already passed SPOC vetting.

## Next ideas

- A physical planet-size cap, low-mass stellar companions and R* as an input, to target the SB1-type false positives.
- Difference-image centroids rather than MOM_CENTR.
- Claret LD tables if a mirror becomes reachable.
- Single transits.
- A "find it" step (in-browser BLS).
