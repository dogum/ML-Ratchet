> **Lab notebook.** The build notes as recorded during the original build in a cloud project (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that build machine: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# ChirpNet & Collision Course — status (30 Oct 2026)

## ChirpNet v2.0 (gravitational-wave ML)
- What it is: small neural networks trained only on simulated mergers (LALSuite waveforms in simulated noise), tested on real LIGO data. A portable numpy release, plus a showpiece page.
- Page: https://claude.ai/artifact/Kd3TYXDnxFYFM1zTonrUc7 (version 2).
- Two zips were delivered from the cloud session. The cloud workspace (`/home/claude/gw`) is ephemeral, so these zips and the page are the durable copies.
  - `chirpnet-v2.zip` (16 MB): the release.
  - **`chirpnet-repo.zip` (21 MB): GitHub-ready snapshot of everything.** It contains:
    - `chirpnet/`: release code, demo and 4 weight files.
    - `data/`: the BBH event files; README gives GW170817 links.
    - `training/`: every script, with ratchet logs in `logs/`.
    - `web/`: page sources.
    - `docs/index.html`: the built page, ready for GitHub Pages.
    - `README.md`, `MODEL_CARD.md`, `.gitignore`.
- Release contents:
  - `chirpnet.py` (numpy/scipy only) and `demo.py`.
  - Four weight files: black-hole finder (rung 4), two-stream coincidence checker (rung 8), DINGO-style posterior flow, neutron-star time-frequency CNN.
  - `MODEL_CARD.md`.
- Results:
  - **Black holes (rung 9 = finder + checker cascade):** all 4 O1/O2 events found. Faintest 9.6 vs loudest real background 6.2. Chirp mass within 10% of LIGO.
  - **One-detector fakes:** loud ones cut from 99.6% to 17%. Not eliminated: GW151226's Hanford signal alone still scores above threshold.
  - **Posteriors:** every GWTC-1 median falls inside the 90% ranges. P-P max gaps are 0.025–0.06 (chance about 0.07). The merger-time posterior is too wide and is not shown.
  - **GW170817:** data from LIGO DCC P1700349 (cleaned 2048 s) and P1700337 (raw L1 with the glitch).
    - BNS rung 3 (released) scores 14.4 vs a loudest background of 5.0 over 6.2 h, with merger time 4 ms off.
    - Its chirp mass, 1.136, is 5% low vs 1.1975, with an overconfident interval. Rung 2 got 1.182, but the ratchet never looks at real events.
    - The BNS model has no coincidence check yet.
- Ratchet rules (black holes):
  1. Gaussian sensitivity must rise.
  2. Objective = sensitivity + glitchy sensitivity − 2 × chirp-mass error.
  3. Halve one-detector fakes without losing more than 0.01 of objective.
  4. Like rule 3, but counting only loud fakes. Rung 9 was accepted at 1.116 vs 1.105.
- BNS ratchet objective: sensitivity − 2 × chirp-mass error, with glitch false alarms ≤ 5%.

## Plans (owner, 30 Oct)
- **Open-source it on the owner's GitHub** so others can find and use it. Start from `chirpnet-repo.zip`.
- To-dos before publishing:
  - Choose a code license (README has a TODO; the LIGO data stays CC BY 4.0).
  - Replace the hardcoded `/home/claude/gw` paths in `training/`.
  - Optionally enable GitHub Pages on `docs/`.
- **Artifact Shelf** (https://github.com/dogum/artifact-shelf): ChirpNet should go on it, possibly as a shelf-styled version of the page, alongside Collision Course. Only on the owner's decision. The GitHub API was not enabled for that repo in the cloud session, so shelving needs repo access or a local machine.
- The project may move into the artifacts-collection or public-projects folder. The work overlaps with both.

## Blocked / next
- **O3 final exam:** the GWOSC servers are blocked from the cloud sandbox. The owner needs to attach H1+L1 files (32 sec, 4 KHz, HDF5) for GW190412, GW190408_181802, GW190828_063405, GW191204_171526, GW200129_065458, GW200224_222234, GW200311_115853 and GW190521.
- Ideas next:
  - A coincidence checker for the BNS model.
  - Longer BNS windows, to fix the chirp-mass bias.
  - Live posterior sampling in the browser.

## Collision Course (WebGL2 galaxy-collision simulator)
- https://claude.ai/artifact/T2b4hUR7A5CVLogNbfkm3T (v2 adds the Local Group scenario).

- The owner enjoys learning outside their field. Explain in plain language, and keep visuals front and center.

## O3 exam — 12 Nov 2026 (in the repository, from a home connection)

GWOSC, blocked from the cloud sandbox, answered from a home connection (v2 API). `training/o3_exam.py` ran the
released v2 cascade and posterior network, unchanged, on the eight O3 events named on 30 Oct, with the O1/O2
exam's protocol (32 s, 4 kHz, H1 + L1; time-slid background from the same files, event windows removed).

| Event | Catalogue | Statistic | Found | Chirp mass (det.) | Catalogue | In 68% |
|---|---|---|---|---|---|---|
| GW190408_181802 | GWTC-2.1 | 11.07 | yes | 24.5 | 23.9 | yes |
| GW190412 | GWTC-2.1 | 12.41 | yes | 15.4 | 15.3 | yes |
| GW190521 | GWTC-2.1 | −0.63 | **no** | — | 98.7 (outside training range) | — |
| GW190828_063405 | GWTC-2.1 | 14.11 | yes | 33.5 | 33.9 | yes |
| GW191204_171526 | GWTC-3 | 12.60 | yes | 10.0 | 9.7 | yes |
| GW200129_065458 | GWTC-3 | 12.35 | yes | 31.4 | 32.1 | yes |
| GW200224_222234 | GWTC-3 | 9.89 | yes | 35.0 | 41.1 | **no** (−15%) |
| GW200311_115853 | GWTC-3 | 11.85 | yes | 32.8 | 32.7 | yes |

- Background: 29,879 time-slid O3 windows, 1.04 h; loudest 7.24 — one window above the 7.0 threshold set on O1/O2.
- Posteriors: all 35 catalogue medians (m1, m2, source-frame chirp mass, χ_eff, distance × 7 events) inside the 90% ranges.
- Merger times are 24–100 ms from the catalogue's (geocentric) GPS times, looser than in O1/O2.
- One-detector views still clear the threshold for GW190408 (both), GW190412 (H1), GW191204 (L1) and GW200311 (both):
  the known limit of the coincidence check.

## GW170817 live, and a gating bug — 12 Nov 2026

- The live instrument gained a neutron-star mode: ChirpNetBNS runs in the browser on 46 s of whitened GW170817
  strain from LIGO-P1700349 (and the raw P1700337 L1 with its glitch). Parity with Python 1e-7. The scan runs in a
  Web Worker. What it shows: statistic 13.8 (14.4 with Python's finer scan), chirp mass 1.122 vs LIGO 1.1975
  (outside its bar), no coincidence checker — sliding L1 by 4 s still reads "Merger" (statistic 7.0), and each
  detector alone clears the 5.5 threshold (H1 8.1, L1 10.7).
- Found while porting: `chirpnet.gate`'s cosine tapers are reversed (the mask falls to 0 at the outer edge of each
  0.1 s ramp and jumps back to 1 next to the zeroed stretch). It matters only near gates. Left as released (the
  browser port matches it exactly); listed as a next rung, since fixing it changes the gated GW170817 numbers.

## Gating bug fixed — 12 Nov 2026

- `chirpnet.gate` now tapers into and out of each gate the right way round, exactly as `training/bns_real.py` (which
  produced the published numbers) does: max difference 0.0 on a synthetic test. The browser port (`bns.ts`) is fixed
  the same way; parity 5e-7.
- Effect: the released code now reproduces the model card again. GW170817 with the raw (glitched) L1, gated:
  11.97 (card 12.0; the reversed tapers gave 12.21). Ungated 9.78 is unchanged.
- No bundled O1/O2 file triggers a gate, so the demo and its golden output are unchanged. In O3 only GW191204's
  H1 file gates (6.61–6.68 s, ~10 s before the merger); re-running `o3_exam.py` changed none of the reported numbers.

## Corrections

- 14 Nov 2026: Card: the rung table's column 'Loud one-detector fakes flagged' was renamed 'passed' (17% is the share of loud one-detector fakes the cascade lets through; rung 4 let 100% through). Rung 9 and neutron-star rung 3 are marked released, not kept. The black-hole chirp-mass range in the limits now reads 5.5–45 M☉, the simulated range (was 'about 6-45'). Network and credits sections added.

## Improvement pass, rules fixed before scoring — 15 Nov 2026

The training banks (LALSuite IMRPhenomD and NRTidalv2 waveforms) and checkpoints were never copied into this
repository, and LALSuite has no Windows build, so no network can be retrained here and the frozen simulated sets
(`val`, `bns_val`) cannot be rebuilt. GW190521 (detector-frame chirp mass 98.7 vs a training range of 5.5–45) needs a
new bank and a retrained finder; it is left as it is. The two failures that can be checked on dev without retraining:

1. **O3 background above the O1/O2 threshold.** Dev data: real O3 H1 + L1 strain that is not exam data
   (`training/o3_dev_background.py`: GWOSC bulk files at seeded times, 32 s pieces, more than 600 s from every
   O3 event GWOSC lists, DATA + CBC_CAT1, no hardware injection; scored with the O3 exam's background protocol).
   Rule, fixed now: if any dev window scores at or above 7.0, the release threshold is re-set by the rule that set 7.0
   (`release_build.py`: loudest background + 0.5, rounded up to the next 0.5), with the background now the O1/O2
   time slides plus the O3 dev time slides. If no dev window reaches 7.0, the failure does not show on dev and nothing
   changes. The network is unchanged, so the simulated objective cannot move; the sensitivity cost of a higher
   threshold cannot be measured on simulations here (no bank) and will be stated as unmeasured.
2. **GW170817 chirp mass 5% low, interval too narrow.** Dev data: TaylorF2 inspirals with spins and tides injected
   into real O2 noise that is not exam data (`training/bns_mf_dev.py`: four 2,048 s pieces of GWOSC bulk data at
   seeded times, more than a day from every O2 event, 120 injections, detector-frame chirp mass 0.9–1.8, SNR 12–35).
   Candidate: the released network finds the merger, then a TaylorF2 matched filter (`chirpnet.BNSMatchedFilter`,
   numpy only) refines the chirp mass on the whole inspiral from 25 Hz. Rule, fixed now, in the neutron-star ladder's
   own units (objective = sensitivity − 2 × median chirp-mass error; detection is unchanged, so the objective moves by
   2 × the error change): keep if, on the even-numbered injections, the matched filter's median chirp-mass error is
   lower than the network's by more than 0.005 (0.01 of objective) AND its 68% range, after one widen-only factor
   fitted on the odd-numbered injections, covers 60–80% of the true values. The failure counts as shown on dev if the
   network's median error on GW170817-like injections (chirp mass 1.1–1.3) exceeds 2% or its 68% coverage is below 50%.

## O3 dev background: the threshold failure is real, and larger — 15 Nov 2026

`training/o3_dev_background.py score` (64 pieces of 32 s, 8 file pairs: 4 in O3a, 4 in O3b; `logs/o3_dev_background.json`):

- 306,176 time-slid windows, 10.6 h of livetime (the O3 exam's background: 29,879 windows, 1.04 h).
- 46 windows at or above 7.0. Loudest 10.88; then 10.63, 10.27, 9.92, 9.91, 9.77, 9.48 (the top ten come from a handful
  of pieces; neighbouring windows overlap by 1.875 s). Quantiles: 99.9% 5.31, 99.99% 7.49.
- The loudest piece (O3b, GPS 1264024768) has a loud L1 glitch that is gated at 17.3 s; the trigger sits next to it,
  with a chirp mass of 6.0 (the training range starts at 5.5) and a checker score of −0.78, just above the −1.0 veto.
  Each detector against white noise scores −2.2 (H1) and −24 (L1): neither view alone looks like a merger.
- The O1/O2 background (57,760 windows, 2.0 h) never passed 6.2. O3 noise, read by this network, is much rougher.

The rule fixed above gives a threshold of 11.5 (10.88 + 0.5, rounded up). Not released: a release must beat the current
one on dev, and the cost of 11.5 in sensitivity cannot be measured without a simulation bank. What dev does say: at 7.0
the network's false-alarm rate in O3 is of order one cluster of windows every 2 h, not "below 0.5 per hour", and four of
the seven in-range O3 exam events (statistics 9.9–12.4) score below the loudest dev background window. The O3 detections
are therefore less significant than the exam table suggests. Next rungs: ship the O3 dev background so `scan()`
reports O3 false-alarm rates from O3 noise (GPS time is known at prediction time), and retrain with O3-like glitches
next to gates once a waveform bank can be built.
