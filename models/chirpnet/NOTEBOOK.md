> **Lab notebook.** The build notes as recorded during the original work in a cloud sandbox (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that sandbox: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# ChirpNet & Collision Course — status (30 Oct 2026)

## ChirpNet v2.0 (gravitational-wave ML)
- What it is: small neural networks trained only on simulated mergers (LALSuite waveforms in simulated noise), tested on real LIGO data. A portable numpy release, plus a showpiece page.
- Page: https://claude.ai/artifact/Kd3TYXDnxFYFM1zTonrUc7 (version 2).
- Two zips were sent in the cloud session chat. The cloud workspace (`/home/claude/gw`) is ephemeral, so these zips and the page are the durable copies.
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

## Plans (Greg, 30 Oct)
- **Open-source it on Greg's GitHub** so others can find and use it. Start from `chirpnet-repo.zip`.
- To-dos before publishing:
  - Choose a code license (README has a TODO; the LIGO data stays CC BY 4.0).
  - Replace the hardcoded `/home/claude/gw` paths in `training/`.
  - Optionally enable GitHub Pages on `docs/`.
- **Artifact Shelf** (https://github.com/dogum/artifact-shelf; local `~/Developer/artifact-shelf`): ChirpNet should go on it, possibly as a shelf-styled version of the page, alongside Collision Course. Only do this when Greg says so. The GitHub API was not enabled for that repo in the cloud session, so shelving needs repo access or his Mac.
- Greg may move this conversation into his artifacts-collection or public-projects project folder. The work overlaps with both.

## Blocked / next
- **O3 final exam:** the GWOSC servers are blocked from the cloud sandbox. Greg needs to attach H1+L1 files (32 sec, 4 KHz, HDF5) for GW190412, GW190408_181802, GW190828_063405, GW191204_171526, GW200129_065458, GW200224_222234, GW200311_115853 and GW190521.
- Ideas next:
  - A coincidence checker for the BNS model.
  - Longer BNS windows, to fix the chirp-mass bias.
  - Live posterior sampling in the browser.

## Collision Course (WebGL2 galaxy-collision simulator)
- https://claude.ai/artifact/T2b4hUR7A5CVLogNbfkm3T (v2 adds the Local Group scenario).

- Greg enjoys learning outside his field. Explain in plain language, and keep visuals front and center.
