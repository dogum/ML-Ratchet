# ML Ratchet repo — running notes

**Published (Nov 2026):** https://github.com/dogum/ML-Ratchet, with its site at https://dogum.github.io/ML-Ratchet/.
- Name: ML Ratchet. Licence: MIT for code, weights and docs; real data stays under its providers' terms (DATA_LICENSES.md).
- Package: `mlratchet`, with a CLI (`list`, `info`, `ladder`, `demo`, `scoreboard`, `new`, `validate`) and `mlratchet.load(id)`.
- Each model folder holds `model.yaml` and `ratchet.yaml`; the README, CLI and site are generated from them.
- The site has a scoreboard (wins/losses vs baselines), a backlog, a method page and the agent brief.
- Not yet in it: an MCP server, and the vehicle project (RoadSim + EnergyNet, TripDetective, HybridPilot, built 13 Nov; see `roadsim-status.md`). RoadSim should become a shared module, `mlratchet/roadsim` or `shared/roadsim`, that the three models import.
- Not yet in it: an MCP server. The notes below are the pre-publication plan, kept for history.

Greg's running document for turning the physics-ML collection into one big open-source repo. He'll work out the details in Claude Code when he's ready; until then, add to this file whenever a repo-level decision, to-do or idea comes up.

## The idea

Physics first, data second. Every model learns only from simulated physics (injected into real instrument noise where possible). It is then examined on real data that was never used to choose anything. The goal is a large open repo with as many models as public data allows. Someone searching for a starting point on an engineering, science or ML problem should find a worked, honest example: physics, simulator, model, ratchet log and real exam.

## Name

- Greg leans toward **ML Ratchet**. Other names he liked: **First Light**, **Ratchet**, **Ab Initio**.
- "Ratchet" names what makes the collection different. A model only moves up a rung if it beats a frozen simulated test by more than a margin and doesn't get worse on a small real dev set. The final exam on real data is never used to choose anything.
- "Ab Initio" ("from first principles") names the physics-first idea.

## Projects so far (latest release of each)

| Project | What it does | Page | Zip |
|---|---|---|---|
| ChirpNet v2 | Black-hole and neutron-star mergers in LIGO data | https://claude.ai/artifact/Kd3TYXDnxFYFM1zTonrUc7 | `chirpnet-v2.zip`, plus `chirpnet-repo.zip` (GitHub-ready snapshot) |
| SunDoctor v2 | Day-by-day solar-farm diagnosis | https://claude.ai/artifact/VqpkFf7pWwWhodohiaqE97 | `sundoctor-v2.zip` |
| CellSense v1.1 | Battery health and life from one discharge | https://claude.ai/artifact/Hc6QmjyN1z8gEttKvdiLoP | `cellsense-v1.1.zip` |
| TransitNet v1.0 | TESS planet vetting and measurement | https://claude.ai/artifact/1hdMeCpp4qDXBN3a8B9Ff5 | `transitnet-v1.0.zip` |
| QuakeNet v1.2 | Earthquake size, distance, depth and direction from one seismometer | https://claude.ai/artifact/DNinPsWLG3EEv2ezCygYJn | `quakenet-v1.2.zip` |
| BearingNet v1.0 | Bearing faults from vibration | https://claude.ai/artifact/Q2TRef2NuSbUcuXpw7gNgu | `bearingnet-v1.0.zip` |
| SwellCast v1.0 | Distant-storm distance, time and direction from one ocean buoy | https://claude.ai/artifact/PiqMh45i3co2RguewgUg9k | `swellcast-v1.0.zip` |
| BeatNet v1.0 | Heartbeat classes from ECG (research only, not a medical device) | https://claude.ai/artifact/BtCFF3jmGchgCqkbjpAyeH | `beatnet-v1.0.zip` |
| StarQuake v1.0 | Red-giant oscillations → mass and radius | https://claude.ai/artifact/Wfu9o3WtAegSyqMzPEKBav | `starquake-v1.0.zip` |

- Related, not ML: Collision Course (https://claude.ai/artifact/T2b4hUR7A5CVLogNbfkm3T) and Soft-Body Lab (own git repo on Greg's Mac, `~/Developer/soft-body-lab`).
- All zips were sent in the 6–7 Nov cloud session chat. The four newest were built by parallel Opus agents from a shared brief (the recipe as a checklist; worth turning into the repo's "how to add a model" guide).
- Note: the four newest zips unpack into a top-level folder (`<name>-v1.0/`); the older five unpack flat. Normalize this in the repo. Cloud workspaces are temporary, so the downloaded zips and the pages are the durable copies.
- Per-project details are in each `claude/<project>-status.md` doc.

## To fix before the code goes public

- **License:** none chosen yet. ChirpNet's README has a TODO for it. LIGO data stays CC BY 4.0. Check every dataset's terms (GWOSC, PVDAQ, Severson/Toyota, MAST/TESS, EarthScope/USGS) and credit each in its model card.
- **Paths:** the `training/` scripts in every zip are hardcoded to cloud paths (`/home/claude/gw`, `/sun`, `/cell`, `/transit`, `/quake`, `/bearing`, `/swell`, `/beat`, `/star`). They need to become relative paths or one config setting.
- **Exams waiting on data from Greg:**
  - ChirpNet's O3 exam: the GWOSC servers are blocked from the cloud sandbox. Greg attaches H1+L1 32 s / 4 kHz HDF5 files for the listed O3 events.
  - CellSense's other chemistries: Sandia NCA/NMC/LFP files from BatteryArchive, downloaded in his browser or via his Mac.
- **Shared layout:** every zip already has a numpy-only model, a model card with its limits, a demo on real examples and the full training code. That should make the merge straightforward.

## Repo shape (ideas, not decided)

- One folder per model, all in the same layout:
  - `model.py` (numpy only)
  - weights
  - `demo.py` and `examples/`
  - `MODEL_CARD.md`
  - `training/`
  - `ratchet.json`
  - `web/` (page source)
- Shared formats for model cards (inputs, outputs, physics, ratchet table, exam, limits, data credits) and ratchet logs (rung, change, sim score, dev gate, accepted, why; exam reported separately).
- A single `pip install` with each model as a subpackage. Weights download on first use, or ship via releases/LFS.
- **MCP server:** each model is one tool, returning error bars and the model card's limits:
  - `chirpnet.scan`
  - `sundoctor.diagnose`
  - `cellsense.health`
  - `transitnet.read`
  - `quakenet.read`
  - `bearingnet.diagnose`
  - `swellcast.read`
  - `beatnet.classify` (with a not-a-medical-device notice)
  - `starquake.read`
- GitHub Pages: each project's showpiece page plus an index.
- A "how to add a model" guide, with the recipe as a checklist:
  1. Real physics, verified against a reference.
  2. Simulator in real noise.
  3. Frozen simulated set.
  4. Dev gate on real units.
  5. Ratchet log.
  6. Real exam never used for choosing.
  7. Baselines, including a dumb one.
  8. Model card with honest limits.
- Possibly a shared core library: ratchet runner, frozen-set versioning and re-scoring, sigma calibration on dev, numpy layers (conv, GELU, BatchNorm folding), JS port helpers.

## Open questions for later

- Monorepo vs one repo per model with an umbrella index.
- Where large data lives: links plus fetch scripts, never re-hosted unless the license allows.
- How to label models whose exams are still partial (ChirpNet O3, CellSense chemistries).
- How to show models that lose to a baseline on part of their exam (BearingNet on CWRU, SwellCast vs the median guess, StarQuake on 4-year light curves). Proposal: every model card gets a "where it wins / where it loses" table up top.
- Medical-adjacent models (BeatNet) need a clear licence and disclaimer section.
