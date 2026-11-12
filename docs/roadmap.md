# Roadmap

Where the collection goes next, beyond each model's own next rungs (in its `model.yaml` under `next`).

## Exams waiting on data (help wanted)

- ~~**ChirpNet O3 exam.**~~ Done 12 Nov 2026 once GWOSC was reachable: 7 of 7 in-range events found, every catalogue
  median inside the 90% posteriors; GW190521 (outside the training range) missed; one O3 background window above
  threshold. See `models/chirpnet/training/logs/o3_exam.json`.
- **CellSense, other chemistries.** Sandia NCA / NMC / LFP cycling (Preger et al. 2020) from BatteryArchive, or
  another public multi-chemistry dataset with a documented download (Zhu et al. 2022; Pozzato et al. 2022).

## The collection as a tool

- **pip-installable models.** `pip install ml-ratchet[quakenet]`, with weights fetched from GitHub Releases on first
  use instead of shipping in the wheel.
- **A physics MCP server.** Each model becomes one tool that returns its readings *with* error bars and the card's
  limits: `chirpnet.scan`, `sundoctor.diagnose`, `cellsense.health`, `transitnet.read`, `quakenet.read`,
  `bearingnet.diagnose`, `swellcast.read`, `beatnet.classify` (with its not-a-medical-device notice),
  `starquake.read`. `mlratchet.load()` already imports any model in isolation, which is most of the work.
- **One design language for the labs.** The nine interactive pages were each built with their own visual identity.
  The website now wraps them in one system; porting each lab natively into the site (shared JS network runtime,
  shared chart components) is the next step.
- **A shared JS runtime** for in-browser networks: numpy-equivalent conv/GELU/BatchNorm in TypeScript, with parity
  tests against the Python modules.

## The method

- **A reusable twin-test harness** in `mlratchet`, since it found the biggest simulator gaps in three projects.
- **Per-stratum σ widening** (StarQuake showed one pooled factor isn't enough across observing lengths).
- **Saliency on real vs twin** inputs, to show where a network looks when it misreads real data.

## New models

See [backlog.md](backlog.md). Twelve candidates have physics, data and an exam lined up.
