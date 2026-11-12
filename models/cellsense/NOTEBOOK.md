> **Lab notebook.** The build notes as recorded during the original work in a cloud sandbox (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that sandbox: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# CellSense — status (31 Oct 2026, v1.1)

- What it is: battery health and cycle life from one fast discharge, learned from physics alone. Third project in Greg's physics-ML collection, after ChirpNet and SunDoctor.
- Page: https://claude.ai/artifact/Hc6QmjyN1z8gEttKvdiLoP (v1.1). Sections:
  - hidden capacity
  - real-cell explorer
  - in-browser "age a cell" lab, now with a "stop the discharge at" slider routed to the partial-discharge network
  - partial-discharge exam table
  - lifetime section with the early-life figure (6 calibration cells, real vs simulated ΔQ₁₀₀₋₁₀)
  - ratchet (9 rungs), how it works, use it, limits
- Release: `cellsense-v1.1.zip` (7.6 MB), containing:
  - `cellsense.py` (numpy; routes curves that stop before 2.0 V to the partial network)
  - three weight files
  - `demo.py` (full and partial readings of b3c33)
  - `MODEL_CARD.md`, `README.md`
  - `training/`, `web/`
- The cloud workspace is ephemeral; the zip and the page are the durable copies.

## v1.1 plan (Greg picked, 30 Oct)
1. Other chemistries, rates and temperatures, using Sandia's data (Preger et al. 2020): Panasonic NCR18650B NCA, LG 18650HG2 NMC, and the same A123 LFP cell. **Blocked**: see below.
2. Partial discharges. **Done** for discharges from full, using Severson cells cut short. Sandia's 20–80% and 40–60% windows still need the Sandia data.
3. A useful lifetime forecast from early-life physics. **Done.**

## Sandia data: blocked
- BatteryArchive.org has no public download link. Its FAQ says to email info@batteryarchive.org.
- Probing the archive's database API from the sandbox was denied as circumvention. Not retried, and should not be.
- Options for Greg:
  - Download the SNL_18650_{NCA,NMC,LFP}_* cycle and time-series CSVs in his browser and attach them.
  - Email BatteryArchive.
  - Approve a different public multi-chemistry dataset with a documented download, for example:
    - Zhu et al. 2022 (Nat. Commun.) NCA/NCM cells on Zenodo
    - Pozzato et al. 2022 (LG M50T NMC811, which matches PyBaMM's Chen2020 parameters) on Mendeley Data

## Early-life physics (the main v1.1 finding)
- One mechanism fits the change real cells show by cycle 100 (Severson's ΔQ₁₀₀₋₁₀(V), the 4C curve sagging between 3.2 and 2.2 V) on all six calibration cells: a growing share of slow, poorly connected graphite, 4–10% by cycle 100.
  - The fits leave 1–4 mAh of a 10–41 mAh signal.
  - Transport-resistance growth, LLI and LAM signatures don't have the right shape; the simulator's staging features are too sharp. Comparisons are smoothed over 24 mV.
- Aging model v2.1:
  - The slow share grows about as cycles^0.8 and levels off at 8–14%.
  - Graphite-loss rate scales as (share / anchor share)³ around each calibration cell.
  - SEI loss is linear in cycles (Attia 2020: these cells aren't sqrt(t)).
  - Transport polarization grows late, about as (n/1000)², plus a term tied to lost Li and graphite.
  - Lifetime-bank reference curve is taken at cycle 10.
- Calibrated on the dev cells only. Simulated log-life vs log var(ΔQ) slope is −0.33 to −0.40 (dev cells: −0.38). The simulated scatter is 0.15, wider than the real 0.06.
- Severson's own variance feature reproduces on this data: r = −0.93 over all 124 cells.

## Ratchet (v1.1)
- Objective health half now averages three readings: with reference, without it, and as partial discharges. Skill is floored at 0.
- New gate: the calibration cells' lifetime error (read every 50 cycles) may not be more than 2 points worse.
- The frozen lifetime set was regenerated twice (v2, v3). All rungs were re-scored.

| Rung | Change | Result |
|---|---|---|
| 1 | Baseline | kept |
| 2 | Transport aging | rejected |
| 3 | Slow graphite | rejected |
| 4 | Lifetime network | rejected |
| 5 | Lifetime with calibrated aging priors | kept |
| 6 | Early-life physics | rejected by the new lifetime gate: dev lifetime error 45% vs 31% |
| 7 | Early life plus whole-life check | **kept**: dev lifetime 22% |
| 8 | Partial discharges, one network | rejected: dev capacity 16.6 vs 11.4 mAh |
| 9 | Partial-discharge specialist (smoothed) | **kept, released** |

- Released system: c1_base (health) + c8_early_sm (life) + c7_partial_sm (partial).
- An unsmoothed twin of rung 7 scored the same on the simulated set and worse on the dev cells (25% vs 22%). It was not entered.

## Real exam
- Health (full curves) is unchanged: 13.2 mAh total-capacity error, fade correlation 0.979.

### Lifetime at cycle 100

| Model | Primary error | Secondary error | Primary RMSE | Secondary RMSE |
|---|---|---|---|---|
| CellSense v1.1 | 22.4% | 13.1% | 145 | 249 |
| CellSense v1.0 | 33% | 20% | — | — |
| Median guess | 34% | 19% | — | — |
| Severson's models | 13–15% | 9–11% | 91–138 | 173–214 |

- Log correlation is 0.89 over all cells.

### Lifetime by stage of life

| Stage | v1.1 | v1.0 |
|---|---|---|
| First quarter | 18% | 24% |
| Second quarter | 23% | 28% |
| Third quarter | 33% | 40% |
| Last quarter | **36%** | 20% |

- The last quarter got worse. The lifetime net reads worn cells about 6% too healthy and overestimates the cycles left. The health net reads the same cells correctly.

### Partial discharges from full (total-capacity error)

| Stopped at | CellSense | No-model baseline |
|---|---|---|
| 0.90 Ah | 17 mAh | 58 mAh |
| 0.75 Ah | 29 mAh | 65 mAh |
| 0.60 Ah | 46 mAh (39 with the reference cut too) | 62 mAh |
| 0.45 Ah | 52 mAh | 56 mAh |
| 0.30 Ah | 42 mAh | 51 mAh |

- Assuming no fade gives 46 mAh. Readings at 0.6 Ah and below are no better than that.

## Next
- Get the Sandia data (Greg's call), then run tasks 31 and 32:
  - NCA_Kim2011 and Chen2020 (NMC811) OCVs
  - temperature and rate dependence
  - calibration on Sandia dev cells
  - multi-chemistry dev gate
  - partial windows
- Fix the late-life lifetime bias, for example by feeding the lifetime net the health net's capacity reading, or with more late-life reads in the bank.
- The simulator's 4C graphite-staging features are sharper than real. Fixing that (electrode-scale heterogeneity) would help everything that reads curve shape.
