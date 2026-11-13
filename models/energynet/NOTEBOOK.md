> **Lab notebook.** EnergyNet was built in a claude.ai cloud session (12–13 Nov 2026) on the shared RoadSim + RoadData
> platform, now in `shared/vehicle/`. The first entry is that session's status document as written; its paths
> (`/home/claude/…`, `/mnt/user-data/…`, `src/`) refer to the cloud sandbox. `MODEL_CARD.md` holds every number.

## 12–13 Nov 2026 — build notes (energynet-status.md, as written)

- **What it is:** predicts the energy a trip will take, before it happens, with an error bar. It covers fuel for conventional cars and hybrids, battery kWh for EVs, and the total for plug-ins. It works for every class (car through Class 8) and every powertrain. It was trained only on RoadSim, and it is Phase 2 project 1 of the vehicle work in ML Ratchet.
- **Page:** https://claude.ai/artifact/UYg6maW3hrMNAG3h8h6jRg
  - Hero: one real exam trip.
  - In-browser lab: 9 vehicles × 8 routes, with sliders for temperature, wind, mass, grade, start charge, and warm/cold start. It shows the component split and a temperature sweep. JS matches numpy to 1e-14.
  - Explorer: 39 real exam trips (seeded draw), scatters, ZTBus.
  - Also: exam tables, twin-calibration chart, ratchet, limits, credits.
- **Release:** `/mnt/user-data/outputs/energynet-v1.0.zip` (3.9 MB). It contains:
  - numpy-only `energynet.py`, `features.py`, `model_np.py`
  - 5-network ensemble weights and dev sigma factors
  - `demo.py` on 5 real exam trips, tested from an unzipped copy
  - `MODEL_CARD.md`, `README.md`, `training/`
- **Workspace:** `/home/claude/road/energynet` (ephemeral).

### Recipe as built

- **Physics:** frozen RoadSim (Phase 1 checks). Fixes are wrappers in `src/wrappers.py`, ready for the coordinator to upstream.
- **Data:** 2,400 training / 400 validation generator vehicles, widened in three ways:
  - engine efficiency 0.68–1.08, because the dev twins needed 0.72–1.07
  - older automatics
  - 35% articulated buses
- **Simulated labels:** window-sticker labels from FTP/HWFET plus the EPA 5-cycle formulas.
- **Routes:**
  - the 1,092 VED train_noise routes (real speed, lidar grade, ASOS weather)
  - EPA and heavy-duty cycles
  - synthetic drives
- **Real-noise analogue:** the network sees station weather, while the simulated car drives T ± 1.5 °C with wind × 0.4–0.9.
- **Bank sizes:** 28.8k training and 3,991 frozen validation examples.
- **Spec masking:** redrawn every epoch, from EPA-like specs down to class + powertrain only.
- **Network:** MLP with 149 inputs → correction to a generic road-load energy estimate, plus σ and 5 energy components. The ensemble has 5 members.
- **Dev twins** (16 dev cars on their own 1,498 trips), bias by stage:

  | Stage | Petrol | Hybrid |
  |---|---|---|
  | Generic spec | −28% | −22% |
  | Label-matched twin | −17% | −20% |
  | + MAF label model | −6% | — |
  | + on-road loads | 0% | 0% |

### Calibration rungs from dev

- **MAF label model:** during fuel cut-off, VED's MAF-estimated fuel reads about 0.2–0.3 g/s instead of zero.
- **On-road loads:** A/C, accessories, and real-world hybrid operation:
  - conventional: 1380 W + 133 W/°C above 15 °C
  - hybrid: 2831 W + 204 W/°C
- **Plug-in hardware patch:** pack power ≥ 1.2 × motor, R0 / 2.5, MG2 torque × 1.8. Real dev plug-ins burned no fuel on 95% of trips above 30% charge; the simulated ones took 31% of their energy from fuel.

### Ratchet

v4 val skill and dev D (mean of 15 real dev MdAPEs):

| Rung | Change | Skill | Dev D | Decision |
|---|---|---|---|---|
| r1_abs | Predicts absolute log energy | .544 | .152 | Kept |
| r2_corr | Correction to the physics estimate | .556 | .160 | Kept |
| r3_maf | MAF label model | .580 | .152 | Kept |
| r4_rw | On-road loads | .641 | .142 | Kept |
| r5_phev | Plug-in hardware patch | .648 | .129 | Rejected: +0.007, below the 0.01 margin |
| r6_ens | 5-member ensemble on v4 physics | .656 | .129 | **Kept, released** |

### Exam (MdAPE)

VED trips, EnergyNet vs label / MOVES-like / class-median baselines:

| Fleet | Trip MdAPE | Car total |
|---|---|---|
| ICE | **8.9%** vs 13.0 / 10.3 / 22.8 | **6.8%** |
| HEV | 16.7% vs 16.2 / **15.3** / 17.2; bias +12% | 16.2% vs label 8.9 |
| EV (2 Leafs) | 22.8% vs 21.5 / **19.0** / 27.1; bias −20% | — |
| PHEV | **20.1%** vs 46.1 / 41.3 / 42.7 (uses logged start SOC) | **8.9%** |

- P2 hybrids: 9.0% vs 18.3%.

EPA exam (cycle labels corrected; regression refit on dev):

| Group | Test | EnergyNet | Regression |
|---|---|---|---|
| Conventional | FTP | 9.6% | **7.1** |
| Conventional | HWY | 15.6% (+15% bias) | **5.0** |
| Conventional | US06 | 11.5% | **6.9** |
| HEV | FTP | **8.7%** | 16.3 |
| HEV | HWY | **5.9%** | 21.5 |
| HEV | US06 | **14.9%** | 15.8 |
| BEV | UDDS | **7.9%** | 9.8 |
| BEV | HWY | **5.0%** | 6.5 |

Heavy duty:

| Set | EnergyNet | Baseline |
|---|---|---|
| ZTBus legs | **16.4%** | 22.1 |
| ZTBus missions | 16.2% | **14.5** |
| Mersin diesel | 14.8% | **8.6** (same-route median) |
| Mersin CNG | 15.5% | **7.1** |
| ICCT | 11.4% | **7.0** |

Coverage on VED is too low even after dev widening:

| Fleet | Inside 68% band | Inside 95% band |
|---|---|---|
| ICE | 55% | 85% |
| HEV | 65% | 88% |
| EV | 47% | 84% |
| PHEV | 65% | 89% |

EPA coverage is near nominal.

### Limits

- Hybrids and EVs lose to simple baselines. Their calibration rests on 5 dev hybrids and 1 dev Leaf.
- Conventional EPA highway is biased: the fleet prior was widened toward old engines.
- Error bars are too narrow on VED.
- Heavy duty is thin and partly synthetic.

### Data bug found (for the coordinator)

- RoadData's EPA wide table swaps cycles:
  - 'HWFE' is labelled FTP
  - 'Federal fuel 2-day exhaust' is labelled HWY
- As a result, conv FTP_mpg/HWY_mpg medians are 35.6/24.5.
- `src/epa_fix.py` rebuilds them in the EnergyNet folder; data/ is untouched.
- The published EPA baseline numbers (8.8/7.0%) refer to the swapped labels.
- The VED spec also double-counts 136 kg (mass_kg already includes it, and payload_kg adds it again).

### Next rungs

1. A cold-start calibration from EPA dev FTP_cold.
2. An HEV/EV twin rung with more dev units, or a per-architecture σ model.
3. A "model-era" input so EPA-modern specs do not inherit the old-fleet prior.

### Sandbox gotchas

- The bank takes about 2 h on 2 shared cores (hybrids run at ~1,500× real time with record=True).
- pandas groupby.apply drops the group column.
- json NaN breaks JS JSON.parse.

## 13 Nov 2026 — integrated into ML Ratchet

- Moved from `energynet-v1.0.zip` into `models/energynet/`. Sandbox paths now go through `training/_paths.py`
  (work folder `models/energynet/work/`, the vehicle platform at `shared/vehicle/` or `$ML_RATCHET_VEHICLE`).
- The demo reproduces the release's output exactly from the repository (checked against the zip's own run).
- Caveat carried from the handoff: built before RoadData's 13 Nov fixes (EPA city/highway labels were swapped; VED
  mass double-counted the 136 kg driver). Re-score after the fixes are upstreamed into RoadSim.

## 13 Nov 2026 — original lab and live instrument

- The original claude.ai page is now `lab/index.html`, unchanged.
- The site's model page has a live instrument (`web/export_instrument.py`, `site/src/instruments/energynet/`). Five real exam trips (one per powertrain); temperature, mass, grade, wind, start charge and cold start change the input and the 5-network ensemble reruns. Parity with Python: 32 cases, worst 1.6e-7.
