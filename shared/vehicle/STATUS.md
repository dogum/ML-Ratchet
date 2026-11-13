> Kept as written on 13 Nov 2026 in the cloud project where the vehicle work was built. Paths such as `claude/…-status.md` and the `.zip` names refer to that project; in this repository the platform is `shared/vehicle/` and the models are `models/energynet`, `models/tripdetective` and `models/hybridpilot`. Page links now point to each model's `lab/`.

# Vehicle project (RoadSim) — status (13 Nov 2026)

Greg's whole-vehicle physics + systems project for ML Ratchet: every architecture (conventional, P2/series/power-split
hybrid, PHEV, BEV), cars through class-8 trucks and buses, at subsystem level, on real roads with real grades and weather.
Built in two phases by parallel Opus agents from `VEHICLE_BRIEF.md` and `PHASE2_BRIEF.md` (both in the zip).

## Deliverables
- **`roadsim-roaddata-v1.0.zip`** (22 MB): the shared layer.
  - `roadsim` package, checks, briefs
  - `realdata` loader, frozen splits, EPA/fueleconomy/heavy-duty tables, cycles, baselines
  - The VED trips (≈1.2 GB) are not included; build scripts and sources are.
- **EnergyNet v1.0**, trip energy predictor: https://claude.ai/artifact/UYg6maW3hrMNAG3h8h6jRg · `energynet-v1.0.zip` · `claude/energynet-status.md`
- **TripDetective v1.0**, infers the vehicle from a trip log: https://claude.ai/artifact/2DFuvMBBAwREG9rx5e3tHg · `tripdetective-v1.0.zip` · `claude/tripdetective-status.md`
- **HybridPilot v1.0**, learned hybrid energy management: https://claude.ai/artifact/3MU8pXNtGc65HTDnQ8Cgup · `hybridpilot-v1.0.zip` · `claude/hybridpilot-status.md`
- Pages are private until shared.
- All demos were run from unzipped copies on 13 Nov.

## Phase 1: RoadSim (simulator)
- **Coverage:** 7 architectures × 7 classes.
  - **Engines:** EPA ALPHA engine maps scaled at constant BMEP; Willans-line model for heavy-duty diesels; cold start with coolant/oil warm-up.
  - **Motors:** ALPHA motor loss maps.
  - **Transmissions:** AT+TC, DCT/AMT, CVT, single, eCVT.
  - **Battery:** equivalent circuit (OCV/R vs SOC and T) with thermal model.
  - **Regen:** blending with motor, battery and stability limits.
  - **Road loads:** wind with yaw, rolling resistance vs temperature/surface/warm-up, grade, air density.
  - **HVAC:** heat pump, PTC or engine heat; A/C.
- **Strategies:** conventional shift/BSFC-optimal/stop-start; thermostat; power-follower; P2 assist; Prius-style split; ECMS; A-ECMS; MPC (15 s preview); DP (offline); PHEV CD-CS and blended; BEV regen variants.
- **Ledger:** closes on 600 stress runs, max error 0.05%.
- **Speed (× real time):**

  | Vehicle | Speed |
  |---|---|
  | Conventional / BEV | ~60,000× |
  | HEV | ~5,000× |
  | DP (per UDDS) | 0.8 s |

- **EPA, 33 held-out real vehicles (median error):**

  | Powertrain | FTP | HWFET | Other |
  |---|---|---|---|
  | Conventional | +5.6% | −4.3% | US06 +0.2% |
  | Power-split HEV | −2.3% | | |
  | BEV | −0.5% (UDDS) | −1.7% | |

- **FASTSim 3.1 cross-check** (same components): conventional within 3%, BEV within 4%.
- **Known limits:**
  - P2 hybrids optimistic in the city (+8 to +31%).
  - Cold FTP +13%; SC03 A/C +12%.
  - Hybrid bus too efficient.
  - Leaf twin uses ~1.5× real battery energy (cabin heat).
  - Toyota HEV twins run the engine less than real cars (35% vs 55% of time).
  - Heavy-duty is checked only against published ranges.

## Phase 1: RoadData (real data)
- **VED, Ann Arbor 2017–18:** 33,015 trips (104k mi) from 384 cars.
  - Grade from USGS 1 m lidar (repeatable to 0.36%).
  - Weather from the Ann Arbor airport ASOS.
  - Fuel is estimated from MAF/trims for most cars (not metered); make/model/year hidden; trips are warm fragments.
- **EPA test cars 2008–2026:** 80,227 tests with road-load ABC. Also fueleconomy.gov labels and 24 cycles.
- **Heavy duty:**
  - ZTBus Zurich: electric trolleybus; 10 missions with traces.
  - Mersin buses: trip totals only.
  - ICCT truck dyno table: 6 exam numbers.
  - No public truck trip data was reachable (NREL Fleet DNA is blocked).
- **Splits:** frozen by vehicle (sha256 d917df78…).
  - dev: 16 VED cars plus 10% of EPA families.
  - exam: the rest.
  - train_noise: 17 unlabelled VED cars, used as real routes.
- **Fixes on 13 Nov, after Phase 2** (splits unchanged; details in `data/DATA.md`):
  1. EPA cycle labels were swapped: code 3 is highway (HWFE), code 21 is FTP. Rebuilt. Conventional regression baselines are now FTP 7.1% / HWY 5.1% / US06 6.9%.
  2. VED mass double-counted the 136 kg driver load. Now curb only.

## Phase 2 results (exam = never used for choosing)

**EnergyNet** (MLP ensemble, correction to a road-load estimate; 6 rungs; dev-calibrated MAF label model and on-road loads). VED per-trip median error:

| Fleet | EnergyNet | Best baseline | Notes |
|---|---|---|---|
| Petrol | **8.9%** | 10.3% | Per car 6.8% |
| Hybrids | 16.7% | **15.3%** | Bias +12% |
| Leafs | 22.8% | **19.0%** | |
| PHEV | **20.1%** | 41% | Uses logged start SOC |

- EPA:
  - Hybrid and BEV: beats the regression (e.g. HEV highway 5.9% vs 21.5%).
  - Petrol: loses (highway 15.6% vs 5.0%, biased by an old-fleet prior).
- ZTBus legs: beats (16.4% vs 22.1%). Mersin and ICCT trucks: loses.
- Error bars too narrow on real trips (55–65% coverage at 68%).

**TripDetective** (CNN + classical per-trip features, pooled across trips; OBD log emulator; v3 mass prior matched to VED-era cars). VED exam, 351 cars:

| Measure | TripDetective | Baseline |
|---|---|---|
| Mass | **116 kg** (7.8%) | Physics fit 205 kg |
| Road-load force | **16–18%** | Physics fit 19–66% |
| Architecture | 98.9% | Rule **99.4%** |
| Engine size | 0.50 L (bias −0.48 L) | Median 0.60 L |

- Mass coverage: 73% inside 1σ.
- Weakness: it reads the kind of car more than the dynamics (+800 kg moves the estimate ~200 kg; CdA barely registers).
- Energy ledger graded on simulated twins only.

**HybridPilot** (16k-parameter policy: learned equivalence factor + P(on), trained by DP imitation and DAgger across 18 class × architecture combos). On 430 exam routes (VED exam routes and held-out vehicles, in RoadSim):

| Strategy | % of the DP optimum (matched SOC) |
|---|---|
| HybridPilot | 99.49% |
| A-ECMS | 99.35% |
| Rule-based | 98.28% |

- Gain over A-ECMS: +0.13 points [−0.06, +0.29]. Power-split +0.41, series bus +0.66, series PHEV −0.93.
- The simulator-vs-real check (exam HEVs) shows RoadSim fuel at 0.80× logged. That is far larger than the gaps between strategies, so the ranking holds only inside RoadSim.
- The fixed SOC correction over-credits (constant-s ECMS 105% vs 99.6% matched).

## Next (for Greg's code agent)
1. **Upstream the dev-calibrated wrappers** into roadsim, bump `PHYSICS_VERSION`, re-run checks and re-score every rung:
   - EnergyNet's MAF label model, on-road loads and PHEV hardware patch
   - TripDetective's OBD emulator and mass prior
   - HybridPilot's twin fixes: engine on above ~70 km/h, electric cabin heat for PHEVs, start penalty
2. **Fix the known simulator gaps:** P2 torque converter and start losses, cold weather, A/C, Leaf heating.
3. **Find real heavy-duty trip data** (Fleet DNA via Greg's browser; the full ZTBus set from ETH was rate-limited).
4. **Replace the hard-coded `/home/claude/road` paths.** Add RoadSim to ML Ratchet as a shared module under the three models.
