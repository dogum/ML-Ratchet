# r2_d1_gru: Bidirectional GRU, 5-member ensemble on 3 banks (round 2, D1)

*Vehicle Bake-off round 2 (24-25 Nov 2026). Pre-registration: `round2/PREREG.md`. Track: trace.*

**What.** Five bidirectional GRUs over 5-s blocks of the 1 Hz trip trace, each correcting EnergyNet's road-load estimate E_phys, trained on EnergyNet v2.1's three simulated banks (RoadSim physics v2) instead of the one bank the v1.0 GRU saw, and averaged in log-quantile space. Round 2's data rung D1: it separates the data and ensembling effect from physics. numpy inference (the v1.0 tracelib cores, unchanged).

**Training.** round2/trace/train_ens.py (imports v1.0 arms_trace/train/nets.py and the v1.0 recipe gru_r1.json unchanged): 12 epochs, token budget 8,192 blocks, <= 64 trips per batch, AdamW peak lr 2e-3 one-cycle cosine, pinball loss on 5 log-quantiles + 0.2 x component MSE, fresh tiers per bank per epoch, last step used. Member m: torch seed m, data seed m. Zero real labels. Banks rebuilt with harness/build_bank.py (round2/bank/build.py; PREREG A1 / A2 for the manifest check).

**Rungs** (frozen sim val: 3,985 trips, frozen tiers; dev D16 = mean of the 16 dev MdAPEs):

| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | Decision | Why |
|---|---|---|---|---|---|---|
| v1-r1 | incumbent: v1.0 trace_gru r1 (one network, one bank of 28,800 trips; not in this folder) | 0.6390 | 0.0843 | 0.1089 | base | v1.0 release, re-scored on the rebuilt frozen val and dev |
| d1-m0 | diagnostic, not a rung: member 0 alone (one network, the 3 banks) | 0.6547 | 0.0813 | 0.1090 | rejected | diagnostic only (PREREG section 1): the data effect alone is +0.0157 skill; D16 0.1090 |
| d1 | v1.0 r1 recipe, 5 members (seeds 0-4), each on all three v2.1 banks (86,400 trips), GPU | 0.6696 | 0.0778 | 0.1084 | released | sim val skill +0.0306 over the incumbent (> 0.01) and dev D16 0.1084 vs 0.1089 (not worse): kept (PREREG section 0 gate) |

**Compute.** 5 members x 12 epochs on 86,400 trips (3 banks), RTX 3080 shared by up to 8 jobs: 14-20 min wall per member (1.39 GPU-h in all), process CPU 0.23-0.31 h per member (1.29 h). The v1.0 rungs r0-r2 (1.65 single-thread CPU-h) are the incumbent's.

**Inputs and features.** as v1.0 trace_gru r1 (arms_trace/common/tracelib.py, unchanged): 5-s block means of the 1 Hz trace (12 channels), spec (49) and road load (5) standardised with train101 statistics, log distance / duration, physics anchor E_phys recomputed from the trace inputs (bake.e_phys_from_trace, declared), target log(y / E_phys), log(E_phys / A0) as an input. Ensemble: mean log-quantile per level over the 5 members, pred = exp(mean log q50) (round2/arms/common/enslib.py)

**Limits.** Same simulator as v1.0 (RoadSim physics v2): the misses every learner shares on dev remain (dev bias: plug-ins -15.4 %, Mersin CNG -18.9 %, ICCT +16.5 %). Bands are raw pinball quantiles (conformal wrapper available). Numpy inference runs 5 GRUs: ~5x the v1.0 arm's predict time. Heavy-duty traces longer than 7,200 s are chunked as in v1.0.
