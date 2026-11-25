# r2_d2_transformer: Transformer over 20-s patches, 5-member ensemble on 3 banks (round 2, D2)

*Vehicle Bake-off round 2 (24-25 Nov 2026). Pre-registration: `round2/PREREG.md`. Track: trace.*

**What.** Five transformer encoders over 20-s patch tokens of the 1 Hz trip trace, each correcting EnergyNet's road-load estimate E_phys, trained on EnergyNet v2.1's three simulated banks (RoadSim physics v2) instead of one, averaged in log-quantile space. Round 2's data rung D2. numpy inference (the v1.0 tracelib cores, unchanged).

**Training.** round2/trace/train_ens.py (imports v1.0 arms_trace/train/nets.py and the v1.0 configs transformer_r1.json then transformer_r2.json unchanged): phase 1 12 epochs peak lr 2e-3 (data seed m), phase 2 warm restart 24 epochs peak lr 1e-3 (data seed m + 100), token budget 8,192 blocks, <= 64 trips per batch, pinball loss on 5 log-quantiles + 0.2 x component MSE, fresh tiers per bank per epoch, last step used. Zero real labels.

**Rungs** (frozen sim val: 3,985 trips, frozen tiers; dev D16 = mean of the 16 dev MdAPEs):

| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | Decision | Why |
|---|---|---|---|---|---|---|
| v1-r2 | incumbent: v1.0 trace_transformer r2 (one network, one bank; not in this folder) | 0.6442 | 0.0829 | 0.1135 | base | v1.0 release, re-scored on the rebuilt frozen val and dev |
| d2-m0 | diagnostic, not a rung: member 0 alone (one network, the 3 banks) | 0.6603 | 0.0793 | 0.1048 | rejected | diagnostic only (PREREG section 1): the data effect alone is +0.0161 skill; D16 0.1048 |
| d2 | v1.0 r2 recipe (r1 phase 12 epochs lr 2e-3, then warm restart 24 epochs lr 1e-3), 5 members (seeds 0-4), each on all three v2.1 banks (86,400 trips), GPU | 0.6715 | 0.0771 | 0.1065 | released | sim val skill +0.0273 over the incumbent (> 0.01) and dev D16 0.1065 vs 0.1135 (better): kept (PREREG section 0 gate) |

**Compute.** 5 members x (12 + 24) epochs on 86,400 trips (3 banks), RTX 3080 shared by up to 8 jobs: 63 min wall per member (5.21 GPU-h in all), process CPU ~1.0 h per member (4.88 h). The v1.0 rungs (1.64 single-thread CPU-h) are the incumbent's.

**Inputs and features.** as v1.0 trace_transformer r2 (arms_trace/common/tracelib.py, unchanged): 5-s block means of the 1 Hz trace (12 channels), 4-block (20-s) patch tokens, spec (49) and road load (5) standardised with train101 statistics, log distance / duration, physics anchor E_phys recomputed from the trace inputs (bake.e_phys_from_trace, declared), target log(y / E_phys), log(E_phys / A0) as an input. Ensemble: mean log-quantile per level over the 5 members, pred = exp(mean log q50) (round2/arms/common/enslib.py)

**Limits.** Same simulator as v1.0 (RoadSim physics v2): the shared dev misses remain (dev bias: plug-ins -9.9 %, Mersin CNG -18.8 %, ICCT +18.1 %). Bands are raw pinball quantiles (conformal wrapper available). numpy inference runs 5 transformers: ~5x the v1.0 arm's predict time.
