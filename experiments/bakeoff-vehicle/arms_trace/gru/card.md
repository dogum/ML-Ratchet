# Bidirectional GRU on the trip trace (`trace_gru`)

Vehicle Bake-off, **trace track**, group B (trace learners, `arms_trace/`). Family: **State-space and recurrent models (bidirectional GRU)** (`ssm` on the Model families page). numpy-only inference (`predict.py` -> `../common/tracelib.py` + `weights.npz`). Released rung: **r1**. Zero real labels in training.

## The built-in assumption
A learned state is carried along the trip, once forwards and once backwards (one GRU of 48 units per direction). Gates decide at every block what to keep and what to overwrite, so it can integrate slow things such as engine warm-up, the battery's state of charge or the cumulative climb, in trip order.

## Network
core: GRU(48 -> 48) forwards + GRU(48 -> 48) on the trip reversed -> Linear(96, 48), residual. Parameters: 53,130 (the four trace arms hold 49k-86k).

## Shared with every trace arm (only the core differs)
- **Inputs (the trace track's, nothing else):** the 1 Hz trace (8 channels), `spec` (49) and `rl` (5).
- **Resampling:** the harness's shared rule, `bake.block_means(trace, 5)`: 5-s blocks of the 8 channel means + mean v^2,
  mean v^3 and the summed positive kinetic-energy increment per unit mass, + the block's valid-second fraction ->
  12 scaled channels per block. Traces longer than 1,440 blocks (7,200 s, the sim bank's cap; only real heavy-duty legs,
  up to 37,919 s) are cut into <= 1,440-block chunks; each chunk is encoded, and the chunk means are averaged weighted by
  their block counts (the mean pooling is intensive, so this is the harness's "mean for intensive quantities").
  Block encodings are cached on `trace_key` (EPA rows repeat 5 cycle traces).
- **Statics (57), repeated at every block (early fusion) and fed again to the head:** spec standardised with the train
  bank's mean / sd (tiers drawn with seed 12345); log mass, log CdA, log Crr, Crr speed coefficient and rotating-mass
  fraction from `rl` (standardised); log distance and log duration of the trip (sum of the 1 Hz speeds, length);
  log(E_phys / A0) from r1 on (0 in r0).
- **Stem:** Linear(69 -> 48) + a residual pointwise MLP (48 -> 96 -> 48, GELU): the input-side nonlinearity the River
  Bake-off found S4D needs, given to every arm from the start.
- **Pooling and head:** final LayerNorm, mean over the valid blocks, [pool | statics] -> Linear(105 -> 64) -> GELU ->
  Linear(64 -> 10): the median log(E / anchor), the 25/5 % and 75/95 % quantiles as cumulative softplus offsets
  (monotone, never cross), and 5 log component ratios (aero, rolling, brakes, aux, powertrain losses) / anchor ->
  `components` (MJ) for the page.
- **Anchor:** r0: A0 = the train bank's median energy per km for the vehicle's (architecture, class)
  (`bake.dumb_rates()`, the climatology rate) x the trip distance: no physics. r1 on: E_phys = EnergyNet's generic
  road-load estimate recomputed from the trace-track inputs by the harness's `bake.e_phys_from_trace` (declared
  feature engineering); the network predicts log(E / E_phys), as EnergyNet does.
- **Loss:** pinball loss of the 5 log-quantiles + 0.2 x MSE of the log component ratios (EnergyNet's weight); trips with
  y <= 0.02 MJ masked (EnergyNet).
- **Training:** the harness's train bank only (28,800 simulated trips, one of EnergyNet v2.1's three banks; zero real
  labels), every trip whole, fresh tiers every epoch (`bake.draw_tiers`), target `bake.target`; batches of trips of
  similar length (<= 8,192 blocks, <= 64 trips; ~929 steps per epoch); AdamW (wd 1e-4), one-cycle cosine (3 % warm-up,
  peak 2e-3 -> 1e-5), gradient clip 1.0, 12 epochs; last step used (no early stopping); same data stream for every arm.
  One thread (`torch.set_num_threads(1)`), nice 5, shared 2-core machine.

## Ratchet

Rule: keep a rung if its full sim val skill beats the incumbent by > 0.01 AND its dev D16 is not worse than the incumbent's by more than 0.01 (HARNESS.md section 7, EnergyNet's model-rung rule). Sim val = the harness's `evaluate.py` on all 3,985 frozen sim val trips (frozen tiers); dev = the dev gate's D16. Paired bootstrap = sim val skill difference vs the incumbent, 2,000 resamples of the 400 val vehicles.

| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | VED ICE / HEV / EV / PHEV | EPA conv FTP / HWY | Mersin diesel / CNG | ICCT | CPU h | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r0 | base: pure trace learner; anchor A0 = sim-train median MJ/km (architecture x class) x distance, no physics estimate; 12 epochs | 0.605 | 0.0918 | 0.1192 | 0.078 / 0.121 / 0.140 / 0.270 | 0.077 / 0.068 | 0.141 / 0.206 | 0.180 | 0.58 | **base** | first rung: what the GRU finds from the raw trace alone (skill 0.605, D16 0.119) |
| r1 | physics anchor: target log(E / E_phys) with E_phys recomputed from the trace inputs (bake.e_phys_from_trace), log(E_phys / A0) as an input; 12 epochs from scratch | 0.639 | 0.0843 | 0.1089 | 0.070 / 0.121 / 0.121 / 0.229 | 0.079 / 0.079 | 0.126 / 0.189 | 0.141 | 0.54 | **released** | sim val skill +0.034 over r0 (paired bootstrap +0.019..+0.046) and D16 0.1089 better than r0 0.1192: kept; released (r2 did not beat it) |
| r2 | 2x training: warm restart from r1 for 12 more epochs (one-cycle cosine, peak lr 1e-3, data stream seed 1) | 0.647 | 0.0826 | 0.1102 | 0.072 / 0.131 / 0.128 / 0.223 | 0.091 / 0.067 | 0.132 / 0.186 | 0.113 | 0.51 | **rejected** | sim val skill +0.008 over r1 (bootstrap -0.001..+0.016) < the 0.01 margin; D16 0.1102 vs 0.1089: no gain from 2x training |

References (same harness): EnergyNet v2.1 (table track, 3x the training data, 5-member ensemble) sim val skill 0.680, D16 0.1133; E_phys alone 0.048 / 0.2278; climatology 0 / 0.3093.

## Released rung r1
Sim val (3,985 trips): skill 0.639, MdAPE 0.0843, bias +0.0046, cov50 / cov90 0.50 / 0.89, crps_log 0.0832. MdAPE by architecture: conv 0.066, hev_parallel_p2 0.107, hev_series 0.111, hev_powersplit 0.086, phev_powersplit 0.208, phev_series 0.240, bev 0.098.

Dev gate (MdAPE, cov50 / cov90; raw bands, no dev widening):

| Set | n | MdAPE | bias | cov50 | cov90 |
|---|---|---|---|---|---|
| VED_ICE | 645 | 0.070 | -0.033 | 0.44 | 0.85 |
| VED_HEV | 275 | 0.121 | +0.015 | 0.44 | 0.88 |
| VED_EV | 258 | 0.121 | +0.004 | 0.39 | 0.78 |
| VED_PHEV | 318 | 0.229 | -0.182 | 0.39 | 0.86 |
| EPA_conv_FTP | 2877 | 0.079 | -0.039 | 0.44 | 0.82 |
| EPA_conv_HWY | 2835 | 0.079 | +0.069 | 0.44 | 0.90 |
| EPA_conv_US06 | 371 | 0.070 | +0.051 | 0.43 | 0.82 |
| EPA_conv_FTPcold | 280 | 0.109 | -0.077 | 0.28 | 0.79 |
| EPA_conv_SC03 | 302 | 0.104 | -0.050 | 0.34 | 0.71 |
| EPA_hev_FTP | 70 | 0.062 | -0.053 | 0.57 | 0.77 |
| EPA_hev_HWY | 72 | 0.097 | +0.097 | 0.42 | 1.00 |
| EPA_bev_FTP | 98 | 0.065 | -0.027 | 0.56 | 0.87 |
| EPA_bev_HWY | 97 | 0.080 | +0.019 | 0.45 | 0.87 |
| Mersin_diesel | 56 | 0.126 | -0.029 | 0.34 | 0.64 |
| Mersin_cng | 56 | 0.189 | -0.189 | 0.16 | 0.38 |
| ICCT | 3 | 0.141 | +0.141 | 0.00 | 0.67 |
| ZTBus_legs | 5 | 0.123 | -0.112 | 0.20 | 1.00 |
| ZTBus_missions | 2 | 0.131 | -0.008 | 0.00 | 1.00 |

## Compute
Training 1.65 single-thread CPU-hours in all (r0 0.58, r1 0.54, r2 0.51 (rejected); + 0.02 for a speed benchmark and a 2-min first launch killed for memory; one thread, shared 2-core machine), measured with time.process_time() in the training process. Budget: about 4. Exam inference: see the registry record's estimate.

## numpy release check
numpy float64 (`predict.py`, each trip unpadded, chunked) vs the torch checkpoint of r1, row by row, on 296 rows (16 sim val trips, 24 VED and 24 EPA dev records, every HD dev record; longest 7,584 blocks = 6 chunks): max |log diff| 7.8e-16 against torch in float64, 2.0e-07 against torch in float32 (`work/npcheck.json`). Training-time check (padded torch batch vs unpadded numpy, 5 val trips): 1.6e-07.

## Limits
- Simulation only (RoadSim physics v2, EnergyNet v2.1's bank recipe): what the simulator does not model (real hybrid and plug-in control, A/C habits, ageing, driver behaviour beyond the speed trace) shows up as dev error. Plug-ins are the weakest dev set (VED PHEV MdAPE ~0.22 vs EnergyNet v2.1's 0.163).
- One third of EnergyNet v2.1's training data (one of its three banks) and a single network (EnergyNet: a 5-member ensemble on a GPU): the sim val gap to EnergyNet (skill ~0.64-0.66 vs 0.680) is partly data and ensembling, not only the input form.
- 5-s blocks keep mean v^2, v^3 and positive kinetic-energy increments but lose the exact sub-5-s sequence of accelerations and stops.
- From r1 on it needs EnergyNet's physics estimate at inference (bake.e_phys_from_trace from the harness); the r0 rung is the physics-free version.
- Heavy-duty traces beyond 7,200 s (ZTBus legs up to 37,919 s) are outside the training range: they are chunked and mean-pooled, an extrapolation checked only on the 5 ZTBus dev legs.
- Bands are the raw pinball-loss quantiles (sim val cov50 / cov90 ~0.50 / 0.89; no dev widening here; arm C's conformal layer calibrates them on dev).
- Sequential: numpy inference loops over the blocks (two directions), the slowest of the four on long trips.

