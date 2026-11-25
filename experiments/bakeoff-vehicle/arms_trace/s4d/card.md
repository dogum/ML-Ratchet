# S4D state-space model on the trip trace (`trace_s4d`)

Vehicle Bake-off, **trace track**, group B (trace learners, `arms_trace/`). Family: **State-space and recurrent models (S4D diagonal state space)** (`ssm` on the Model families page). numpy-only inference (`predict.py` -> `../common/tracelib.py` + `weights.npz`). Released rung: **r2**. Zero real labels in training.

## The built-in assumption
Memory is linear and time-invariant: each channel is filtered by a sum of decaying and oscillating modes (S4D, 16 complex modes per channel, ZOH, step sizes 1e-3 to 1e-1 per block, i.e. memories of about 10 to 1,000 blocks), one filter running forwards and one backwards over the trip; the nonlinearity sits only between layers (GELU, GLU) and in the shared input MLP stem.

## Network
core: 3 x [LayerNorm -> bidirectional S4D (N2 16, forward and backward kernels, D skip) -> GELU -> Linear(48, 96) -> GLU], residual. Parameters: 48,762 (the four trace arms hold 49k-86k).

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
| r0 | base: pure trace learner; anchor A0 = sim-train median MJ/km (architecture x class) x distance, no physics estimate; 12 epochs | 0.616 | 0.0882 | 0.1149 | 0.077 / 0.117 / 0.143 / 0.282 | 0.087 / 0.078 | 0.115 / 0.184 | 0.140 | 0.96 | **base** | first rung: what S4D finds from the raw trace alone (skill 0.616, D16 0.115) |
| r1 | physics anchor: target log(E / E_phys) with E_phys recomputed from the trace inputs (bake.e_phys_from_trace), log(E_phys / A0) as an input; 12 epochs from scratch | 0.645 | 0.0844 | 0.1200 | 0.074 / 0.119 / 0.124 / 0.231 | 0.094 / 0.071 | 0.125 / 0.212 | 0.185 | 0.90 | **kept** | sim val skill +0.028 over r0 (bootstrap +0.011..+0.038); D16 0.1200 vs 0.1149, worse by 0.005, inside the 0.01 tolerance: kept |
| r2 | 2x training: warm restart from r1 for 12 more epochs (one-cycle cosine, peak lr 1e-3, data stream seed 1) | 0.655 | 0.0806 | 0.1187 | 0.071 / 0.130 / 0.118 / 0.239 | 0.099 / 0.072 | 0.119 / 0.192 | 0.154 | 0.87 | **released** | sim val skill +0.0103 over r1 (bootstrap +0.003..+0.020), just over the 0.01 margin; D16 0.1187 vs 0.1200: kept; released |

References (same harness): EnergyNet v2.1 (table track, 3x the training data, 5-member ensemble) sim val skill 0.680, D16 0.1133; E_phys alone 0.048 / 0.2278; climatology 0 / 0.3093.

## Released rung r2
Sim val (3,985 trips): skill 0.655, MdAPE 0.0806, bias +0.0016, cov50 / cov90 0.50 / 0.89, crps_log 0.0813. MdAPE by architecture: conv 0.063, hev_parallel_p2 0.099, hev_series 0.096, hev_powersplit 0.083, phev_powersplit 0.212, phev_series 0.188, bev 0.096.

Dev gate (MdAPE, cov50 / cov90; raw bands, no dev widening):

| Set | n | MdAPE | bias | cov50 | cov90 |
|---|---|---|---|---|---|
| VED_ICE | 645 | 0.071 | -0.039 | 0.40 | 0.83 |
| VED_HEV | 275 | 0.130 | +0.010 | 0.41 | 0.85 |
| VED_EV | 258 | 0.118 | -0.030 | 0.44 | 0.77 |
| VED_PHEV | 318 | 0.239 | -0.206 | 0.35 | 0.74 |
| EPA_conv_FTP | 2877 | 0.099 | -0.017 | 0.36 | 0.79 |
| EPA_conv_HWY | 2835 | 0.072 | +0.037 | 0.54 | 0.90 |
| EPA_conv_US06 | 371 | 0.095 | +0.094 | 0.40 | 0.78 |
| EPA_conv_FTPcold | 280 | 0.143 | -0.092 | 0.28 | 0.78 |
| EPA_conv_SC03 | 302 | 0.110 | -0.040 | 0.32 | 0.76 |
| EPA_hev_FTP | 70 | 0.086 | -0.086 | 0.47 | 0.76 |
| EPA_hev_HWY | 72 | 0.085 | +0.080 | 0.56 | 1.00 |
| EPA_bev_FTP | 98 | 0.095 | -0.095 | 0.43 | 0.84 |
| EPA_bev_HWY | 97 | 0.092 | -0.092 | 0.35 | 0.67 |
| Mersin_diesel | 56 | 0.119 | -0.036 | 0.32 | 0.64 |
| Mersin_cng | 56 | 0.192 | -0.192 | 0.11 | 0.38 |
| ICCT | 3 | 0.154 | +0.152 | 0.00 | 0.00 |
| ZTBus_legs | 5 | 0.139 | -0.139 | 0.60 | 0.80 |
| ZTBus_missions | 2 | 0.098 | -0.097 | 0.50 | 1.00 |

## Compute
Training 2.75 single-thread CPU-hours in all (r0 0.96, r1 0.90, r2 0.87; + 0.02 for speed benchmarks and a 2-min first launch killed for memory; one thread, shared 2-core machine), measured with time.process_time() in the training process. Budget: about 4. Exam inference: see the registry record's estimate.

## numpy release check
numpy float64 (`predict.py`, each trip unpadded, chunked) vs the torch checkpoint of r2, row by row, on 296 rows (16 sim val trips, 24 VED and 24 EPA dev records, every HD dev record; longest 7,584 blocks = 6 chunks): max |log diff| 1.0e-15 against torch in float64, 2.0e-06 against torch in float32 (`work/npcheck.json`). Training-time check (padded torch batch vs unpadded numpy, 5 val trips): 2.8e-06.

## Limits
- Simulation only (RoadSim physics v2, EnergyNet v2.1's bank recipe): what the simulator does not model (real hybrid and plug-in control, A/C habits, ageing, driver behaviour beyond the speed trace) shows up as dev error. Plug-ins are the weakest dev set (VED PHEV MdAPE ~0.22 vs EnergyNet v2.1's 0.163).
- One third of EnergyNet v2.1's training data (one of its three banks) and a single network (EnergyNet: a 5-member ensemble on a GPU): the sim val gap to EnergyNet (skill ~0.64-0.66 vs 0.680) is partly data and ensembling, not only the input form.
- 5-s blocks keep mean v^2, v^3 and positive kinetic-energy increments but lose the exact sub-5-s sequence of accelerations and stops.
- From r1 on it needs EnergyNet's physics estimate at inference (bake.e_phys_from_trace from the harness); the r0 rung is the physics-free version.
- Heavy-duty traces beyond 7,200 s (ZTBus legs up to 37,919 s) are outside the training range: they are chunked and mean-pooled, an extrapolation checked only on the 5 ZTBus dev legs.
- Bands are the raw pinball-loss quantiles (sim val cov50 / cov90 ~0.50 / 0.89; no dev widening here; arm C's conformal layer calibrates them on dev).
- Linear time-invariant filters: what the trip does is weighted the same wherever it happens; interactions come only through the pointwise nonlinearities. Its D16 drifted up from r0 (0.1149) to r2 (0.1187) while sim skill rose; each step was inside the 0.01 tolerance.

