# Transformer on the trip trace (20-s patch tokens) (`trace_transformer`)

Vehicle Bake-off, **trace track**, group B (trace learners, `arms_trace/`). Family: **Attention and transformers** (`attn` on the Model families page). numpy-only inference (`predict.py` -> `../common/tracelib.py` + `weights.npz`). Released rung: **r2**. Zero real labels in training.

## The built-in assumption
Any part of the trip can look at any other part: 20-s patch tokens (4 blocks each) attend to every other token of the trip (3 pre-LN layers, 4 heads of 12), with only a sinusoidal position code for order. Nothing is built in about locality or time direction; it has to be learned.

## Network
core: 20-s patch embedding Linear(4 x 48 -> 48) + sinusoidal positions -> 3 x pre-LN encoder block (4 heads, d 48, feed-forward 96, GELU, non-causal, padding masked) -> tokens repeated back to their 4 blocks and added to the stem output. Parameters: 86,394 (the four trace arms hold 49k-86k).

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
| r0 | base: pure trace learner; anchor A0 = sim-train median MJ/km (architecture x class) x distance, no physics estimate; 12 epochs | 0.608 | 0.0914 | 0.1210 | 0.080 / 0.113 / 0.155 / 0.304 | 0.102 / 0.074 | 0.116 / 0.179 | 0.153 | 0.18 | **base** | first rung: what the transformer finds from the raw trace alone (skill 0.608, D16 0.121) |
| r1 | physics anchor: target log(E / E_phys) with E_phys recomputed from the trace inputs (bake.e_phys_from_trace), log(E_phys / A0) as an input; 12 epochs from scratch | 0.630 | 0.0866 | 0.1087 | 0.074 / 0.117 / 0.119 / 0.215 | 0.094 / 0.060 | 0.136 / 0.171 | 0.232 | 0.17 | **kept** | sim val skill +0.0215 over r0 (bootstrap +0.006..+0.035) and D16 0.1087 better than r0 0.1210: kept |
| r2 | 3x training (the River lesson: transformers need more steps): warm restart from r1 for 24 more epochs (one-cycle cosine, peak lr 1e-3, data stream seed 1); 36 epochs in all | 0.644 | 0.0829 | 0.1135 | 0.067 / 0.124 / 0.129 / 0.220 | 0.092 / 0.062 | 0.135 / 0.172 | 0.162 | 0.33 | **released** | sim val skill +0.0145 over r1 (bootstrap +0.004..+0.025); D16 0.1135 vs 0.1087, worse by 0.005, inside the 0.01 tolerance: kept; released (r3 did not beat it) |
| r3 | 10-s patch tokens (2 blocks) instead of 20-s (4 blocks): finer attention resolution, the only place the transformer coarsens the shared 5-s input; 36 epochs from scratch (= r2's total training), peak lr 2e-3, data seed 0 | 0.640 | 0.0837 | 0.1122 | 0.072 / 0.126 / 0.135 / 0.200 | 0.104 / 0.084 | 0.135 / 0.187 | 0.171 | 0.90 | **rejected** | sim val skill -0.0037 vs r2 (bootstrap -0.017..+0.010): no gain at 2.7x the CPU of r2; D16 0.1122 vs 0.1135 is a move under 0.01 (seed noise): rejected |

References (same harness): EnergyNet v2.1 (table track, 3x the training data, 5-member ensemble) sim val skill 0.680, D16 0.1133; E_phys alone 0.048 / 0.2278; climatology 0 / 0.3093.

## Released rung r2
Sim val (3,985 trips): skill 0.644, MdAPE 0.0829, bias +0.0037, cov50 / cov90 0.48 / 0.87, crps_log 0.0830. MdAPE by architecture: conv 0.064, hev_parallel_p2 0.111, hev_series 0.103, hev_powersplit 0.083, phev_powersplit 0.224, phev_series 0.211, bev 0.097.

Dev gate (MdAPE, cov50 / cov90; raw bands, no dev widening):

| Set | n | MdAPE | bias | cov50 | cov90 |
|---|---|---|---|---|---|
| VED_ICE | 645 | 0.067 | -0.028 | 0.43 | 0.83 |
| VED_HEV | 275 | 0.124 | +0.033 | 0.40 | 0.87 |
| VED_EV | 258 | 0.129 | +0.034 | 0.37 | 0.69 |
| VED_PHEV | 318 | 0.220 | -0.177 | 0.33 | 0.77 |
| EPA_conv_FTP | 2877 | 0.092 | -0.032 | 0.39 | 0.80 |
| EPA_conv_HWY | 2835 | 0.062 | +0.021 | 0.54 | 0.93 |
| EPA_conv_US06 | 371 | 0.097 | +0.097 | 0.32 | 0.81 |
| EPA_conv_FTPcold | 280 | 0.118 | -0.082 | 0.31 | 0.78 |
| EPA_conv_SC03 | 302 | 0.103 | -0.011 | 0.29 | 0.67 |
| EPA_hev_FTP | 70 | 0.110 | -0.110 | 0.49 | 0.76 |
| EPA_hev_HWY | 72 | 0.074 | +0.024 | 0.78 | 1.00 |
| EPA_bev_FTP | 98 | 0.076 | -0.068 | 0.48 | 0.84 |
| EPA_bev_HWY | 97 | 0.076 | -0.076 | 0.40 | 0.76 |
| Mersin_diesel | 56 | 0.135 | +0.016 | 0.23 | 0.62 |
| Mersin_cng | 56 | 0.172 | -0.172 | 0.05 | 0.39 |
| ICCT | 3 | 0.162 | +0.162 | 0.33 | 0.33 |
| ZTBus_legs | 5 | 0.126 | -0.113 | 0.40 | 0.80 |
| ZTBus_missions | 2 | 0.136 | -0.010 | 0.00 | 1.00 |

## Compute
Training 1.64 single-thread CPU-hours in all (r0 0.19, r1 0.17, r2 0.33, r3 0.90 (rejected); + 0.05 for a 1-epoch smoke run (0.015), a speed benchmark, a 2-min first launch killed for memory and r3 steps lost to a container restart; one thread, shared 2-core machine), measured with time.process_time() in the training process. Budget: about 4. Exam inference: see the registry record's estimate.

## numpy release check
numpy float64 (`predict.py`, each trip unpadded, chunked) vs the torch checkpoint of r2, row by row, on 296 rows (16 sim val trips, 24 VED and 24 EPA dev records, every HD dev record; longest 7,584 blocks = 6 chunks): max |log diff| 2.1e-09 against torch in float64, 3.1e-07 against torch in float32 (`work/npcheck.json`). Training-time check (padded torch batch vs unpadded numpy, 5 val trips): 3.6e-07.

## Limits
- Simulation only (RoadSim physics v2, EnergyNet v2.1's bank recipe): what the simulator does not model (real hybrid and plug-in control, A/C habits, ageing, driver behaviour beyond the speed trace) shows up as dev error. Plug-ins are the weakest dev set (VED PHEV MdAPE ~0.22 vs EnergyNet v2.1's 0.163).
- One third of EnergyNet v2.1's training data (one of its three banks) and a single network (EnergyNet: a 5-member ensemble on a GPU): the sim val gap to EnergyNet (skill ~0.64-0.66 vs 0.680) is partly data and ensembling, not only the input form.
- 5-s blocks keep mean v^2, v^3 and positive kinetic-energy increments but lose the exact sub-5-s sequence of accelerations and stops.
- From r1 on it needs EnergyNet's physics estimate at inference (bake.e_phys_from_trace from the harness); the r0 rung is the physics-free version.
- Heavy-duty traces beyond 7,200 s (ZTBus legs up to 37,919 s) are outside the training range: they are chunked and mean-pooled, an extrapolation checked only on the 5 ZTBus dev legs.
- Bands are the raw pinball-loss quantiles (sim val cov50 / cov90 ~0.50 / 0.89; no dev widening here; arm C's conformal layer calibrates them on dev).
- Attention cost grows with the square of the number of tokens (fine up to 1,440 blocks per chunk).
- Finer tokens did not help: 10-s patches (r3, 36 epochs from scratch) scored sim val skill 0.640 vs the released 20-s patches' 0.644, so the 20-s patch is the released resolution.

