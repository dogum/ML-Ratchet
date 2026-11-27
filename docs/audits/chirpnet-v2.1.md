# ChirpNet v2.1 | Untimed BBH search verified; limited background exposure and O3 FAR bound

**Cross-checked clean BBH search controls with limited false-alarm exposure.** Reviewer A independently reran reviewer B's `d903cb6` probe against model base `61c70bf`, audited dead units across all four shipped weight archives (`finder`, `coinc`, `pe`, `bns`), and derived a clustered O3 false-alarm lower bound directly from archived per-piece O3 background maxima. [Original report and probe](reviews/reviewer-b/chirpnet-v2.1.md) · [Search & clustering](../../models/chirpnet/chirpnet.py#L202) · [O3 dev background](../../models/chirpnet/training/logs/o3_dev_background.json)

**What passed and reproduced.**
- **Untimed BBH search & shift controls:** Running the public scanner across all four packaged 32 s H1/L1 BBH strain files (`GW150914`, `GW151012`, `GW151226`, `GW170104`) at threshold `7.0` returned exactly one trigger per file (`11.48`, `9.55`, `10.13`, `9.83`) without catalogue timestamps. Shifting both whitened traces by `+4.000 s` moved every loudest trigger by `+4.000 s` within `1 µs`. Independently permuting samples within each detector dropped scores to `−3.53 .. −2.31`; zeroing both strain channels gave `−4.22`.
- **Catalogue-time vs. scan-trigger PE:** Replacing the catalogue time with the untimed scan trigger shifted median detector-frame chirp mass by at most **`0.53 M☉`** across the four events (`GW150914`: `32.35 -> 32.71`, `GW151012`: `18.44 -> 17.91`, `GW151226`: `9.86 -> 9.96`, `GW170104`: `24.14 -> 24.02`). Passing zero strain to the PE network still yields a broad prior-like posterior (`18.5–19.3 M☉`), confirming PE is conditional on an upstream detection.
- **Export parity & ratchets:** Folded Torch↔NumPy parity reproduced within `2.87e-6` (finder), `1.91e-6` (coinc), `7.16e-7` (PE), and `4.77e-7` / `1.20e-7` (BNS TF map / network). All 10 BBH rows and 3 BNS rows replayed under the rules in force at each row (with the caveat that the same four real BBH events were inspected repeatedly across development rungs, while numerical keep decisions used simulation metrics).

**False-alarm calibration & second-pass O3 lower bound.**
- **In-sample threshold tuning (`0.499 / h`):** The release archive contains `57,760` overlapping time-slide windows (`7,220 s = 2.006 h`, max score `6.197`). Because `release_build.py` sets threshold `7.0 = ceil_0.5(max + 0.5)`, zero exceedances on this sample are guaranteed by construction, and [`chirpnet.py:186`](../../models/chirpnet/chirpnet.py#L186) reports `1 / 2.006 h = 0.499 / h` with no confidence interval.
- **Independent O3 dev transfer & clustered lower bound:** [`o3_dev_background.json`](../../models/chirpnet/training/logs/o3_dev_background.json) logs **46 window exceedances** `>= 7.0` (max `10.875`) over `38,272 s` (`10.63 h`). Inspecting per-piece maxima in `pieces` shows **4 distinct 32 s O3 pieces** out of 64 exceeded `7.0` (`O3a_1245827072_1245829792`: `7.762`; `O3b_1262661632_1262662176`: `7.774`; `O3b_1262661632_1262665376`: `7.117`; `O3b_1264021504_1264024768`: `10.875`). Because separate GPS segments cannot merge under the scanner's local clustering, the observed count is at least four clusters across this finite time-slide sample, or **at least 0.376 clusters per hour of slide exposure** at threshold `7.0` (the model card already discloses the O3 degradation and notes `threshold_new = 11.5` under the same rule).
- **Second-pass weight health:** All numeric arrays across all four weight archives (`chirpnet_weights.npz`, `chirpnet_coinc_weights.npz`, `chirpnet_pe_weights.npz`, `chirpnet_bns_weights.npz`) are finite with **0 dead active units** (in `chirpnet_pe_weights.npz`, exactly 6 of 12 output rows in each `f0w2..f7w2` are zero, matching the masked pass-through half of the 6-D RealNVP coupling layers).

The 10.63 hours count shifted pairings of 64 short pieces. The four-piece minimum places no statistical lower confidence bound on a continuously operating detector’s alert rate. [Exposure calculation](../../models/chirpnet/training/o3_dev_background.py#L131)

**What remains open.** Exact clustered event-level FAR with uncertainty intervals on continuous O3 background, online causal whitening latency (packaged files use offline whole-record whitening), and reproducing the external GW170817 matched-filter benchmark require raw GWOSC archives outside the repository.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/chirpnet_v2_1_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_d903cb6_probe.py chirpnet
```

Uses NumPy, SciPy, h5py, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_d903cb6_probe.py)
