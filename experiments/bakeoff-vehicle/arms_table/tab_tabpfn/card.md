# tab_tabpfn: TabPFN in context on simulated rows (Vehicle Bake-off, table track)

**A lens, not a release.** Built with PriorLabs-TabPFN.

**What it is.** TabPFN v2 is a transformer pretrained by Prior Labs on synthetic tables. Here it is used purely in
context, with no training and no gradient step. It is shown 2,048 simulated trips (inputs + correction target), then asked
for the predictive distribution of each new trip.
- The context is simulated train-bank trips only: **zero real labels**.
- It is a yardstick: what a generic tabular foundation model reads from the same simulated bank, with no training.
- It is not a release candidate. It needs torch and the TabPFN package (`numpy_only: false`), and its licence and size
  differ from the collection's numpy releases.

**Inputs.** A scoped subset of the table track's 151 EnergyNet inputs (`featsel.py`):
- the 38 top-ranked inputs by the GBM arm's r0 gain ranking, computed on the simulated train bank only;
- architecture and class as two categorical codes, taken from the one-hot blocks.

The full 151 inputs needed 1.5 GB and 24 ms per row at 1,024 context rows, against 0.87 GB and 7 ms for 40 inputs. They
were also no better on a 512-trip probe (median |log error| 0.112 vs 0.105).

**Target and outputs.** z = log(E / E_phys). TabPFN returns quantiles 5 / 25 / 50 / 75 / 95 % of z; pred = the median;
E = E_phys x exp(z). No components.

**Settings.**
- TabPFN v2 regressor, `n_estimators` 1, `random_state` 0, CPU, one thread.
- `fit_mode` 'fit_with_cache': the context's keys and values are computed once at import, about 22 s CPU.
- Prediction runs in chunks of 512 rows, about 10.6 ms CPU per row.
- Context = `ctx.npz`: 2,048 random train trips (y > 0.02 MJ) under one tier draw (default_rng(5)). By architecture:
  conv 960, P2 180, series 82, power-split 168, plug-in power-split 95, plug-in series 62, BEV 501.
- Code: tabpfn 10.2.0 plus pydantic_settings, pydot, safetensors and skrub, copied into `pylib/` from the Prior Wars venv.
- Weights: `weights/tabpfn-v2-regressor.ckpt` (Prior Labs, from the Prior Wars download).

## Ratchet (rule: sim val skill > incumbent + 0.01 AND dev D16 <= incumbent + 0.01)

Settings were limited to those that keep one process at or under 1.2 GB and the exam at or under about 30 CPU-minutes.

| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | Decision |
|---|---|---|---|---|---|
| r0 | 1,024 random context trips, 40 inputs | 0.527 | 0.1110 | 0.1348 | base |
| r1 | 2,048 random context trips | **0.563** | **0.1017** | **0.1249** | **released** (+0.036) |
| r2 | 2,048 trips, half balanced over arch x class | 0.544 | 0.1067 | 0.1259 | rejected (-0.019) |
| r3 | 3,072 trips x 30 inputs (more rows, fewer columns) | 0.561 | 0.1013 | 0.1298 | rejected (-0.002) |

References: EnergyNet v2.1 0.680 / 0.1133; the MLP control 0.668 / 0.1203.

**What the rungs say.**
- Context rows are the lever, up to the memory limit; 4,096 rows needed 1.7 GB.
- Balancing the context toward rare architectures costs skill on the common ones.
- TabPFN's bands are the best calibrated of the four table families with no calibration step: sim val cov50 / cov90
  0.47 / 0.88; dev VED ICE 0.51 / 0.91, PHEV 0.53 / 0.94.
- Its point accuracy on the simulator is the lowest. On real dev it is close to the trained arms: D16 0.125 vs 0.120-0.128.

**Dev gate, r1 (MdAPE, bias, cov50 / cov90).**
- VED: ICE 0.066 (0.51 / 0.91); HEV 0.138; EV 0.135; PHEV 0.164 (-0.121 bias, 0.53 / 0.94).
- EPA conventional: FTP 0.083; HWY 0.115 (+0.110 bias); US06 0.185 (+0.185 bias); cold FTP 0.095; SC03 0.117.
- EPA hybrid FTP 0.122; EPA BEV city 0.069, highway 0.178.
- Heavy duty: Mersin diesel 0.106, CNG 0.147; ICCT 0.147.

**Compute.**
- No training (0 CPU-hours). Contexts take about 3 s each to build.
- The context forward pass at load costs about 22 s CPU in every process.
- Prediction: sim val 66 s CPU, dev gate 128 s.
- Exam estimate: about 121k rows, roughly 21 CPU-minutes.
- Peak memory in an evaluation process: about 1.1 GB.
- Setup probes of speed and memory (8 runs, about 0.13 CPU-hours) are not training.

**Limits.**
- Highway and US06 test cycles are biased high (EPA conventional HWY +11 %, US06 +19 %), and BEV highway is biased low
  (-18 %). The context holds few steady high-speed dynamometer rows.
- Point accuracy on the simulator is about 0.1 skill below the trained arms.
- Needs torch.
- The licence requires the attribution "Built with PriorLabs-TabPFN" (Prior Labs License 1.1: `weights/LICENSE.txt`,
  `pylib/tabpfn-10.2.0.dist-info/licenses/`).
