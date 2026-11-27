# ChirpNet v2.1 | Does the search need the catalogue time?

**Reviewer B, first pass.** Model base `61c70bf`, audit checkout `fb19db6`. The packaged binary-black-hole examples support the search and parameter-inference claims for those four events. The clearest limitation is how the false-alarm rate is interpreted outside the time-slide background used to set the threshold.

## Search without the catalogue timestamp

I ran the public scanner across each 32-second H1/L1 strain pair with the release cadence and threshold of 7.0. Each returned one trigger. I then shifted both whitened detector traces by four seconds, leaving the file timestamps fixed. The loudest trigger moved four seconds on all four files, within one microsecond. Independently permuting samples **within each detector** in the catalogue-centered window preserved its sample distribution and dropped its score below threshold on all four files (−3.53 to −2.31). Zeroing both strain channels gave −4.22. This control tests whether the finder uses coherent strain structure. [Search and clustering](../../../../models/chirpnet/chirpnet.py#L202) · [Shipped strain provenance](../../../../models/chirpnet/examples/README.md)

| Event | Untimed scan score | PE median with catalogue time | PE median with scan time |
|---|---:|---:|---:|
| GW150914 | 11.48 | 32.35 | 32.71 |
| GW151012 | 9.55 | 18.44 | 17.91 |
| GW151226 | 10.13 | 9.86 | 9.96 |
| GW170104 | 9.83 | 24.14 | 24.02 |

PE values are detector-frame chirp mass in solar masses, using the same 1,024 latent draws and each pair's measured noise spectrum. The original rung evaluator takes the loudest window in a catalogue-time neighborhood, and one PE evaluation helper centers on the catalogue time. The published PE table builder uses the scan trigger. The interactive instrument exporter centers its default demonstration window using the catalogue time. Replacing the catalogue time with that trigger moved the median by at most **0.53 solar masses** in these tests. Placing the merger at 0.85 or 1.75 seconds in the two-second window gave medians of 17.87–18.88 for GW151012, the largest spread among these four. This tests four event-containing files, with whole-record offline whitening. Continuous-stream efficiency, detection latency and population-wide placement sensitivity remain open. [Rung evaluator](../../../../models/chirpnet/training/realeval.py#L40) · [PE evaluator](../../../../models/chirpnet/training/eval_pe.py#L45) · [PE table builder](../../../../models/chirpnet/training/build_pe_web.py#L18) · [Instrument exporter](../../../../models/chirpnet/web/export_instrument.py#L115)

## What the reported false-alarm number measures

The release archive has 57,760 overlapping background windows and 7,220 seconds (2.006 hours) of effective time-slide exposure. Its maximum score is 6.197. The threshold rule rounds the maximum plus 0.5 upward to the next half-unit, yielding **7.0**. Zero exceedances at the chosen threshold are therefore guaranteed on this calibration sample. The API returns **0.499 per hour** as one count divided by that exposure. It supplies no confidence level. Even with independent Poisson counts and a threshold fixed in advance, zero counts over that exposure would give a one-sided 95% upper limit of 1.49 per hour. Those assumptions do not supply an interval for this tuned, overlapping-window search. [Threshold construction](../../../../models/chirpnet/training/release_build.py#L12) · [Rate calculation](../../../../models/chirpnet/chirpnet.py#L186)

The archived independent O3 dev background is a direct check of transfer. It reports **46 windows** at or above 7.0 in 10.63 time-slide hours, with a maximum of 10.875. These are window exceedances. The public scanner groups nearby windows into candidates, so counting distinct false alerts requires the same grouping on independent background. The model card already discloses the O3 degradation. An event-level false-alarm rate with an uncertainty statement remains unverified. [O3 dev results](../../../../models/chirpnet/training/logs/o3_dev_background.json) · [Model card](../../../../models/chirpnet/MODEL_CARD.md#L109)

## Export and validation

- All numeric arrays in the finder, coincidence checker, posterior estimator and neutron-star network were finite. On the four real BBH windows, the folded Torch finder, checker and fixed-latent PE samples matched NumPy within `2.87e-6`, `1.91e-6` and `7.16e-7` respectively. A seeded synthetic neutron-star input gave time-frequency-map and network-output differences of `4.77e-7` and `1.20e-7`. This verifies release arithmetic. Reconstructing the original BatchNorm fold or reproducing the real GW170817 matched-filter result requires other artifacts.
- The BBH check replays all ten logged baseline, candidate and rejudgment rows under the rule in force at each row. The three BNS decisions also match. The BBH rule evolved during development and the same real events appear in successive logs. Published numerical keep decisions use simulation metrics. The repeated readouts do not establish that the four events served as a blinded final exam. [BBH decisions](../../../../models/chirpnet/training/logs/ratchet_log.json) · [BNS decisions](../../../../models/chirpnet/training/logs/bns_ratchet_log.json)
- The PE model conditions on a detected signal. With zero strain but each event's original noise spectrum, it still returned a broad posterior with median chirp mass of 18.5–19.3 solar masses. This is consistent with its injection training distribution. Parameter output in isolation is insufficient evidence of detection. Simulation-based posterior coverage, BNS widening and fresh-seed training were outside this replay. [PE sampling](../../../../models/chirpnet/training/pe_sampler.py) · [Release calibration](../../../../models/chirpnet/MODEL_CARD.md#L122)

Run from the repository root with NumPy, SciPy, h5py and CPU PyTorch.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/chirpnet_v2_1_probe.py
```
