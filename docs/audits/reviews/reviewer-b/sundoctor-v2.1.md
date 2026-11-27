# SunDoctor v2.1 · CSV order and missing daylight generation

**First pass · material reporting defects.** Reversing the row order of two shipped solar-meter extracts makes the public CSV reader return zero valid days and zero generation. Keeping the original order and masking only midday power leaves all 30 days valid at each site while measured generation disappears from the monthly total. Model snapshot `61c70bf`, audit checkout `ea2bfde`. [CSV reader](../../../../models/sundoctor/sundoctor.py#L250) · [Energy calculation](../../../../models/sundoctor/sundoctor.py#L191)

Reviewer B took the first 30 days of each packaged meter file and used its demo site settings. Each run used the same irradiance and temperature readings. The gaps remove five hours of power per day, local time.

| Input to the public reader | California fixed inverter · valid days / kWh | Colorado tracker · valid days / kWh |
|---|---:|---:|
| Original | 30 / 7,248.35 | 30 / 54,311.03 |
| Same rows newest first | 0 / 0.00 | 0 / 0.00 |
| Re-sorted by timestamp | 30 / 7,248.35 | 30 / 54,311.03 |
| 10:00–15:00 missing | 30 / 2,840.16 | 30 / 17,855.38 |
| 00:00–05:00 missing | 30 / 7,248.35 | 30 / 54,311.03 |

The CSV path parses timestamps but passes their existing order to interpolation and binary search, both of which expect ascending coordinates. Sorting restores the original results exactly. Separately, a day is counted when more than half of its grid has measurements. The kWh calculation sums known power and substitutes zero for missing intervals. As a result, the reported `kwh_made` falls **60.8% and 67.1%** with all days still counted. The night-gap control isolates which hours were lost. These are retrospective meter diagnoses. Neither test establishes an error on the original ordered exam records or a change in fault-detection accuracy. [Solar-time interpolation](../../../../models/sundoctor/sundoctor.py#L149) · [Summary](../../../../models/sundoctor/sundoctor.py#L241)

**Model controls.** All weights were finite with zero near-zero rows among 2,047 convolution and linear outputs. Reconstructing `training/model2.py` from the released folded weights gave maximum daily/global head errors of `6.20e-6 / 9.54e-7` on a real fortnight and `1.34e-5 / 1.91e-6` on a seeded random input. The random daily-head error exceeds the audit protocol’s strict `1e-5` absolute target. Both networks read the same input in that check.

All five v1 and four v2 logged decisions matched the simulation promotion rule. The v2.1 table records 1.482 → 1.564 (+0.082 against a +0.01 margin). The raw v2.1 scoring log and full frozen tensors were unavailable for independent rescoring. The v2 trainer seeds PyTorch and simulates withheld irradiance. Interval coverage on daytime power gaps, comparisons with independently metered monthly energy and a rerun of the multi-day outage twins are open. [Release ladder](../../../../models/sundoctor/ratchet.yaml) · [v2.1 scorer](../../../../models/sundoctor/training/ratchet21.py)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/sundoctor_v2_1_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wave5_export_probe.py sundoctor
```
