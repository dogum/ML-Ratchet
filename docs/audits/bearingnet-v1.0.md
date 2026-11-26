# BearingNet v1.0 · Missing vibration yields a healthy label

**Cross-checked critical public-inference defect.** Inserting a single NaN at the midpoint of each of the 14 shipped recordings made the default reader return `N` (healthy) on all 14, with all four probabilities NaN. Ten of those recordings have a documented fault. The reader transforms windows, averages log probabilities across windows and applies `argmax`. Nonfinite scores propagate through the aggregation, and the first class wins. Model snapshot `61c70bf`, audit checkout `73ee7e6`. [Public reader](../../models/bearingnet/bearingnet.py#L58) · [First-pass report and missing-sensor controls](reviews/reviewer-a/bearingnet-v1.0.md)

| Shipped recordings | First window | Default aggregation |
|---|---:|---:|
| All labels correct | 6/14 | 5/14 |
| Healthy recordings correctly called healthy | 0/4 | 0/4 |
| Healthy recordings called outer-race fault | 3/4 | 4/4 |

The two Paderborn healthy recordings both come from bearing K005. With one window, CWRU file 97 switched to a ball-fault label and still raised a false alarm. Using multiple windows did not resolve the healthy-class errors in this demo. Its stratified sample is too small to estimate a deployment false-alarm rate. Actual missing-sample prevalence in the full exam remains unmeasured.

**Split check.** The eight distinct CWRU exam file IDs in the examples have zero overlap with the ten archived dev IDs. The five distinct Paderborn example bearing IDs have zero overlap with the documented three dev bearings (K006, KA01, KI03) and the three background bearings (K001–K003). The Paderborn loader excludes those six by bearing ID. The CWRU loader partitions by *file* in each fault stratum. Separate recording IDs alone do not establish separate physical bearings. The original CWRU catalogue and full split manifest were unavailable here. [Split builder](../../models/bearingnet/training/realdata.py#L26) · [Archived dev IDs](../../models/bearingnet/training/bn_devprobe_r2.json) · [Paderborn roster](../../models/bearingnet/NOTEBOOK.md#L18)

Reviewer A’s original weight, folded forward and four-rung probe also passed. Dead-sensor zero traces produced outer-race alarms. The clearest separation of a software error from ordinary model error is the healthy string paired with a wholly nonfinite posterior.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/bearingnet_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/peer_73ee7e6_probe.py bearingnet
```
