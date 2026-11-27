# QuakeNet v2. Can station metadata recover the exam answer?

Independent first pass, 2026-11-25. The model is the multi-station QuakeNet v2, release 1.0.
The original check used `2dd401ef79dffd29acad2b83d8b47a9567ecb395`. The model directory is identical at
`61c70bfb2a27f3d592fab317ccd40a048e020b4b`, the main revision used to start the audit branch.

## Catalogue-assisted location — CRITICAL DEFECT for an independent-location claim

**What failed?** The public API takes station coordinates, timestamped waveforms and a rough P-wave arrival for each station. In the exam, the catalogue epicentre, depth and origin time determine the theoretical P arrival at each station. In the shipped examples, adding the record start time to the arrival guess recovers that theoretical arrival. The model card discloses the catalogue guess. The question here is how much location information the guess supplies before the waveform is read. [Arrival calculation, lines 58–65](../../../../models/quakenet-v2/training/fetch2.py#L58) · [Example construction, lines 11–15](../../../../models/quakenet-v2/training/make_examples.py#L11) · [Model card limitation](../../../../models/quakenet-v2/model.yaml#L143)

**What changed when tested?** A least-squares fit used only station coordinates and relative arrival times. Subtracting the mean arrival removes the unknown origin time. The fit searched latitude, longitude and depth with the shipped ak135 travel-time table. It opened only the metadata field and consulted the event coordinates after fitting. On all four shipped ten-station examples, the fitted epicentre agreed with the catalogue to within five metres.

| Event | Metadata-only miss (m) | Released network miss (km) |
|---|---:|---:|
| us6000ngn7 | 4.24 | 103 |
| us7000kulj | 1.12 | 25 |
| us7000nx17 | 2.22 | 71 |
| usb000scur | 3.08 | 129 |

**What consequence remains unmeasured?** Metre-scale agreement measures consistency with the catalogue that generated the timestamps. The experiment establishes that a shortcut is available. The network's dependence on it and its location error on independently detected events remain open. A test of that use case would select stations and pick arrivals from continuous recordings, lock predictions, then join the catalogue for scoring.

## Split and rung checks

The manifest has 40 unique development IDs and 1,734 unique exam IDs, with zero overlap. All 400 multi-station IDs belong to the exam. These are manifest counts. The two-station exam reports 1,456 eligible events. Recomputing the simulated objective from its four logged components, then applying the strict 0.01 improvement margin and the 1.05 development-error ratio gate, reproduced all 15 candidate decisions across the three frozen sets. [Split manifest](../../../../models/quakenet-v2/splits.json) · [Promotion log](../../../../models/quakenet-v2/training/ratchet.json)

## Reproduce

Run from the repository root with NumPy and SciPy. The first command fits the metadata baseline and checks the recorded decisions. The second runs the released network on the same examples.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/quakenet_v2_probe.py
OPENBLAS_NUM_THREADS=1 python -B models/quakenet-v2/demo.py
```

**Review scope.** This pass covers protocol items 1, 5 and 8 and the event-ID portion of item 6. The remaining training, export, calibration and full-exam checks are open. The verdict above applies to interpreting this exam as evidence for independent location.
