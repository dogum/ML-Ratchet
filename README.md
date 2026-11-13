<div align="center">

<img src="site/public/favicon.svg" width="56" alt="" />

# ML Ratchet

**Trained on physics. Graded on reality.**

An open collection of machine-learning models that learned only from physics simulations, moved up a rung
only when they beat a frozen test *and* a small real dev set, and were graded once on real data they never saw.
The losses are published too.

[**Website**](https://dogum.github.io/ML-Ratchet/) · [Run them live](https://dogum.github.io/ML-Ratchet/#live) · [The story](https://dogum.github.io/ML-Ratchet/story/) · [The method](docs/method.md) · [Scoreboard](https://dogum.github.io/ML-Ratchet/scoreboard/) · [Add a model](docs/adding-a-model.md) · [Backlog](docs/backlog.md)

</div>

---

## The collection

| Model | The question it answers | Real exam | Record vs baselines |
|---|---|---|---|
| [**ChirpNet**](models/chirpnet/) | Is there a black-hole or neutron-star merger in this LIGO data, and how heavy was it? | 13 events from O1–O3 + time-slid background | 8 wins · 3 losses |
| [**SunDoctor**](models/sundoctor/) | Why did this solar array make less power today, and what did each cause cost? | 2 real PV farms, independent answer keys | 4 wins · 0 losses · 1 tie |
| [**CellSense**](models/cellsense/) | How worn is this battery cell, why, and how many cycles are left? (LFP, and now NMC811) | 118 LFP cells + 7 NMC811 cells | 3 wins · 4 losses · 2 mixed · 1 tie |
| [**TransitNet**](models/transitnet/) | Is this dip in a TESS light curve a planet, and how big is it? | 1,196 planets + 832 false positives | 3 wins · 2 losses |
| [**QuakeNet**](models/quakenet/) | How big, how far, how deep, which way — from one seismometer? | 3,105 one-station records | 4 wins · 1 loss |
| [**BearingNet**](models/bearingnet/) | Healthy, or an outer-race, inner-race or ball defect? | 2 rigs, one never heard | 3 wins · 3 losses |
| [**SwellCast**](models/swellcast/) | How far away was the storm whose swell is arriving at this buoy? | 288 hurricane-swell arrivals | 2 wins · 2 losses |
| [**BeatNet**](models/beatnet/) | Is each heartbeat normal, supraventricular, ventricular or fusion? *(research only)* | MIT-BIH DS2, 49,705 beats | 3 wins · 2 losses |
| [**StarQuake**](models/starquake/) | How massive and how big is this red giant, from how it rings? | 420 Kepler giants × 5 lengths | 4 wins · 1 loss · 1 tie |
| [**EnergyNet**](models/energynet/) | How much fuel or electricity will this trip take, and where does it go? | 25,793 VED trips, EPA tests, buses, trucks | 5 wins · 5 losses |
| [**TripDetective**](models/tripdetective/) | From a trip log alone: what powertrain, how heavy, what engine, how much drag? | 351 real cars | 3 wins · 1 loss |
| [**HybridPilot**](models/hybridpilot/) | How should a hybrid run its engine, second by second? *(graded in simulation)* | 430 real routes, simulated twins | 2 wins · 2 losses · 1 tie |

Every model also runs **live in your browser** on the website — the real network, ported to TypeScript and checked
against Python, on real exam records, with controls that change the input so you can watch the reading respond.

**0** real labelled examples were used for training. **93** rung decisions are logged, **41** of them rejections.
Every model runs on numpy in seconds and ships real exam examples, so you can check the numbers yourself. The three
vehicle models share one simulator and dataset: [the vehicle platform](docs/vehicle-platform.md).

## Try it

```bash
git clone https://github.com/dogum/ML-Ratchet && cd ML-Ratchet
pip install -e .                 # numpy, scipy, pyyaml; add ".[models]" for pandas + h5py (SunDoctor, ChirpNet)

mlratchet list                   # every model and its exam record
mlratchet info quakenet          # the card: reads, returns, exam, wins, losses, limits
mlratchet ladder quakenet        # every rung it tried, kept or not, and why
mlratchet demo quakenet          # run it on four real earthquakes
mlratchet scoreboard             # wins and losses across the collection
```

Or use one model on its own — each folder is self-contained:

```bash
cd models/quakenet && pip install numpy scipy && python demo.py
```

```python
import mlratchet
QuakeNet = mlratchet.load("quakenet").QuakeNet     # imports in isolation; models never collide
```

## The method in six lines

1. **Real physics**, checked against an independent reference (QuakeNet vs Syngine: r = 0.9999).
2. **Simulated signals laid into real instrument noise** — quiet seismometers, healthy machines, quiet stars.
3. **Train only on simulations.** No real labelled example is ever a training sample.
4. **The ratchet.** Keep a rung only if it beats the incumbent on a frozen simulated set by more than a margin
   (usually 0.01) AND passes a gate on a handful of real dev units. Log every rung, kept or not.
5. **Calibrate the simulator on dev only.** The measured mismatch becomes the next rung; the twin test finds it.
6. **One final exam** on real data that chose nothing — beside a classical method and a dumb baseline, with error-bar
   coverage and where it fails.

Read [docs/method.md](docs/method.md) for the why, and [docs/lessons.md](docs/lessons.md) for what the first nine models taught.

## Repository layout

```
models/<id>/          one folder per model, all the same shape (docs/model-contract.md)
  model.yaml            the structured card — README, CLI and website are generated from it
  ratchet.yaml          every rung, kept or not, and why
  <id>.py, *.npz        numpy-only inference and weights
  demo.py, examples/    a demo on real exam examples, in seconds
  MODEL_CARD.md         the full card; NOTEBOOK.md is the build story
  training/             every script that built it, as run (paths via training/_paths.py)
  lab/index.html        the original interactive page; the network runs in the browser
mlratchet/            the Python package: registry, CLI, schemas, and the ratchet building blocks
  ratchet.py            Ratchet.judge / rescore and ready-made dev gates
  calibrate.py          widen-only error bars and coverage
  metrics.py, nn.py     skills and dumb baselines; numpy layers and BatchNorm folding
  templates/model/      the scaffold behind `mlratchet new`
docs/                 the method, the contract, lessons, the agent brief, data sources, backlog, roadmap
site/                 the website (Astro + TypeScript), built from models/ and docs/, deployed to GitHub Pages
  src/instruments/      each model's network running live in the browser (parity-tested against Python)
shared/vehicle/       a shared platform: RoadSim (whole-vehicle simulator) + RoadData (real trips, frozen splits)
tests/                core tests, the contract, the vehicle platform, and every demo against its golden output
```

## Add a model

```bash
pip install -e ".[dev]"
mlratchet new gridpulse --name GridPulse
mlratchet validate gridpulse
```

Then follow [docs/adding-a-model.md](docs/adding-a-model.md). Building with an AI research agent? Hand it
[docs/agent-brief.md](docs/agent-brief.md) — the last four models were built by parallel agents from that brief.
Twelve candidates with physics, data and an exam already lined up are in [the backlog](docs/backlog.md).

**Help wanted:** CellSense's other chemistries are waiting on data (ChirpNet's O3 exam, once blocked the same way,
is now done: 7 of 7 in-range events found). See [docs/roadmap.md](docs/roadmap.md).

## Licence and data

Code, weights and docs: [MIT](LICENSE). The real data is not ours — small excerpts are bundled so demos run
offline, and they stay under their providers' terms ([DATA_LICENSES.md](DATA_LICENSES.md)). Please cite the
providers. **BeatNet is for research and teaching only and is not a medical device.**

If you use ML Ratchet, please cite it ([CITATION.cff](CITATION.cff)) and the data behind the model you used.
