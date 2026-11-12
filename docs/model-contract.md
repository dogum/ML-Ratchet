# The model contract

Every model is one folder under `models/` with the same layout. A released model is self-contained: copy the
folder anywhere, `pip install numpy scipy`, and `python demo.py` runs on real exam examples in seconds.
`mlratchet validate` checks the contract; CI runs it on every push.

```
models/<id>/
├── README.md            generated from model.yaml + ratchet.yaml (mlratchet readme --write)
├── model.yaml           the structured card — schema: mlratchet/schema/model.schema.json
├── ratchet.yaml         every rung tried, kept or not, and why — schema: mlratchet/schema/ratchet.schema.json
├── MODEL_CARD.md        the long-form card: inputs, outputs, physics, network, ratchet, exam, limits, credits
├── NOTEBOOK.md          the lab notebook: how it was built, dead ends, data quirks, gotchas
├── <id>.py              inference — numpy (+ scipy) only; never imports mlratchet
├── pipeline.py          optional: reading and preprocessing raw instrument data
├── <id>_weights.npz     weights (BatchNorm folded), with a JSON `meta` entry
├── demo.py              runs on examples/ in seconds
├── examples/            a few real exam examples (fixed random draw within strata)
├── training/            every script that built it, as run
│   └── _paths.py        portable paths: code here, outputs in ../work or $ML_RATCHET_WORK/<id>
├── lab/                 optional
│   ├── index.html       the interactive page, self-contained (the network runs in the browser)
│   └── src/             its sources
└── work/                git-ignored: downloads, banks, checkpoints, builds
```

## model.yaml

The single source of truth for what a model is and how it did. The README, the CLI and the website all read it.
Key fields:

| Field | What |
|---|---|
| `id`, `name`, `version`, `status` | `status` is `released`, `partial` (released, part of the exam waits on data), `in-progress` or `idea` |
| `domain`, `tags`, `accent` | `accent` is the one colour the site uses for this model |
| `tagline`, `question`, `summary` | Plain words. The question is the one thing it answers. |
| `reads`, `returns` | Inputs with units; each output with its error bar |
| `physics.summary/simulator/checks` | The governing physics, what the simulator includes, and independent checks **with numbers** |
| `noise`, `network` | Where the noise came from; the architecture |
| `data[]` | Every source with URL, licence and role (truth, noise, dev, exam, physics) |
| `split.dev`, `split.exam` | Who is in which, and the seed |
| `exam.results[]` | One row per headline metric: model value, baselines (`classical`, `dumb`, `published`), and a `verdict` — `win`, `loss`, `tie`, `mixed` or `info` |
| `exam.coverage[]` | Exam coverage of the nominal 68% bars |
| `wins`, `losses`, `limits` | Said plainly. Every model loses somewhere; the card says where. |
| `help_wanted`, `next` | Blocked data, and the next rungs worth trying |
| `usage`, `files`, `demo_seconds` | How to run it |

**Verdicts are honest by construction.** A row is a `win` only if the model beats every baseline listed on that row.
If the dumb baseline wins, the verdict is `loss`, even if the classical method lost too.

## ratchet.yaml

```yaml
model: quakenet
ladders:
  - name: QuakeNet
    objective: Mean skill over Mw, distance, log depth and back-azimuth on the frozen simulated set
    rule: Keep if … beats the incumbent by more than 0.01 AND the dev records get no more than 5% worse.
    margin: 0.01
    dev_gate: Dev-record errors may not get more than 5% worse
    frozen_sets: [val, val2, …, val8]   # every version the rungs were re-scored on
    scored_on: val8
    columns_source: Real exam, reported after the fact; never used to choose
    columns: ["Mw", "Distance (°)", …]
    rungs:
      - { id: r1, change: Textbook ak135 Earth, sim: 0.556, decision: kept, why: First rung., metrics: [0.21, 11.0, 110, 8.7] }
```

`decision` is `kept`, `rejected` or `released` (exactly one released rung per ladder that shipped). A model can have
several ladders (ChirpNet: black holes and neutron stars; SunDoctor: v1 and v2). The raw per-rung logs the training
scripts wrote stay in `training/`.

## Inference code

- numpy and scipy only (pandas or h5py only where the input format needs it). No torch at inference time.
- BatchNorm folded into convolutions; numpy matches the training framework to about 1e-5, and the card says so.
- Error bars already widened by the dev-fitted factors, with a way to get the raw ones.
- Loads its weights relative to its own file, so it runs from any working directory.

## Examples and the demo

- A handful of real **exam** examples (never dev), drawn by a fixed seed within strata — not picked by result.
  BearingNet's demo gets 5 of 14 right, and prints that.
- `demo.py` prints the model's reading next to the truth, and finishes in seconds (ChirpNet: about a minute).
- `tests/golden/<id>.txt` holds the demo's output; CI checks every demo still reproduces it.
