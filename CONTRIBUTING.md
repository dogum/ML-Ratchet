# Contributing

Thanks for wanting to help. The collection grows one honest model at a time.

## Ways to help

- **Run a demo and report what you see** — especially on another OS or numpy version.
- **Close a blocked exam.** ChirpNet's O3 exam and CellSense's other chemistries are waiting on data
  ([roadmap](docs/roadmap.md)).
- **Climb a rung.** Every model lists its next rungs in `model.yaml` under `next`. A rung is a contribution even if
  the ratchet rejects it — log it.
- **Build a new model.** Pick one from [the backlog](docs/backlog.md) or propose one with the *New model idea*
  issue template, then follow [adding a model](docs/adding-a-model.md).
- **Improve the site or the labs** (`site/`, `models/*/lab/`).

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
mlratchet validate          # every model honours the contract
mlratchet readme --check    # READMEs are in sync with their YAML
pytest                      # core tests + every demo against its golden output (ChirpNet takes ~1 min; -m "not slow" skips it)

cd site && npm install && npm run dev                   # the website at http://localhost:4321/ML-Ratchet/
```

## The rules that make it worth anything

1. No real labelled example is ever a training sample.
2. The exam never chooses anything. If you looked at the exam to pick a rung, that rung's exam is no longer an exam —
   say so in the card.
3. Every rung is logged in `ratchet.yaml`, kept or not, with the reason.
4. Every exam has a classical method and a dumb baseline beside it, and the card says where the model loses.
5. Data is downloaded only from documented public endpoints and credited; it is only bundled if its terms allow.

## Pull requests

- Edit `model.yaml` / `ratchet.yaml`, then run `mlratchet readme --write` (READMEs are generated).
- If a demo's output changes on purpose, regenerate its golden file (`python tests/test_demos.py --regen`) and say why.
- Keep inference numpy-only and the demo fast.
- Plain writing: short sentences, real units and numbers, no hype, say what failed.

By contributing you agree your code is released under the MIT licence and you follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
