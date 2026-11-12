# Notes for AI agents working in this repository

ML Ratchet is a collection of physics-first ML models. Read `docs/method.md` and `docs/model-contract.md` first.
To build a new model, follow `docs/agent-brief.md` exactly.

## Single sources of truth

- `models/<id>/model.yaml` and `ratchet.yaml` drive the model READMEs, the CLI and the website.
  After editing them run `mlratchet readme --write`. Never hand-edit `models/*/README.md` or `docs/backlog.md`.
- `docs/backlog.yaml` drives `docs/backlog.md` and the site's backlog page.
- `models/*/lab/index.html` are the original self-contained pages (large, base64-embedded); don't reformat them.
- `models/*/NOTEBOOK.md` and `docs/history/` are historical records; add dated entries, don't rewrite history.

## Live instruments

`site/src/instruments/README.md` is the contract for in-browser instruments: `models/<id>/web/export_instrument.py`
writes `site/public/instruments/<id>/`; `site/src/instruments/<id>/{net,parity,ui}.ts` port and present it; `npm test`
in `site/` checks the browser network against Python.

## Checks before you finish

```bash
mlratchet validate
mlratchet readme --check
pytest -q                      # demos are compared to tests/golden/
cd site && npm run check && npm test && npm run build
```

## Rules

- No real labelled example is ever a training sample; the exam never chooses anything; log every rung.
- Inference stays numpy/scipy-only and must not import `mlratchet` (each model folder is copyable on its own).
- Training scripts write only to `work/` via `training/_paths.py`; never commit data downloads, banks or checkpoints.
- Don't route around blocked data hosts; record them in the model's `help_wanted`.
- Plain writing: short sentences, real units and numbers, say what failed.
