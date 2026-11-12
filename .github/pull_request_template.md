## What this changes

<!-- One or two sentences. For a new rung, the change and what the ratchet said. -->

## Checklist

- [ ] `mlratchet validate` passes
- [ ] `mlratchet readme --write` run after editing any `model.yaml` / `ratchet.yaml`
- [ ] `pytest` passes (demos reproduce their golden output, or the golden output was regenerated on purpose and the PR says why)
- [ ] For a new or changed model: no real labelled example was used for training, the exam was not used to choose anything, and the card reports where it loses
- [ ] New data sources are listed in `DATA_LICENSES.md`, and only bundled if their terms allow it
