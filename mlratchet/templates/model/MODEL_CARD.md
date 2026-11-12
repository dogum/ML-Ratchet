# __NAME__ model card (v0.1)

**What it does.** TODO: one paragraph, plain words.

**How it learned.** Only from simulations. No real labelled example is in its training data.

## Inputs

- TODO: name, shape, units, preprocessing, valid range.

## Outputs

| key | meaning |
|---|---|
| TODO | TODO, with its 68% error bar |

Error bars are widened by one factor per output, fitted on the dev set only. The rule only ever widens.

## Physics and simulation

- **Governing physics:** TODO
- **Physics check:** TODO (number + file)
- **Noise:** TODO (which real recordings)
- **Corrections measured on dev:** TODO (each one became a rung)

## Network

TODO: architecture, parameters, loss, training steps.

## Ratchet

A rung is kept only if it beats the incumbent by more than 0.01 on the frozen simulated set AND passes the dev gate.
Every rung is in `ratchet.yaml`, kept or not.

## Final exam

TODO: table of the model vs a classical method vs a dumb baseline, on real data never used to choose anything.
Coverage of the 68% bars. Where it fails (fixed random draw within strata, never cherry-picked).

## Limits

- TODO

## Data credits and licences

- TODO: every dataset, its URL, its licence, and the citation the provider asks for.
