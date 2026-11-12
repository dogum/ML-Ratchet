# Live instruments

Every model gets a **live instrument** on its page: the real network, running in the visitor's browser, on real exam
examples it never trained on — with controls that change the input so you can watch the reading (and its error bars)
respond. Nothing on screen is precomputed except the reading pipeline's preprocessing; the network itself runs live.

QuakeNet is the reference implementation. Read these first, in order:

1. `site/src/instruments/types.ts` — the contract.
2. `site/src/lib/nn/ops.ts`, `pack.ts`, `tensor.ts` — the numpy-equivalent runtime (conv1d, conv2d, linear, GELU with
   the same A&S erf as the Python modules, pooling, concat) and the tensor-pack loader.
3. `mlratchet/webexport.py` — `write_pack`, `roundtrip`, `write_json`, `instrument_dir`.
4. `models/quakenet/web/export_instrument.py` — the exporter.
5. `site/src/instruments/quakenet/net.ts`, `parity.ts`, `ui.ts` — the port, its parity hook, its UI.
6. `site/src/instruments/kit.ts` and `site/src/styles/instrument.css` — the shared look: shell, picker, slider,
   button, scheduler, gauges, probs, compass, canvas, gridLines, trace, label, heatmap, rng, SCOPE colours.
7. `site/tests/parity.test.ts` — the parity test every instrument must pass.

## Files for a model `<id>`

| File | What |
|---|---|
| `models/<id>/web/export_instrument.py` | Runs the model's own Python code on its real `examples/`, writes `site/public/instruments/<id>/`: `weights.{json,bin}` (float16 via `write_pack(..., dtype="f16")`), `examples.{json,bin}` (network inputs as the pipeline produces them, plus anything the UI draws; truth and labels in `meta`), and `reference.json` (`{cases: [{example, variant, ..., outputs: [raw network outputs]}], tolerance}`). Compute reference outputs with weights passed through `roundtrip(W, "f16")` so the comparison is exact. |
| `site/src/instruments/<id>/net.ts` | Pure port of the Python `forward()` (and the post-processing to readable outputs). No DOM. Import with `.ts` extensions (`../../lib/nn/ops.ts`) — Node runs the parity test with type stripping. |
| `site/src/instruments/<id>/parity.ts` | `export async function run(case, load)` → the raw outputs for one reference case. `load(name)` returns a parsed pack. |
| `site/src/instruments/<id>/ui.ts` | `export async function mount(ctx)`: fetch the packs (`fetchPack(\`${ctx.base}weights\`)`), build the instrument with the kit, run the network on every change (wrap in `scheduler`), and call `ui.stamp(ms, note)` after each run. Support `ctx.compact` (fewer controls). |

The registry (`index.ts`) discovers `ui.ts` automatically. A model page shows its instrument when
`site/public/instruments/<id>/reference.json` exists.

## Rules

- **Parity:** `cd site && npm test` must pass for your model (tolerance ≤ 2e-3 relative; QuakeNet gets 1.4e-6). Include
  at least one perturbed variant (a deterministic change like a cut or a scale) in `reference.json`.
- **Honest controls:** a control changes the *input*, then the network runs. Use controls that teach something about
  the physics or the model's limits (noise, record length, observing length, a time slide between detectors, a
  partial discharge, sensor on/off…). If an operation needs preprocessing you cannot reproduce in JS, precompute a
  small set of variants in Python and let a slider snap between them — the network still runs live.
- **Real data only** from the model's `examples/` (or other exam records drawn by a fixed seed). Show the truth next
  to every reading.
- **Budget:** ≤ ~5 MB per instrument in total, ≤ ~1 s per run on a laptop (debounce with `scheduler`).
- **Look:** use the kit and `instrument.css` (`.inst-*` classes). Draw on the dark scope (`SCOPE` colours), the model's
  accent (`ctx.accent`) for the signal. Mono for numbers. Don't add dependencies.
- Don't edit shared files (`kit.ts`, `ops.ts`, `pack.ts`, `tensor.ts`, `types.ts`, `index.ts`, `instrument.css`,
  pages, components). Put model-specific helpers (extra ops, drawing) in your own folder.
