# Pilot: FireSim spread kernel in Rust → WebAssembly + Python

Written 22 Nov 2026 in the cloud project for a repository session; kept as written apart from this line. Not started. This is the first step toward running ML Ratchet's *simulators*, not just its networks, live in the browser.

## Goal
Port FireSim's minimum-travel-time (MTT) spread kernel to one Rust crate with two front ends:
- **Browser:** `wasm-bindgen` / `wasm-pack`, called from a site instrument. A fire spreads across a real dev landscape as the visitor drags wind speed, wind direction and fuel moisture.
- **Python:** `PyO3` / `maturin`, a drop-in alternative to the numba kernel behind a flag, e.g. `firesim.simulate(..., engine='rust')`.

## Scope (pilot only)
**In:**
- Rothermel surface spread for the 53 fuel models: the precomputed per-cell ROS / ellipse parameters can stay in Python and be passed in.
- The MTT Dijkstra solver with the 16-cell stencil plus two-point updates.
- Time-varying weather by hour.
- Non-burnable cells and timed lines. **Include the U1 fix:** a diagonal step is blocked when *either* side cell is blocked by a line (`t_block <= t`), not only when it is non-burnable. See `shared/fire/HANDOFF.md` and FirePilot's `leak_test.py`.

**Out, for later:** crown fire, spotting, the satellite emulator, moisture models.

## Guardrails
1. **The verified Python package stays the reference.** The Rust kernel must reproduce, within stated tolerances:
   - `checks/analytic.json`: ellipse area, L/W and grid-orientation bias; the two-fuel strip; the fuel break with 0 leaked cells.
   - the arrival-time rasters of the 40 dev replays: max |Δt| ≤ 1e-6 h on identical inputs, or a documented tolerance.
2. **Add a parity test to CI** (`pytest`), comparing the Python and Rust engines on a seeded set of landscapes.
3. **Add a site parity test** in the style of `site/src/instruments` (`npm test`), comparing WASM against the Python golden outputs.
4. **Determinism:** the same inputs give the same arrival times on every platform. Document any floating-point ordering differences.
5. **Size budget:** the WASM bundle stays ≤ ~300 KB gzipped. One dev landscape at 90 m (fuel, slope, aspect, canopy) is shipped as compact typed arrays.

## Done when
- `firesim.simulate(engine='rust')` passes every FireSim check that the numba engine passes.
- The site has a "Fire lab" instrument where a real dev landscape burns live in the browser at interactive speed: about 1 s for 24 h on 300 × 300 cells.
- Timing is reported for Python+numba, Python+Rust and browser WASM on the same case.

## If it works
Next candidates, in order of how visual and how hot they are:
1. WaterSim's HBV/GR4J core plus routing: "drag a storm over a basin".
2. GridSim's swing equation: "trip a plant, watch the frequency".
3. RoadSim's powertrain step.

Keep one crate per platform under `shared/<platform>/rust/`.
