# FirePilot offline lab

Open `index.html` in a browser. It is the page published with the release (19 Nov 2026), unwrapped into a plain HTML
file. It holds the released value network, six exam fires and the exam, fidelity and ratchet tables.

- **Explorer.** Pick a fire, a budget (300 or 1000 m of line per working hour) and a plan: FirePilot, anchor and flank,
  indirect 1 km and 2 km, direct attack, r0, MPC or the perfect-foresight bound. The map shows the stored 72 h outcome
  in FireSim's truth world, the line (built in time, unfinished, overrun) and the observed FEDS perimeter about 72 h
  after detection. These outcomes are precomputed: they need FireSim.
- **Lab.** The value network plans live in the browser (`web/fp.js`, a port of the features and the network): re-plan,
  change the budget, or draw your own plan and see its predicted share of the growth stopped. The browser planner
  searches fewer candidates than Python (45 fixed plans, 60 random ones and one refinement round of 24, with its own
  seeded random numbers; Python: 205 and two rounds of 48), so its pick can differ from the exam plan.

The six fires are a seeded draw (seed 20261122), two per detected-area tercile: Rucker (Arizona 2017), Barren Ridge
(Idaho 2017), Owl (New Mexico 2018), Telephone (Arizona 2019), Gold (California 2020) and Wolf Creek (Oklahoma 2023).
They are not the three fires in `examples/`. The page stores its inputs quantized (arrival time to 1/16 h, flame
length to 0.05 m); `web/check_js.js` compared the page's JavaScript with Python on them: predictions within 8.5e-7.

## How this file was made

`web/build_page.py` builds the page from the exam's stored per-fire inputs and masks (`results/exam/`), which were not in
the release zip, so it cannot be re-run here. The published page was `web/template.html` with `web/fp.js` and one
`const DATA = {...}` line filled in, inside the page host's wrapper; on import this was checked to the byte.
`web/make_lab.py` writes this file: the same fill of the template, with the page's DATA line, in a plain
`<!doctype html>` document without the host wrapper. The body is unchanged. Fonts load from Google Fonts, with system
fallbacks offline. To rebuild after editing `template.html` or `fp.js`: `python web/make_lab.py` (it reads the DATA line
from the current `index.html`).

The site instrument (`site/src/instruments/firepilot/`) is a separate TypeScript port of the same features and network,
checked against Python on the three `examples/` fires; `web/export_instrument.py` writes its data.

Data credits: FEDS-MTBS (CC BY 4.0), MTBS, LANDFIRE and HRRR (US public data); see `examples/README.md`.
