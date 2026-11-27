# LawFinder v1.0 · Released expressions and their domains

**Cross-checked critical EV energy defect.** On a paired battery-electric drive the released law predicts *less* battery energy as the road gets steeper uphill. RoadSim, the simulator supplying its training labels, predicts the opposite. Model snapshot `61c70bf`, review at `73ee7e6`. [First pass](reviews/reviewer-a/lawfinder-v1.0.md) · [Bank label calculation](../../models/lawfinder/training/road_bank.py#L119)

| Grade on the same 600-second drive | Released law (MJ/km) | RoadSim DC battery energy (MJ/km) |
|---|---:|---:|
| −2% | 0.5123 | 0.0085 |
| Flat | 0.4075 | 0.3316 |
| +2% | 0.3242 | 0.6594 |

This replay used the same BEV, speed trace, starting state of charge, temperature and simulator seed. It computed rolling resistance, grade, acceleration and energy using the bank’s own units and formulas. In particular, rolling resistance is supplied in thousandths (`R=8.5` here). Reviewer A’s ±6% example used `R=0.01` and `U=0` with nonzero net grade. The example above resolves both issues. The sign also follows directly from the expression. At `e=1` and any nonnegative speed, its exponential base is below one, while the exponent rises with grade and positive kinetic energy. The archived subgroup scores establish poor EV performance. They cannot by themselves attribute the entire error to this sign. [Feature definitions and ranges](../../models/lawfinder/training/law_road.py#L26) · [Reported exam](../../models/lawfinder/training/results/exam_road_reported.json)

**The search did produce alternatives.** At 20 m/s and fixed `U=2, R=10`, predictions at net grades −2%, 0%, +2% were 0.4481, 0.3545 and 0.2805 MJ/km. Six of 21 logged p2 candidates increased over that grid (`c18, c20, c21, c22, c24, c25`). The other 15 were flat within numerical tolerance. All six missed the pooled frozen-set promotion margin. For example, `p2-c24` scored 0.764 against an incumbent of 0.767 and a required improvement above 0.01. An EV-specific evaluation would be needed to choose among candidates. All 238 logged decisions reproduced under the historical rules. [Road ratchet](../../models/lawfinder/training/results/road_ratchet.json)

**Building and grid scope.** Heating slope falls from +1.872 at year 2020 to −0.128 at 2022 and −3.128 at 2025, in annual kWh/m² per additional thousand heating degree days. The declared year range ends at 2020. For the two grid laws, the intermediate value matters. On a regular grid of demand 16–46 GW and inverter share 0.05–0.80, 615 of 2,356 upstream predictions fell below the downstream law’s published minimum inertia of 90 GVA·s. At a 1 GW loss, 252 yielded NaN. Every intermediate value inside the downstream range gave a finite result. These counts describe the test grid. The real nadir exam supplies observed inertia directly. [Building inputs](../../models/lawfinder/training/law_bldg.py) · [Grid exam](../../models/lawfinder/training/exam_grid.py)

**Weather-cache exposure.** The road bank calls RoadSim with copied trip arrays. The building bank copies a cached city weather frame, shifts temperature and dewpoint, and calls BuildingSim. In a synthetic year using the real `bldg_bank.one` entry point, raising wind from 2 to 8 m/s with the same seed returned **188.80 kWh/m²** on a warm cache and **220.24 kWh/m²** after clearing it. BuildingSim’s prepared-weather key includes temperature and omits wind. The ordinary bank randomizes temperature and reuses a city’s wind, so the paired collision does not establish corruption of historical training labels. [Building bank](../../models/lawfinder/training/bldg_bank.py#L25) · [Shared cache](../../shared/building/buildingsim/buildingsim/weather.py#L218)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B docs/audits/reviews/reviewer-a/lawfinder_v1_0_probe.py
python3 -B docs/audits/reviews/reviewer-b/peer_73ee7e6_probe.py lawfinder
python3 -B docs/audits/reviews/reviewer-b/lawfinder_bank_probe.py
```

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Road law's battery energy falls with grade for every BEV | Yes | **Documented, not refitted.** The released law is the search's output; changing it means a new search and a new exam. The card's limits now state the inversion with the audit's numbers, and `next` queues a sign-checked search (energy must not fall with net grade on the frozen set) or an EV evaluation among the six p2 candidates that already rise with grade. | `models/lawfinder/model.yaml` limits and next. |
| Homes law extrapolates past 2020; grid laws chain below the nadir law's 90 GVA·s floor | Yes | **Fixed as a guardrail**: `Law.predict` warns when any input is outside its declared bank range (`var_info[..]['range']`); `strict=True` raises; `Law.out_of_range(**kw)` counts them. Exam code paths are unchanged (a warning, not a change of value). | Test added. One of the demo's own real grid-nadir events has a loss above the 2 GW range and now warns. |
| BuildingSim's prepared-weather cache key omits wind | Yes (shared) | Fixed with the building platform (the cache lives in `shared/building/buildingsim/weather.py`). | See the building audit resolutions. |
| (found here) `LawFinder()` opened its JSON with the platform default encoding | Yes | **Fixed**: `encoding='utf-8'`. On Windows (cp1252) the `°`, `⁻` and `·` in the law texts were mis-decoded. | — |

### Follow-up (27 Nov 2026): the sign-checked search, run

The p2 banks were rebuilt from their seeds and reproduce every logged frozen-set score exactly. The sign check on val_p2
(energy may not fall as net grade rises) fails the released p1-c23 on 21 % of rows and on every BEV row. It also fails
the six grade-rising p2 candidates, which fall on 4-14 % of the non-BEV rows. A new p2 search ran with the check as a
hard constraint (1,200 s, 143,084 evaluations) and found 14 laws, none of which uses grade sensibly. The best,
p2s-c19, ignores grade and scored R² 0.757 against the incumbent's 0.767, so it was **rejected** under the unchanged rule.
The exam was not opened, and p1-c23 stays released with the documented limit. If the check were a gate (a rule change,
not adopted), p1-c17 would be released. See `models/lawfinder/NOTEBOOK.md`, 2026-11-27.

### Follow-up (27 Nov 2026): the sign gate adopted; road law v1.1

The owner adopted the sign check as a release rule for the road ladder. The rule came from this audit, not from any exam
result. Re-walking the logged ladder with no new search released **p1-c17** (R² 0.754, dev 17.5 %). p1-c17 does not
read grade. The release was committed before the exam ran once as road law v1.1. Results: VED trips 18.7 % MdAPE (v1.0
18.6 %), EPA tests 19.4 % (17.3 %), EPA battery-electric 20.0 % (43.1 %), VED Leafs 33.7 % (43.3 %). The grade
inversion is gone, and electric cars improve most, while petrol and hybrid highway tests get worse. See
`models/lawfinder/NOTEBOOK.md`, 2026-11-27 (road law v1.1).
