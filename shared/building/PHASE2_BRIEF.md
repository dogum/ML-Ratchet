# BuildingSim Phase 2 brief: four ratchet projects on BuildingSim + BuildingData

> Kept as written on 16-17 Nov 2026 in the cloud project where the buildings work was built, with the owner's name made neutral and the sandbox paths rewritten to repository paths (the sandbox's `bldg/` is `shared/building/`; the model folders are `models/loadcast`, `models/buildingdetective`, `models/faultfinder` and `models/thermopilot`).

## Read first
- `docs/agent-brief.md` (the sandbox's AGENT_BRIEF.md): recipe, deliverables, shared-machine rules. 2 cores and 7 GB RAM are shared by **four** agents, so be frugal: small networks, modest banks, one thread, nice.
- `shared/building/BUILDING_BRIEF.md`: the shared contract.
- Phase 1 docs:
  - `shared/building/buildingsim/README.md` (section "Using it in Phase 2") and `LIMITS.md`
  - `shared/building/data/DATA.md`
  - check JSONs in `shared/building/checks/`

## Calling Phase 1
```python
import sys; sys.path.insert(0,'shared/building/buildingsim'); sys.path.insert(0,'shared/building')
import buildingsim as bs, buildingdata as bd
from buildingsim.generator import labels
specs = bs.generate_buildings(N, seed=..., primary_use=None, climate_zone=None)
w = bd.site_weather('Hog'); m = bd.site_weather_meta('Hog')
out = bs.simulate(spec, (w, dict(lat=m['lat'], lon=m['lon'], elev=m['elev'], tz=m['tz'])), '2016-01-01', '2017-01-01',
                  dt='1h', faults=[...], prices=bd.tariff_prices(...), controller=policy)
out['ts'], out['trend'], out['summary'], out['ledger']['ok'], labels(spec, out)
sim = bs.Simulator(spec, wx, start, end, dt='15min', prices=tou); obs = sim.reset(); obs, info = sim.step(action); sim.finish()
```
- Speed: ~0.2 s per building-year at 1 h, ~0.7 s at 15 min.
- BDG2 twins: `checks/check_bdg2.py:spec_from_bdg2` infers the system from which meters exist.

Loaders:
- **BDG2:** `bd.bdg2(meter, building=, split=)`, `bd.bdg2_meta()`, `bd.bdg2_spec(id)`, `bd.site_weather(site)`, `bd.tmy(site)`
- **Modelled stocks:** `bd.resstock()`, `bd.comstock()`, `bd.resstock_hourly(ids)`
- **Surveys:** `bd.recs(split=)`, `bd.cbecs(split=)`
- **Faults:** `bd.faults(split=)`, `bd.fault_data(dataset, run_id)`
- **Other homes:** `bd.other('hue'|'wpuq')`
- **Tariffs:** `bd.tariffs()`, `bd.tariff_prices(...)`, `bd.bill(key, kW)`
- **Splits and baselines:** `bd.splits()`, `bd.baselines()`

## Frozen splits
`shared/building/data/proc/splits.json`, seed 20261116. NEVER change it.

| Data | Prior / input | Dev | Exam |
|---|---|---|---|
| BDG2 (by building, stratified) | — | 52 | 1,584 |
| RECS (respondents) | 9,247 | 1,848 | 7,401 |
| CBECS (respondents) | 3,217 | 661 | 2,558 |
| Fault runs (grouped by facility and date) | — | 20 | 217 |

- BDG2 time split: history = local 2016, forecast = local 2017.
- Simulated training may use real metadata only from BDG2 dev buildings and RECS/CBECS prior + dev respondents.
- Real weather and tariffs may be used freely as simulator inputs.
- Only dev may be used for calibration, gates and any fitting; exam is for the final report only.

## Phase 1 caveats to respect
- **Real BDG2 dev twins built from metadata alone:**
  - campus chilled water, steam and hot water come out ~3× too low (median ratio 0.15–0.29)
  - electricity totals scatter widely (median 0.98, IQR 0.50–1.42), with weak daily correlation (0.34)

  Dev calibration of magnitudes, operating hours and campus behaviour is the first job for anything graded on BDG2.
- **Generator intensities by use type** (site energy vs real CBECS/RECS medians):
  - schools 1.8×, healthcare 1.9×, labs 2.0×, food 1.5×, single-family 1.4× (too high)
  - lodging 0.66× (too low)

  Recentre on dev/prior respondents if your model depends on it.
- **Simulator model limits:**
  - single zone only
  - first-order fault and part-load models
  - ideal thermostat within a step
  - 5R1C steady losses are 6.9 % below the naive Σ U·A, so use `labels()['UA_network_W_K']` as the UA label
- **Trend-log channel names differ.** BuildingSim emits OAT/RAT/MAT/SAT/…; BuildingData uses `T_oa`/`T_ra`/`T_ma`/`T_sa`/…. Map between them in your own code and document the mapping.
- **Data quirks** (DATA.md):
  - mixed BDG2 timezones (fixed to UTC)
  - Gator electricity is daily totals spread hourly (excluded from hourly labels)
  - Eagle thermal meters were /1000
  - Peacock's weather mismatch
  - NASA POWER runs up to 2.5 °C cold, so use station temperature for loads
  - BDG2 is CC BY-SA 4.0: derived data must stay share-alike, so note it in model cards
- **EnergyPlus is not available here.** Checks lean on BESTEST ranges and ResStock/ComStock twins.

## Shared code rules
- Do NOT edit `shared/building/buildingsim/`, `shared/building/buildingdata/` or `shared/building/data/`. Physics changes and calibrations go in YOUR folder as wrappers, versioned in your ratchet, for the coordinator to upstream later.
- Your folder: `models/<name>/`. Keep a `STATUS.md` there. Put screenshots and temp files in a subfolder of your own folder; never use generic names in the shared scratchpad.
- Disk: ≤ 1.5 GB each (~13 GB free in total).

## Deliverables (each agent)
- Release zip ``<name>-v1.0.zip` in the cloud project's outputs folder`. Prefer a standalone numpy model; vendor small feature code.
- A showpiece Artifact page titled with your project name. Load the artifact-design skill first.
- Project doc `claude/<name>-status.md`.
- A compact final report, including upstream candidates, rule notes (anything you're unsure was allowed) and the best next rungs.
