# shared/building: BuildingSim + BuildingData

The third shared platform: a building-energy simulator (BuildingSim) and real building data with frozen splits
(BuildingData), under four models: [LoadCast](../../models/loadcast/), [BuildingDetective](../../models/buildingdetective/),
[FaultFinder](../../models/faultfinder/) and [ThermoPilot](../../models/thermopilot/). Overview, checks and what the
models found: [docs/building-platform.md](../../docs/building-platform.md). Built on 16-17 Nov 2026 in a claude.ai
sandbox; Phase 1 against [BUILDING_BRIEF.md](BUILDING_BRIEF.md), Phase 2 against [PHASE2_BRIEF.md](PHASE2_BRIEF.md).

```
shared/building/
├── buildingsim/      the simulator: package buildingsim/, README.md (physics, API, numbers), LIMITS.md, demo.py,
│                     tools/ (priors, stock weather), refdata/ (TMYx + BESTEST weather, priors; git-ignored)
├── buildingdata/     the real-data loader (pandas/numpy/pyarrow)
├── data/             DATA.md (sources, cleaning, splits, baselines), build.py (the data command), scripts/,
│   │                 MANIFEST.json (sha256 of all 409 v1.0 data files)
│   └── proc/splits.json   the frozen splits, seed 20261116, sha256 field 7ac15e9f…; never edit it
├── checks/           check_*.py, the shipped v1.0 JSONs, run_all.sh, compare.py, REPRODUCED.md
├── HANDOFF.md        what exists and the prioritised work list    STATUS.md   status on 17 Nov 2026
└── BUILDING_BRIEF.md, PHASE2_BRIEF.md   the contracts, kept as written
```

## Get the data

Not in git. One root holds `data/`, `buildingsim/refdata/` and `checks/out/`: `$ML_RATCHET_BUILDING`, default this
folder.

```bash
pip install -e ".[building]"
mlratchet data building --from-zip buildingsim-buildingdata-v1.0.zip   # unpack the v1.0 release (230 MB), verify
mlratchet data building --only verify                                   # sha256 of all 409 files, splits, loader
mlratchet data building --only splits                                   # re-run make_splits.py in data/work/
mlratchet data building --list                                          # the rebuild stages and their hosts
```

The release (sha256 c918802d…; also accepted as `buildingsim-minusdata.zip` plus the `proc/<folder>.zip` chunks)
holds `data/proc/` (bdg2 130 MB, other 49 MB, eia, faults, tmy, tariffs, baselines; 271 MB in all),
`buildingsim/refdata/` and the per-unit twin results in `checks/out/`. BDG2 and everything derived from it are
CC BY-SA 4.0. Never shipped, rebuilt by the scripts: `data/proc/stock/` (ResStock/ComStock, ~1.2 GB), the raw
downloads (so `bd.site_asos` has no files) and the CoolProp/pythermalcomfort venv for `check_components.py`.

## Use it

```python
import sys; sys.path[:0] = ['shared/building/buildingsim', 'shared/building']   # what the models' _paths.py do
import buildingsim as bs, buildingdata as bd
w = bs.read_epw(bs.epw_path('Chicago'))             # (weather frame, meta); 17 bundled TMYx cities
out = bs.simulate(dict(primary_use='office', floor_area_m2=5000, climate_zone='5A', system='vav_chiller_boiler'),
                  w, '2019-01-01', '2020-01-01', dt='1h')
e = bd.bdg2('electricity', split='dev')          # 17,552 h x 52 dev buildings, UTC, bad hours NaN
bd.site_weather('Hog'); bd.recs(split='dev'); bd.faults(split='exam'); bd.other('hue'); bd.tariff_prices('pge_e_tou_c', 2017)
```

Checks: `checks/run_all.sh` re-runs everything (about 30 min; the twin checks need the stock samples, components a
CoolProp venv); outputs go to `checks/out/`, never over the shipped JSONs. `python checks/compare.py` compares.
[checks/REPRODUCED.md](checks/REPRODUCED.md) records the first re-run here: every check that could run, including
the real-BDG2 twins, reproduced, and `make_splits.py` reproduces the frozen splits. `pytest -q tests/test_building.py` runs
the fast tests (data tests skip when the data are absent).
