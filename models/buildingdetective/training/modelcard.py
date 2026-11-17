"""Write release/MODEL_CARD.md from exam.json, ratchet.json, calibration and the twin test (numbers never typed by hand).
usage: python3 modelcard.py"""
import json
import os

import common as C
from buildpage import summary

T = '''# BuildingDetective v1.0: model card

## What it does
Reads a building from its meters and the weather. Outputs, each with a one-sigma error bar or class probabilities:
floor area, year built, envelope conductance UA (W/K) and UA per floor area, thermal time constant (h), heating and
cooling balance points (degC), site heating / cooling energy per kelvin (W/K), non-HVAC share of site energy, weekday
operating hours and weekend/weekday ratio (commercial only), primary use (12 classes), heating system (heat pump,
electric resistance, on-site fuel, district, none) and cooling system (none, DX, chiller, district).

Three networks for three input resolutions: hourly meters (one year, at least ~30 days), 12 monthly bills, or one
annual total per fuel. Meter channels: electricity, on-site fuel, district heat (steam / hot water), district chilled
water. Weather: hourly outdoor temperature and global horizontal irradiance (monthly / annual: temperature and degree
days only).

## How it was made (the ML Ratchet recipe)
1. **Physics.** BuildingSim (Phase 1 of this project): single-zone 5R1C heat balance with solar, infiltration,
   ventilation and moisture; HVAC equipment with capacity / COP / EIR curves, part load, defrost, fans, pumps, plants;
   verified by the Phase 1 checks (analytic RC step and steady conduction, BESTEST ranges, ledger closure, ResStock and
   ComStock twins; `bldg/checks/*.json`). Labels come from the simulator itself: UA = the 5R1C network conductance
   (`labels()['UA_network_W_K']`), tau, balance points, end-use shares, schedules, system classes.
2. **Simulated data in real conditions.** {n_train:,} generated buildings (30 % homes, 70 % commercial in 10 types,
   drawn from ResStock / ComStock / CBECS priors) on real weather: the 18 BDG2 sites in 2016 and 2017 and TMYx cities.
   Each meter is then given real-world faults (gaps, zero runs, spikes, level shifts, clock offsets, daily-spread
   readings, shared unmodelled loads, sub-meter scale errors, noise; billing windows and missing months for bills;
   unreported fuels for surveys).
3. **Calibrated on dev only** (wrappers around BuildingSim, upstream candidates):
   - W1 intensity: one factor per use type so generated site EUI medians match RECS / CBECS prior+dev medians.
   - W2 campus: district meters scaled by a distribution factor fitted on the 52 BDG2 dev twins (heat {eff_h:.2f},
     chilled water {eff_c:.2f}; meter = delivered / factor), 30 % of commercial buildings on campus service.
   - W3 always-on base loads (physics v2) from the dev twin test (table below): a share of each district meter's annual
     energy moved into a flat base (heat 0.70, chilled water 0.50), plus an unmodelled flat electricity load.
4. **Network.** A 2-layer MLP (256 GELU units, ~0.37M parameters for hourly) on ~1,200 engineered features per
   building: daily load in temperature bins (weekday / weekend), the hour-of-week profile, cold- and hot-day profiles,
   a PRISM 5-parameter change-point fit, correlations with exponentially smoothed temperature (6 time constants, the
   thermal-mass signal) and hourly lag correlations, plus robust statistics. Heads: Gaussian mean and log-sigma for 12
   regression targets (NaN labels masked) and three softmax classifiers. Trained only on simulations.
5. **Ratchet.** Frozen validation set = bank shards 900-904 (1,000 buildings) of the current physics version. A rung is
   kept if its skill (mean of 12 R2 clipped at 0 and 3 chance-rescaled balanced accuracies) beats the incumbent by
   more than 0.01 AND its real dev composite does not drop more than 0.02. Physics changes re-score all incumbents.
6. **Error bars.** One widening factor per output (widen-only), fitted on dev: real BDG2 / RECS / CBECS dev units for
   floor area and year built, the simulated twins of the 52 BDG2 dev buildings for everything without real truth.
   Class temperatures (>= 1) fitted on dev labels.

## Ratchet
| rung | change | frozen-val skill | dev gate | decision |
|---|---|---|---|---|
{rungs}

## Twin test (dev): real BDG2 dev buildings vs their simulated twins
| meter, feature | real | twin v1 | twin v2 (W3) |
|---|---|---|---|
{twin}

## Final exam (run once after all choices were frozen)
"real" = measured buildings or survey answers. "simulated (exact)" = held-out BuildingSim buildings. "sim vs sim" =
EnergyPlus meters (NREL ResStock / ComStock) graded against a BuildingSim twin: two models, not reality. Bold = beats
every baseline listed. Baselines marked TRUE are given the true use type, which the Detective is not.

{exam}

{coverage}

## Error-bar widening factors (dev)
{widen}

## Where it fails
{limits}

## Data and licences
- Building Data Genome 2: Miller et al., Sci. Data 7, 368 (2020). **CC BY-SA 4.0**: anything derived from BDG2 (the
  example series in `examples/`, BDG2-calibrated wrappers) stays share-alike.
- EIA RECS 2020 and CBECS 2018 public microdata (public domain).
- HUE: Makonin, Data in Brief 23 (2019), Harvard Dataverse doi:10.7910/DVN/N3HGRN (CC BY 4.0).
- NREL ResStock 2025 R1 / ComStock 2025 R3 (CC BY 4.0). "Data includes information from the ResStock and ComStock
  datasets developed by the National Renewable Energy Laboratory (NREL; now National Laboratory of the Rockies) with
  funding from the U.S. DOE." Used for generator priors (Phase 1) and the simulator-vs-simulator check only.
- TMYx 2011-2025 (climate.onebuilding.org); NOAA ISD weather inside BDG2.
- Code: MIT, ML Ratchet (github.com/dogum/ML-Ratchet).
'''


def md_exam(ex):
    out = []
    for g in ex:
        out.append(f"### {g['group']}\n| task | input | truth | n | Detective | baselines |\n|---|---|---|---|---|---|")
        for r in g['rows']:
            b = '; '.join(f"{x['name']}: {x['v']}" for x in r['base'])
            out.append(f"| {r['task']} | {r.get('input', '')} | {dict(real='real', sim='simulated (exact)', simsim='sim vs sim').get(r['truth'], 'sim vs sim')} | "
                       f"{r['n'] if r['n'] is not None else ''} | {'**' + r['det'] + '**' if r['win'] else r['det']} | {b} |")
        out.append('')
    return '\n'.join(out)


def main():
    S = summary()
    cal = json.load(open(os.path.join(C.ROOT, 'models', 'calibration.json')))
    fc = json.load(open(os.path.join(C.ROOT, 'models', 'final', 'calibration.json')))
    rungs = '\n'.join(f"| {r['name']} | {r['desc']} | {r['val']} | {r['gate']} | {r['decision']} |" for r in S['rungs'])
    twin = '\n'.join(f"| {r['f']} | {r['real']} | {r['v1']} | {r['v2']} |" for r in S['twin'])
    widen = '| mode | ' + ' | '.join(C.REG) + ' | class temperature (use, heat, cool) |\n|---' * 1 + '|---' * (len(C.REG) + 1) + '|\n'
    for m, v in fc.items():
        widen += f"| {m} | " + ' | '.join(f"{v['widen'][k]:.2f}" for k in C.REG) + f" | {', '.join(f'{t:.2f}' for t in v['temperature'])} |\n"
    txt = T.format(n_train=S['n_train_buildings'], eff_h=cal['campus']['eff_heat_median'], eff_c=cal['campus']['eff_chw_median'],
                   rungs=rungs, twin=twin, exam=md_exam(S['exam']), coverage=S['coverage_note'], widen=widen,
                   limits='\n'.join('- ' + x for x in S['limits']))
    open(os.path.join(C.ROOT, 'release', 'MODEL_CARD.md'), 'w').write(txt)
    print('MODEL_CARD.md', len(txt))


if __name__ == '__main__':
    main()
