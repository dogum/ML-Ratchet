"""Write model.yaml, ratchet.yaml and MODEL_CARD.md for the release from the results JSONs."""
import json, os, importlib
import numpy as np
import yaml
import lawratchet as lr
import lawsearch as ls
import pretty

ROOT = lr.ROOT
REL = f'{ROOT}/release'
LAWS = ['water', 'fire', 'grid_inertia', 'grid_nadir', 'bldg']
PLAT = {'water': 'WaterSim', 'fire': 'FireSim', 'grid_inertia': 'GridSim', 'grid_nadir': 'GridSim', 'bldg': 'BuildingSim'}
KEY = {'water': ('mae', 'mean |error| of Q/P', '', 'lower'), 'fire': ('mae_log10', 'mean |log10 ratio| of new area', '', 'lower'),
       'grid_inertia': ('mdape', 'MdAPE of inertia', '%', 'lower'), 'grid_nadir': ('mae', 'MAE of the nadir depth', 'Hz', 'lower'),
       'bldg': ('mdape', 'MdAPE of annual site energy per m2', '%', 'lower')}
KIND = {'fu_dev': 'classical', 'fu_2.6': 'classical', 'fu_sim': 'classical', 'flowcast': 'published', 'dumb': 'dumb',
        'rothermel_dev': 'classical', 'rothermel_sim': 'classical', 'fl_prev': 'classical', 'firecast': 'published',
        'clim': 'dumb', 'persist': 'dumb', 'k_x_demand': 'classical', 'gridpulse': 'published', 'swing_ramp': 'classical',
        'gridpulse_5s': 'published', 'signature_dev': 'classical', 'signature_sim': 'classical', 'loglin_dev': 'classical',
        'sim_law': 'classical'}


def exam(name):
    if name.startswith('grid_'):
        p = f'{ROOT}/results/exam_grid.json'
        return json.load(open(p))[name.split('_')[1]] if os.path.exists(p) else None
    p = f'{ROOT}/results/exam_{name}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def r3(v):
    return None if v is None or not np.isfinite(v) else float(f'{v:.3g}')


def collect():
    out = {}
    for n in LAWS:
        p = f'{ROOT}/results/{n}_ratchet.json'
        E = exam(n)
        if not os.path.exists(p) or E is None:
            continue
        L = importlib.import_module(f'law_{n}')
        R = json.load(open(p))
        rel = next(r for r in R['rungs'] if r['id'] == R['released'])
        kn = lr.knee(R)
        txt = lambda r: pretty.text(ls.from_json(r['tree']), np.array(r['consts']), L.VARS)
        out[n] = dict(L=L, R=R, E=E, rel=rel, kn=kn, rel_text=txt(rel), kn_text=txt(kn), txt=txt)
    return out


def verdict(E, k):
    law = E['rows']['law'][k]
    riv = [v[k] for kk, v in E['rows'].items() if kk not in ('law', 'sim_law') and v.get('n', 0) >= 3 and v.get(k) is not None
           and np.isfinite(v[k])]
    best = min(riv)
    return 'win' if law < best * 0.98 else ('tie' if law <= best * 1.02 else 'loss')


def model_yaml(C):
    results, coverage = [], []
    for n, c in C.items():
        k, desc, unit, better = KEY[n]
        E = c['E']
        bl = [{'name': v['name'], 'kind': KIND.get(kk, 'classical'), 'value': r3(v[k])} for kk, v in E['rows'].items()
              if kk != 'law' and v.get('n', 0) >= 3 and v.get(k) is not None and np.isfinite(v[k])]
        results.append({'metric': f"{c['L'].TITLE}: {desc} ({E['rows']['law']['n']} real exam units)", 'better': better,
                        'unit': unit, 'model': r3(E['rows']['law'][k]), 'baselines': bl, 'verdict': verdict(E, k),
                        'note': f"Released law {c['rel']['id']}: {c['L'].TARGET['symbol']} = {c['rel_text']}. "
                                f"{PLAT[n]}'s own law ({c['kn']['id']}): {c['kn_text']}."})
        if 'cov68' in E['rows']['law']:
            coverage.append({'output': c['L'].TITLE, 'actual': round(100 * E['rows']['law']['cov68'], 1)})
    y = {
        'id': 'lawfinder', 'name': 'LawFinder', 'version': '1.0', 'status': 'released', 'released': '2026-11-22',
        'order': 31, 'domain': 'Method', 'tags': ['symbolic regression', 'equations', 'Pareto front', 'textbook laws',
                                                  'water balance', 'fire growth', 'grid inertia', 'energy signature'],
        'accent': '#1d6fb8',
        'tagline': 'What does each simulator believe, in one line, and is it true?',
        'question': 'What readable law does each shared simulator imply, and does that law hold on the platform\'s real exam?',
        'question_short': 'One-line laws extracted from the simulators, graded on real exams',
        'summary': ('LawFinder runs symbolic regression on simulated banks from WaterSim, FireSim, GridSim and BuildingSim '
                    'and returns short equations with error bars: the long-term runoff ratio of a basin, the next-day '
                    'growth of a wildfire, GB system inertia, the GB frequency nadir after a trip, and the annual energy '
                    'intensity of a single-family home. Each law is a rung on a ratchet of complexity levels and physics '
                    'versions, chosen on a frozen simulated set and a small real dev set, then examined once on the '
                    'platform\'s real exam beside the textbook law (its constant fitted on the same dev set), the '
                    'collection\'s network and a dumb baseline.'),
        'reads': 'Per law: water - aridity PET/P, snow fraction, P-PET seasonal synchrony, soil storage index; fire - '
                 'perimeter (km), daily 10 m wind (m/s), 100-h dead fuel moisture (%), fuel group; grid inertia - demand '
                 '(GW), inverter-based share; grid nadir - MW lost, inertia (GVA s), demand; homes - HDD and CDD (base '
                 '18 C), floor area, year built.',
        'returns': [f"{c['L'].TITLE}: {c['L'].TARGET['label']} with a 68 % bar (frozen-set residual sd, dev-widened)"
                    for c in C.values()],
        'platform': 'method',
        'physics': {
            'summary': 'No new physics: the collection\'s shared simulators are used unchanged (WaterSim v1, FireSim v1, '
                       'GridSim 1.1 via GridPulse\'s GB state sampler, BuildingSim 1.0/1.4). Each law is a statement '
                       'about what one simulator implies.',
            'simulator': ['WaterSim: 16,000 generated basins (two physics versions), 4 runoff models, 2 snow routines, real '
                          'ERA5-Land forcing from free windows, 10-year runs; w2 adds FlowCast\'s dev-fitted ERA5 water-yield multiplier',
                          'FireSim: 1,680 multi-day fires on guarded real dev LANDFIRE windows or parametric landscapes, real dev '
                          'HRRR weather; f2 uses the dev-calibrated ROS factors',
                          'GridSim: generate_states GB fleets (inertia law); GridPulse\'s GB sampler p0/p1 for trips, measured '
                          'like GridData\'s catalogue (1 s means, 10 s fall <= -0.15 Hz)',
                          'BuildingSim: 2,600 single-family homes from the ResStock-bootstrapped generator, TMYx weather '
                          'shifted by U(-3, 3) C; b2 = physics 1.4'],
            'checks': ['lawsearch recovers a planted Budyko-Fu law with 1 % noise to the noise floor (MSE 1.0e-4 vs 1.0e-4)',
                       'Every simulator\'s own Phase 1 checks apply (shared/*/checks)',
                       'Release evaluator (numpy) equals the search engine to machine precision; the page\'s JS equals numpy (training/results/js_parity.json)']},
        'noise': 'Platform noise models as shipped: WaterSim forcing error on the inputs the law sees; FireSim runs on real '
                 'HRRR; GridSim events measured as 1 s means; BuildingSim weather shifts. No real labels in training.',
        'network': 'No network: symbolic regression (lawsearch, numpy/scipy; PySR-style regularised evolution with '
                   'Levenberg-Marquardt constants and a hall of fame per complexity). Operators + - * / ^ exp log sqrt inv sq; '
                   'complexity = node count, up to 25.',
        'approach': {'real_backdrop': 'Real ERA5-Land forcing windows, real dev LANDFIRE/HRRR, NESO train-year operating points, TMYx weather',
                     'physics_inputs': 'Every law input is a physical quantity (aridity, perimeter, inertia, degree days)',
                     'physics_anchor': False, 'real_data_roles': ['calibration', 'gate', 'error_bars', 'exam']},
        'data': [
            {'name': 'Caravan / CAMELS-US (ERA5-Land forcing, streamflow)', 'url': 'https://zenodo.org/records/7944025', 'licence': 'CC BY 4.0', 'role': 'dev, exam, physics (free forcing windows)'},
            {'name': 'USGS NWIS daily values', 'url': 'https://waterdata.usgs.gov', 'licence': 'Public domain', 'role': 'dev, exam'},
            {'name': 'FEDS-MTBS fire progression; MTBS; LANDFIRE; HRRR; gridMET', 'url': 'https://zenodo.org/records/20187963', 'licence': 'CC BY 4.0 / US public data', 'role': 'dev, exam, physics'},
            {'name': 'NESO 1 s frequency, outturn inertia, generation mix; Elexon REMIT/B1610 labels via GridData', 'url': 'https://www.neso.energy/data-portal', 'licence': 'NESO Open Data Licence; Elexon BMRS licence', 'role': 'dev, exam, physics (train-year operating points)'},
            {'name': 'EIA RECS 2020', 'url': 'https://www.eia.gov/consumption/residential/', 'licence': 'Public domain', 'role': 'dev, exam'},
            {'name': 'TMYx weather (climate.onebuilding.org)', 'url': 'https://climate.onebuilding.org', 'licence': 'Free with attribution', 'role': 'physics'}],
        'split': {'dev': 'Each platform\'s frozen dev split, copied in splits.json: 52 CAMELS basins; 40 fires (791 fire-days); 4 GB events of 2019; RECS dev single-family homes',
                  'exam': 'Each platform\'s exam split: 619 CAMELS basins (WY1990-99); FireData exam fire-days 2012-2024; 410 GB events (inertia) and the loss-labelled ones (nadir), 2018-2025; RECS exam single-family homes'},
        'exam': {'short': 'Five platforms\' real exams', 'summary': 'Each law run once on its platform\'s real exam by a separate script, beside the textbook law tuned on dev, the collection\'s network and a dumb baseline.',
                 'results': results, 'coverage': coverage},
        'limits': json.load(open(f'{ROOT}/web/limits.json')) if os.path.exists(f'{ROOT}/web/limits.json') else [],
        'help_wanted': ['RoadSim trip-energy law (VED/EPA exam) not built in this batch', 'Daymet/NLDAS forcing would separate law skill from ERA5-Land bias (WaterData help_wanted)'],
        'next': json.load(open(f'{ROOT}/web/next.json')) if os.path.exists(f'{ROOT}/web/next.json') else [],
        'usage': 'from lawfinder import LawFinder; lf = LawFinder(); mu, sigma = lf["water"].predict(phi=0.8, fs=0.1, sync=-0.2, S=0.14)',
        'files': ['lawfinder.py', 'lawfinder_laws.json', 'demo.py', 'examples/', 'training/', 'splits.json'],
        'demo_seconds': 1,
    }
    return y


def ratchet_yaml(C):
    ladders = []
    for n, c in C.items():
        R, E = c['R'], c['E']
        k = KEY[n][0]
        dec = {d['id']: d for d in R['history'][-1]['decisions']}
        rungs = []
        for r in R['rungs']:
            d = dec[r['id']]
            ex = (E.get('rungs') or {}).get(r['id'], {})
            rungs.append({'id': r['id'], 'change': f"{r['change']}: {c['L'].TARGET['symbol']} = {c['txt'](r)}",
                          'sim': r3(r['sim']), 'decision': 'released' if r['id'] == R['released'] else d['decision'],
                          'why': d['why'], 'metrics': [r3(r['dev']), r3(ex.get(k))]})
        ladders.append({'name': f"LawFinder {c['L'].TITLE} ({PLAT[n]})",
                        'objective': f"R^2 of {c['L'].TARGET['label']} on the frozen simulated set",
                        'rule': R['rule'], 'margin': R['margin'], 'dev_gate': 'Dev error may not get more than 5 % worse',
                        'frozen_sets': [h['frozen_set'] for h in R['history']], 'scored_on': R['history'][-1]['frozen_set'],
                        'columns_source': 'Dev error (the gate), then the real exam metric reported after the fact (never used to choose)',
                        'columns': ['Dev error', f'Exam {KEY[n][1]}'], 'rungs': rungs})
    return {'model': 'lawfinder', 'ladders': ladders}


def main():
    C = collect()
    os.makedirs(REL, exist_ok=True)
    yaml.safe_dump(model_yaml(C), open(f'{REL}/model.yaml', 'w'), sort_keys=False, allow_unicode=True, width=120)
    yaml.safe_dump(ratchet_yaml(C), open(f'{REL}/ratchet.yaml', 'w'), sort_keys=False, allow_unicode=True, width=120)
    print('laws in card:', list(C))


if __name__ == '__main__':
    main()
