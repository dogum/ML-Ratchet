"""Assemble web/flowcast.html from web/page.html + fcnet.js + data + weights + summary."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *

ex = json.load(open(latest('results/exam_r1b_exam_cal.json')))
R = json.load(open(latest('ratchet.json')))
now = json.load(open(latest('results/nowcast_summary.json'))) if os.path.exists(latest('results/nowcast_summary.json')) else None
r0 = json.load(open(latest('results/sim_r0_p0.json')))
d0 = pd.read_parquet(f'{ROOT}/results/r0_dev.parquet')
short = {'r1': 'Water-yield correction (first try, old loss)',
         'r0b': 'Loss fix: weight each simulated basin by what the network can see',
         'r1b': 'Water-yield correction measured on dev basins',
         'r2': 'Flashier runoff: full dev parameter shift',
         'r3': 'Realistic ERA5-Land forcing error',
         'r4': 'Flashier runoff + realistic forcing error',
         'r5': 'Longer training (5,000 more steps)'}
detail = {'r1': 'True rain = ERA5-Land × m, with m fitted on the 52 dev basins (more water in wet, mountain basins; less in dry plains). Trained with the old loss; it lost on its own frozen set, which exposed the loss bias below.',
          'r0b': "Kratzert's NSE* loss scales each basin by the spread of its own flow. In simulation that spread depends on hidden parameters, so wet, flashy draws were down-weighted and the network ran low. Scaling by Budyko's runoff ratio from the forcing removed the bias: dev KGE 0.32 → 0.44, high-flow bias −51 → −27 %.",
          'r1b': 'log m = −0.09 + 0.06 elev(km) + 0.15 snow share + 0.40 log(P/3 mm), scatter 0.15. Over-prediction in dry dev basins fell from ×1.76 to ×1.24; dev NSE 0.436 → 0.425 (within the 5 % gate). Released.',
          'r2': 'WaterSim applies half of the dev-calibrated parameter shift; with it, simulated flashiness is 0.62× the real record. The full shift matches it (1.00×). Peaks and KGE improved (KGE 0.49, FHV −16 %) but dev NSE fell to 0.380.',
          'r3': 'A dev basin\'s calibrated twin under the default forcing noise still matched its clean run at NSE 0.88, while reality allows 0.56. Daily rain cv 0.4 plus timing jitter brings that to 0.72. Lost on both sim and dev.',
          'r4': 'Retrying flashiness with noisier forcing: KGE 0.49 and peaks 45 %, but dev NSE 0.378.',
          'r5': 'Fine-tuning the incumbent for longer gave dev 0.434 but no gain on the frozen set (0.419 vs 0.425).'}
rungs = [{'rung': 'r0', 'change': '<b>Baseline.</b> WaterSim as shipped: both attribute priors, four runoff models, two snow routines, real ERA5-Land forcing from non-exam years, gauge and forcing noise. 12,000 simulated basin-runs.',
          'sim': r0['nse_median'], 'sim_inc': r0['nse_median'], 'dev': float(d0.nse.median()), 'kept': True}]
for r in R['rungs']:
    rungs.append({'rung': r['rung'], 'change': f"<b>{short[r['rung']]}.</b> {detail[r['rung']]}", 'sim': r['sim_nse'],
                  'sim_inc': r['sim_nse_incumbent'], 'dev': r['dev']['nse_median'], 'kept': r['kept']})
fl = ex['flow']
note = (f"Main exam: 619 basins, water years 1990–1999. Recent exam: 603 basins, 2016–2023. Medians over basins. "
        f"FlowCast is ungauged: it never saw flow from these basins. GR4J, persistence and climatology use each basin's own record; "
        f"NWM was calibrated by NOAA on many of these gauges with independent forcing. FlowCast beats U-dev regionalisation at "
        f"{round(100*fl['val/nn_dev']['flowcast_wins_frac'])} % of basins and NWM at {round(100*fl['val/nwm']['flowcast_wins_frac'])} %. "
        f"Its 90 % band held {round(100*fl['val/flowcast']['cov90'])} % of days in the median basin ({round(100*fl['val/flowcast']['cov90_pooled_mean'])} % on average): too wide, not too narrow.")
how = [
    ['Simulator', 'WaterSim (Phase 1): HBV, GR4J, SAC-SMA and VIC-lite runoff models, degree-day and energy-balance snow, unit-hydrograph and Muskingum–Cunge routing, reservoirs and irrigation. Verified against airGR and TUWmodel to 1e-7 mm and closed water balances.'],
    ['Simulated basins', 'Each training run is a random basin (half from jittered dev-basin attributes, half from a parametric prior) with random parameters, one model and one snow routine, driven by 5 years of real ERA5-Land forcing from a random CAMELS basin in a non-exam window. 12,000 runs per physics version, plus 800 frozen for validation.'],
    ['Real noise', 'The network sees forcing after an error model (rain bias and daily scatter, snow undercatch, temperature error) and learns a gauge-like target with rating-curve error. The physics rung adds dev-measured differences between ERA5-Land rain and the water that really leaves each basin.'],
    ['Network', 'One LSTM layer, 64 units, 25,000 weights. Each day it reads 11 weather features and 18 basin facts (area, elevation, slope, forest, soil, karst, reservoirs, irrigation, urban, and climate indices of the forcing). It outputs mean flow and five quantiles, each in units of the basin\'s mean rainfall. 365-day warm-up, then 365 days predicted.'],
    ['Ratchet', 'Frozen simulated sets per physics version, a 52-basin real dev gate on water years 2001–2014, and twin tests: each dev basin re-simulated with its calibrated parameters and with random prior parameters, read by the same network. The exam ran once, after every choice.'],
    ['Error bars', 'Quantiles from a pinball loss, widened (never narrowed) on dev only: the 50 % band × 1.00, the 90 % band × 1.08 about the median.']]
limits = [
    f"It does not beat the classical ungauged baseline overall: main-exam median NSE {fl['val/flowcast']['nse']:.2f} vs {fl['val/nn_dev']['nse']:.2f} for U-dev regionalisation, which borrows GR4J parameters calibrated on the real dev gauges. It ties or wins in arid, regulated and large basins and loses clearly in snowy and humid ones.",
    f"Snow basins are the biggest gap (median NSE {ex['strata']['val/snow (frac_snow>0.3)']['flowcast']:.2f} vs {ex['strata']['val/snow (frac_snow>0.3)']['nn_dev']:.2f} for U-dev). Only 10 of the 52 dev basins are snowy, and no snow rung was tried.",
    f"Peaks are too low: annual daily maxima × 1.38 miss instantaneous peaks by a median {ex['peaks']['val/mean']['median_abs_pct_err']:.0f} % (bias {ex['peaks']['val/mean']['median_bias_pct']:.0f} %). The network predicts a mean under uncertainty, which smooths floods. Flashier physics fixed the peaks on dev but cost NSE.",
    'The 90 % band is too wide in most basins (97 % coverage at the median) yet misses many zero-flow days in dry basins. Widen-only calibration cannot narrow it.',
    'Trained only on ERA5-Land forcing (Caravan). Other forcing products have other biases; the attributes must use the same conventions (HydroATLAS-style slope, SoilGrids conductivity).',
    'The water-yield correction was fitted on the same 52 dev basins that gate it (4 coefficients), so its dev score is partly in-sample.']
credits = ('Data: Caravan v1.6 (Kratzert et al. 2023, CC BY 4.0) with CAMELS-US (Newman et al. 2015; Addor et al. 2017), ERA5-Land forcing '
           '(Muñoz-Sabater et al. 2021, Copernicus) and HydroATLAS attributes (Linke et al. 2019); USGS NWIS daily values and annual peaks (public domain); '
           'NOAA National Water Model v3.0 retrospective (public domain). Baselines and splits: WaterData (Phase 1, agent B). Simulator: WaterSim (Phase 1, agent A). '
           'Literature context: Kratzert et al. 2019, HESS and WRR. Part of ML Ratchet: trained on physics, graded on reality.')
S = {'exam': ex, 'examNote': note, 'rungs': rungs, 'how': how, 'limits': limits, 'credits': credits, 'nowcast': now}
def clean(o):
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


DATA = json.dumps(clean(json.load(open(f'{ROOT}/web/data.json'))), separators=(',', ':'), allow_nan=False)
S = clean(json.loads(json.dumps(S, default=float)))
page = open(f'{SRC}/page.html').read()
page = page.replace('%%DATA%%', DATA).replace('%%WEIGHTS%%', open(f'{ROOT}/web/weights.json').read())
page = page.replace('%%SUMMARY%%', json.dumps(S, separators=(',', ':'), allow_nan=False)).replace('%%FCNET%%', open(f'{SRC}/fcnet.js').read())
open(f'{ROOT}/web/flowcast.html', 'w').write(page)
print('size MB', len(page) / 1e6)
