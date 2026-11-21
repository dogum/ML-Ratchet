"""Summarise spread baseline rows -> proc/baselines/spread_summary.json and a markdown table (stdout).
Growth fire-days only (observed new area > 0). Means with 95 % bootstrap CIs over FIRES (1000 resamples, seed 0)."""
import json, sys
import numpy as np, pandas as pd
import os
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
P = f'{_DATA}/data/proc'
KEYS = ['ap', 'f1', 'dice_new', 'dice_total', 'log10_area_ratio']


def rows(split):
    out = []
    for l in open(f'{P}/baselines/spread_{split}.jsonl'):
        out += [r for r in json.loads(l)['rows'] if r['method'] != '_kgrid']
    return pd.DataFrame(out)


def boot(g, key, n=1000):
    per = g.groupby('fire_id')[key].agg(['sum', 'count'])
    per = per[per['count'] > 0]
    rng = np.random.default_rng(0); s, c = per['sum'].values, per['count'].values
    if len(s) < 2: return [np.nan, np.nan]
    i = rng.integers(0, len(s), (n, len(s)))
    m = s[i].sum(1) / c[i].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def table(d):
    res = {}
    for m, g in d.groupby('method'):
        r = dict(n_fire_days=int(len(g)), n_fires=int(g.fire_id.nunique()))
        for k in KEYS:
            gg = g[np.isfinite(g[k])]
            r[k + '_mean'] = float(gg[k].mean()); r[k + '_median'] = float(gg[k].median())
            if k in ('ap', 'f1', 'dice_new'): r[k + '_mean_ci95'] = boot(gg, k)
        r['abs_area_err_km2_median'] = float(g.area_err_km2.abs().median())
        res[m] = r
    return res


def main():
    f = pd.read_parquet(f'{P}/fires.parquet').set_index('Event_ID')
    S = {}
    for split in ['dev', 'exam']:
        d = rows(split)
        d = d[d.new_obs_km2 > 0]
        d['period'] = d.fire_id.map(f.exam_period); d['region'] = d.fire_id.map(f.region); d['fuel'] = d.fire_id.map(f.fuel_group)
        S[split] = dict(all=table(d))
        if split == 'exam':
            for per in ['early', 'main', 'recent']: S[split]['period_' + per] = table(d[d.period == per])
            for reg in sorted(d.region.dropna().unique()): S[split]['region_' + reg] = table(d[d.region == reg])
            for fu in sorted(d.fuel.dropna().unique()): S[split]['fuel_' + fu] = table(d[d.fuel == fu])
        S[split]['n_fires_scored'] = int(d.fire_id.nunique())
    S['params'] = json.load(open(f'{P}/baselines/spread_params.json'))
    S['published_context'] = {
        'source': 'Gerard et al. 2023, WildfireSpreadTS (NeurIPS Datasets & Benchmarks), Table 5, test AP mean +- sd over 12 year folds',
        'task_difference': 'WSTS target = next-day VIIRS active-fire pixels at 375 m (not newly burned area), pooled test AP; '
                           'not comparable number-for-number with our FEDS new-burn AP at 90 m',
        'persistence_1d': '0.193 +- 0.065', 'logistic_regression_1d': '0.286 +- 0.092', 'res18_unet_1d_all': '0.341 +- 0.086',
        'utae_5d_vegetation': '0.372 +- 0.088'}
    json.dump(S, open(f'{P}/baselines/spread_summary.json', 'w'), indent=1)
    for split in ['dev', 'exam']:
        print(f'\n{split}: growth fire-days, {S[split]["n_fires_scored"]} fires')
        print('| method | n days | AP mean [95% CI] | AP median | F1 mean | Dice new mean | Dice total median | log10 area ratio median |')
        print('|---|---|---|---|---|---|---|---|')
        for m, r in S[split]['all'].items():
            print(f"| {m} | {r['n_fire_days']} | {r['ap_mean']:.3f} [{r['ap_mean_ci95'][0]:.3f}, {r['ap_mean_ci95'][1]:.3f}] | "
                  f"{r['ap_median']:.3f} | {r['f1_mean']:.3f} | {r['dice_new_mean']:.3f} | {r['dice_total_median']:.3f} | "
                  f"{r['log10_area_ratio_median']:.2f} |")


if __name__ == '__main__':
    main()
