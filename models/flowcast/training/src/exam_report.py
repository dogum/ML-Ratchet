"""Final exam tables for one checkpoint (run ONCE, after every decision was made on sim + dev).
usage: python3 exam_report.py <ckpt> <tag>
Needs results/<tag>.parquet + preds/<tag>_{val,recent}.npz from evalreal.py exam.
Writes results/exam_<tag>.json: FlowCast vs B's baselines on the same basins, flood peaks, coverage, strata."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import waterdata as wd
from scipy.stats import spearmanr

BL = f'{WATER_DATA}/data/proc/baselines'
R_INST = json.load(open(f'{BL}/peak_summary.json'))['ratio_inst_over_daily_dev']   # 1.384, B's dev ratio


def wyear(idx):
    return idx.year + (idx.month >= 10).astype(int)


def ann_max(s, minn=330):
    s = s.dropna(); w = wyear(s.index); g = s.groupby(w); c = g.count()
    return g.max()[c >= minn]


def main(ckpt, tag):
    fc = pd.read_parquet(f'{ROOT}/results/{tag}.parquet')
    fc['method'] = 'flowcast'
    bl = pd.read_parquet(f'{BL}/flow_scores.parquet')
    bl = bl[bl.split == 'exam']
    a = wd.attributes('camels')
    out = {'ckpt': ckpt, 'tag': tag, 'flow': {}, 'peaks': {}, 'strata': {}}
    for per in ('val', 'recent'):
        f = fc[fc.period == per].set_index('basin')
        for m in ('flowcast', 'nn_dev', 'nn_k12', 'gr4j', 'nwm', 'clim', 'persist1'):
            g = f if m == 'flowcast' else bl[(bl.period == per) & (bl.method == m)].set_index('basin')
            common = g.index.intersection(f.index)
            gc = g.loc[common]
            out['flow'][f'{per}/{m}'] = {
                'n': int(len(g)), 'nse': float(g.nse.median()), 'kge': float(g.kge.median()),
                'fhv': float(g.fhv.median()), 'flv': float(g.flv.median()),
                'peak_abs_err_daily': float(g.peak_abs_err_pct_median.median()),
                'timing_d': float(g.peak_timing_days.median()), 'frac_nse_lt0': float((g.nse < 0).mean()),
                'nse_531': float(g[a.loc[g.index, 'in_531'].values].nse.median()),
                'n_common': int(len(common)), 'nse_common': float(gc.nse.median()),
                'flowcast_nse_common': float(f.loc[common].nse.median()),
                'flowcast_wins_frac': float((f.loc[common].nse > gc.nse).mean()) if m != 'flowcast' else None}
            if m == 'flowcast':
                out['flow'][f'{per}/{m}'].update({'cov90': float(f.cov90.median()), 'cov50': float(f.cov50.median()),
                                                  'cov90_pooled_mean': float(f.cov90.mean()),
                                                  'width90_rel': float(f.width90_rel.median())})
        # strata (FlowCast vs U-dev and NWM, NSE medians)
        aa = a.loc[f.index]
        strata = {'snow (frac_snow>0.3)': aa.frac_snow > 0.3, 'arid (aridity>1)': aa.aridity_FAO_PM > 1,
                  'humid (aridity<0.6)': aa.aridity_FAO_PM < 0.6, 'regulated (reg_class mid/high)': aa.reg_class.isin(['mid', 'high']),
                  'large (>2000 km2)': aa['area'] > 2000}
        for sname, mask in strata.items():
            ids = f.index[mask.values]
            row = {'n': int(len(ids)), 'flowcast': float(f.loc[ids].nse.median())}
            for m in ('nn_dev', 'nwm', 'gr4j'):
                g = bl[(bl.period == per) & (bl.method == m)].set_index('basin')
                row[m] = float(g.loc[g.index.intersection(ids)].nse.median())
            out['strata'][f'{per}/{sname}'] = row
    # flood peaks: annual instantaneous peaks vs FlowCast annual max daily x R (B's protocol)
    pk = pd.read_parquet(f'{WATER_DATA}/data/proc/nwis/peaks.parquet')
    bad = pk.peak_cd.str.contains(r'(^|,)(5|6|7|C)(,|$)', regex=True) | pk.date_partial | pk.peak_cms.isna()
    pk = pk[~bad]
    rows = []
    for per, (s0, s1), (y0, y1) in [('val', VAL, (1990, 1999)), ('recent', REC, (2016, 2023))]:
        P = np.load(f'{ROOT}/preds/{tag}_{per}.npz')
        idx = pd.date_range(s0, s1)
        for b in P.files:
            p = pk[pk.gauge_id == b].groupby('wy').peak_cms.max().loc[y0:y1]
            if len(p) == 0:
                continue
            area = wd.spec(b)['area_km2']
            arr = P[b]
            for name, col in (('mean', 0), ('q95', 5)):
                e = ann_max(pd.Series(arr[:, col] * area / 86.4, index=idx)) * R_INST
                j = pd.concat([p, e], axis=1, keys=['o', 'e']).dropna()
                j = j[j.o > 0]
                for y, r in j.iterrows():
                    rows.append({'basin': b, 'wy': int(y), 'period': per, 'est': name, 'obs': r.o, 'e': r.e,
                                 'pct_err': 100 * (r.e - r.o) / r.o})
    sc = pd.DataFrame(rows)
    sc.to_parquet(f'{ROOT}/results/peaks_{tag}.parquet')
    for (per, name), g in sc.groupby(['period', 'est']):
        pb = g.groupby('basin').pct_err
        out['peaks'][f'{per}/{name}'] = {'n_basins': int(g.basin.nunique()), 'n_peaks': int(len(g)),
                                          'median_abs_pct_err': float(pb.apply(lambda x: x.abs().median()).median()),
                                          'median_bias_pct': float(pb.median().median()),
                                          'spearman_pooled_log': float(spearmanr(np.log(g.e), np.log(g.obs)).statistic)}
    out['peaks']['baselines_B'] = json.load(open(f'{BL}/peak_summary.json'))
    jdump(out, f'{ROOT}/results/exam_{tag}.json')
    for k, v in out['flow'].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    for k, v in out['peaks'].items():
        if k != 'baselines_B':
            print(k, v)
    for k, v in out['strata'].items():
        print(k, v)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
