"""Cross-check SunDoctor's long-run view of one inverter against NREL's RdTools (independent, standard methods):
year-on-year degradation and stochastic-rate-and-recovery (SRR) soiling, on the same inverter's data."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, warnings, numpy as np, pandas as pd
sys.path.insert(0, _P.SRC)
import realexam as RX, simbatch as SB, evaluate as E
warnings.filterwarnings('ignore')
import rdtools

def daily_pi(p, g, ta, geo, clip=0.806):
    """daily performance index for RdTools: measured / expected energy over unclipped daylight"""
    pe = SB.expected_power(np.nan_to_num(np.maximum(g, 0)), np.nan_to_num(ta, nan=15), geo['ca'])
    m = np.isfinite(p) & np.isfinite(g) & (g > 0.2) & (p < 0.9 * clip) & (pe > 0.05)
    num = np.where(m, p, 0).sum(1); den = np.where(m, pe, 0).sum(1); ins = np.where(np.isfinite(g), np.maximum(g, 0), 0).sum(1) * 0.25
    pi = np.where((m.sum(1) >= 8) & (den > 0), num / np.maximum(den, 1e-9), np.nan)
    return pi, ins

def main(ckpt, j=10):
    days, P, G, T, fr, geo = RX.load()
    p = P[:, :, j].astype(float); valid = (np.isfinite(p) & np.isfinite(G)).mean(1) > 0.5
    import torch
    if torch.load(ckpt, weights_only=False).get('version') == 2:          # v2 checkpoints run through the v2 inputs
        import evaluate2 as E2, realexam2 as RX2; o = RX2.diagnose(E2.load(ckpt), p, G, T, geo, False)
    else:
        net = E.load(ckpt); o1 = RX.run_inverter(net, p, G, T, geo); o = RX.run_inverter(net, p, G, T, geo, RX.reference(o1['pf'], valid & (o1['outage'] < 0.2)))
    healthy = np.isfinite(fr[:, j]) & (np.abs(fr[:, j] - 1) < 0.04) & valid & (o['outage'] < 0.1)
    pi, ins = daily_pi(p, G, T, geo)
    idx = pd.DatetimeIndex(days).tz_localize('US/Pacific', ambiguous='NaT', nonexistent='NaT')
    s = pd.Series(np.where(healthy, pi, np.nan), index=idx).dropna(); s = s[(s > 0.5) & (s < 1.3)]
    rd, ci, info = rdtools.degradation_year_on_year(s, confidence_level=68.2)
    # SunDoctor: degradation = linear trend of its clean-array level (pf / (1 - soil)) on healthy days
    clean = pd.Series(np.where(healthy, o['pf'] / np.maximum(1 - o['soil'], 0.5), np.nan), index=idx).dropna()
    yrs = (clean.index - clean.index[0]).days / 365.25; sl, ic = np.polyfit(yrs, clean.values, 1); sd_rd = 100 * sl / ic
    # soiling
    srr_out = None
    try:
        from rdtools import soiling as _soil
        di = pd.date_range(idx.min(), idx.max(), freq='D')
        sd = pd.Series(np.where(healthy, pi, np.nan), index=idx).reindex(di); sd[(sd < 0.5) | (sd > 1.3)] = np.nan
        insol = pd.Series(ins, index=idx).reindex(di)
        sr, sr_ci, sr_info = _soil.soiling_srr(sd, insol, reps=200)
        srr_out = {'insolation_weighted_soiling_ratio': float(sr), 'ci': [float(x) for x in sr_ci]}
        prof = sr_info['soiling_ratio_perfect_clean']
        sd_soil = pd.Series(o['soil'], index=idx).reindex(prof.index)
        both = pd.concat([1 - prof, sd_soil], axis=1).dropna(); srr_out['daily_corr_with_sundoctor'] = float(both.corr().iloc[0, 1])
        srr_out['mean_loss_rdtools'] = float(both.iloc[:, 0].mean()); srr_out['mean_loss_sundoctor'] = float(both.iloc[:, 1].mean())
        mon = both.groupby(both.index.month).mean(); srr_out['monthly'] = {int(m): [round(float(a), 4), round(float(b), 4)] for m, (a, b) in mon.iterrows()}
    except Exception as e:
        srr_out = {'error': str(e)[:200]}
    out = {'inverter': j + 1, 'healthy_days': int(healthy.sum()), 'rdtools_yoy_degradation_pct_per_yr': float(rd), 'rdtools_ci68': [float(c) for c in ci],
           'sundoctor_degradation_pct_per_yr': float(sd_rd), 'soiling': srr_out}
    print(json.dumps(out, indent=1)); json.dump(out, open(ckpt.replace('.pt', '_rdtools.json'), 'w'), indent=1)
    np.savez_compressed(ckpt.replace('.pt', '_inv%02d_series.npz' % (j + 1)), days=np.asarray(days.values.astype('datetime64[D]').astype(np.int64)),
                        pi=pi, healthy=healthy, **{k: o[k] for k in ('pf', 'soil', 'frac', 'shade', 'clip', 'outage', 'step_p') + (('snow', 'stuck') if 'snow' in o else ())})
    return out

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 10)
