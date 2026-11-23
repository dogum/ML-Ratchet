"""Dev only: how far are real P picks from the ak135 P time predicted from the USGS origin, compared with simulated
picks (simulated picker error + the simulated 3-D residual)? Real: pick - (USGS origin + ak135 P(distance, depth)).
Also the S picks. Writes data/calib_picks.json."""
import sys, json, numpy as np
from _paths import H; sys.path.insert(0, H + 'src'); sys.path.insert(0, H + 'src/v12')
import realnodes as RN, pipeline as PL, sim2 as S2, tt as TT
SP = json.load(open(H + 'splits.json')); out = dict(real=[], real_s=[], peak=[])
for sp, k, s, n, f in RN.records(('dev',)):
    z = np.load(f); x = z['x'].astype(np.float64); m = json.loads(str(z['meta'])); e = SP['events'][k]
    w, ls, p, peak = PL.read(x, 300); t_start = m['p_time'] - 300 - m['event']['time']; d = m['dist']
    out['real'].append(t_start + p - float(TT.tt('P', d, max(e['depth'], 1)))); out['peak'].append(peak)
    spk = S2.s_pick(x, p); out['real_s'].append((t_start + spk - float(TT.tt('S', d, max(e['depth'], 1)))) if spk >= 0 else None)
B = np.load(H + 'bank/val1_samples.npz'); sim = B['n_pickerr'] + B['n_res']; simS = B['n_serr'] + B['n_res']
r = np.array(out['real']); rs = np.array([v for v in out['real_s'] if v is not None])
q = lambda v: [round(float(x), 1) for x in np.percentile(v, [5, 16, 50, 84, 95])]
res = dict(n_real=len(r), real_P_pct_5_16_50_84_95=q(r), sim_P_pct=q(sim), real_P_within5=float(np.mean(np.abs(r) < 5)), sim_P_within5=float(np.mean(np.abs(sim) < 5)),
           real_P_robust_sd=float(1.4826 * np.median(np.abs(r - np.median(r)))), sim_P_robust_sd=float(1.4826 * np.median(np.abs(sim - np.median(sim)))),
           real_S_found=float(len(rs) / len(r)), sim_S_found=float(np.mean(np.isfinite(simS))), real_S_pct=q(rs), sim_S_pct=q(simS[np.isfinite(simS)]),
           real_peak_median=float(np.median(out['peak'])), sim_peak_median=float(np.median(B['n_peak'])))
json.dump(dict(summary=res, real=out['real'], real_s=out['real_s']), open(H + 'data/calib_picks.json', 'w'), indent=0); print(json.dumps(res, indent=1))
