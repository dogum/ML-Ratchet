"""Injection tests: take real, healthy two-week stretches of one 2107 inverter (all 23 siblings agree it was healthy),
add a known fault with physics, and ask the network to find it. Real weather, real sensor quirks, real noise; known truth."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC)
import realexam as RX, simbatch as SB, evaluate as E, pvsim as S

CLIP = 0.806            # this inverter's AC limit per unit of DC nameplate (30 kW / 37.2 kW)

def unclipped(p, pe):
    """estimate the power the array would have delivered without the AC limit: unclipped steps as measured; clipped steps
    from the physics prior scaled by that day's measured/expected ratio on unclipped daylight"""
    r = np.where((p < 0.95 * CLIP) & (pe > 0.1), p / np.maximum(pe, 1e-3), np.nan)
    with np.errstate(all='ignore'): k = np.nanmedian(r, 1, keepdims=True)
    return np.where(p >= 0.95 * CLIP, np.maximum(p, pe * np.nan_to_num(k, nan=1.0)), p)

def windows(fr, P, G, n, rng, D=14):
    out = []
    while len(out) < n:
        j = int(rng.integers(24)); s = int(rng.integers(0, len(G) - D))
        f = fr[s:s + D, j]; p = P[s:s + D, :, j]
        if np.all(np.isfinite(f)) and np.all(np.abs(f - 1) < 0.03) and (np.isfinite(p) & np.isfinite(G[s:s + D])).mean() > 0.9 and np.nanmax(p) > 0.3:
            out.append((j, s))
    return out

def predict_window(net, p, g, ta, geo, sl, ref):
    X = RX.channels(p, g, ta, geo, sl)
    X = np.concatenate([X, np.full((1,) + X.shape[1:], ref, np.float32), np.full((1,) + X.shape[1:], 1.0, np.float32)])
    return {k: v[0] for k, v in E.predict(net, X[None]).items()}

def main(ckpt, n=120, seed=5):
    net = E.load(ckpt); days, P, G, T, fr, geo = RX.load(); rng = np.random.default_rng(seed); D = 14
    W = windows(fr, P, G, n, rng); res = {k: [] for k in ('dead', 'soil', 'shade', 'clip', 'outage')}
    for (j, s) in W:
        sl = slice(s, s + D); p = np.nan_to_num(P[sl, :, j].astype(float)); g = G[sl]; ta = T[sl]
        pe = SB.expected_power(np.nan_to_num(np.maximum(g, 0)), np.nan_to_num(ta, nan=15), geo['ca'][sl])
        pu = unclipped(p, pe); base = predict_window(net, p, g, ta, geo, slice(0, D), 1.0) if False else None
        Pfull = P[:, :, j].astype(float)
        def run(pnew):
            Pm = Pfull.copy(); Pm[sl] = pnew
            o = predict_window(net, Pm, G, T, geo, sl, ref_level)
            return o
        Pm = Pfull.copy(); ref_level = 0.0
        o0 = predict_window(net, Pfull, G, T, geo, sl, 0.0); ref_level = float(np.nanmax(o0['pf']))     # the healthy level of this stretch
        o0 = run(p)
        # 1. a string dies (1 or 2 of 6) on day 7
        k = int(rng.choice([1, 2])); fr_t = np.where(np.arange(D) >= 7, 1 - k / 6, 1.0)
        o = run(np.minimum(CLIP, pu * fr_t[:, None]))
        res['dead'].append({'k': k, 'k_pred_after': float(np.median(np.round(6 * (1 - np.clip(o['frac'][8:], 0, 1))))), 'k_pred_before': float(np.median(np.round(6 * (1 - np.clip(o['frac'][:6], 0, 1))))),
                            'step_day_pred': int(np.argmax(o['step_p'])), 'step_p7': float(o['step_p'][7])})
        # 2. soiling: a ramp to 2-8% over the fortnight
        smax = rng.uniform(0.02, 0.08); sd = smax * np.arange(D) / (D - 1)          # 0.15-0.6 %/day, a realistic dusty-season rate
        o = run(np.minimum(CLIP, pu * (1 - sd)[:, None]))
        res['soil'].append({'true_last': float(smax), 'pred_last': float(o['soil'][-1] - o0['soil'][-1]), 'pf_drop_true': float(smax), 'pf_drop_pred': float(o0['pf'][-1] - o['pf'][-1]),
                            'k_pred_last': float(np.round(6 * (1 - np.clip(o['frac'][-1], 0, 1))))})
        # 3. shading: a tree to the east-southeast blocks the low morning sun (beam only)
        az0 = rng.uniform(95, 135); hgt = rng.uniform(12, 28)
        sf = S.horizon_shade(geo['el'][sl], geo['az'][sl], np.array([0, az0 - 25, az0 - 20, az0 + 20, az0 + 25, 359.9]), np.array([0, 0, hgt, hgt, 0, 0]), 4.0)
        kc = np.nan_to_num(np.maximum(g, 0) / np.maximum(geo['poa_clear'][sl] / 1000.0, 0.05)); beam = np.clip((kc - 0.3) / 0.7, 0, 1) * 0.85
        ps = pu * (1 - sf * beam); true_sh = 1 - ps.sum(1) / np.maximum(pu.sum(1), 1e-6)
        o = run(np.minimum(CLIP, ps))
        res['shade'].append({'true': float(true_sh.mean()), 'pred': float(np.mean(o['shade'] - o0['shade']))})
        # 4. a smaller inverter: AC limit at 0.70 of DC nameplate (DC/AC ratio 1.43)
        pc = np.minimum(pu, 0.70); true_c = 1 - pc.sum(1) / np.maximum(pu.sum(1), 1e-6); o = run(pc)
        res['clip'].append({'true': float(true_c.mean()), 'pred': float(o['clip'].mean()), 'ac_limit_pred': float(o['ac_limit']), 'ac_limit_true': 0.70})
        # 5. outages: a 3-hour trip on day 3 and a dead day on day 9
        po = p.copy(); po[3, 44:56] = 0; po[9] = 0
        true_o = np.zeros(D); true_o[3] = 1 - po[3].sum() / max(p[3].sum(), 1e-6); true_o[9] = 1.0; o = run(po)
        res['outage'].append({'true3': float(true_o[3]), 'pred3': float(o['outage'][3]), 'pred9': float(o['outage'][9]), 'false': float(np.mean(np.delete(o['outage'], [3, 9])))})
    S_ = {}
    d = res['dead']; S_['dead_count_right_after'] = float(np.mean([r['k_pred_after'] == r['k'] for r in d])); S_['dead_healthy_before'] = float(np.mean([r['k_pred_before'] == 0 for r in d]))
    S_['dead_onset_found_pm1'] = float(np.mean([abs(r['step_day_pred'] - 7) <= 1 for r in d]))
    so = res['soil']; S_['soil_pf_drop_mae'] = float(np.mean([abs(r['pf_drop_pred'] - r['pf_drop_true']) for r in so])); S_['soil_attrib_mae'] = float(np.mean([abs(r['pred_last'] - r['true_last']) for r in so]))
    S_['soil_mistaken_for_dead_string'] = float(np.mean([r['k_pred_last'] > 0 for r in so]))
    sh = res['shade']; S_['shade_true_mean'] = float(np.mean([r['true'] for r in sh])); S_['shade_mae'] = float(np.mean([abs(r['pred'] - r['true']) for r in sh])); S_['shade_corr'] = float(np.corrcoef([r['true'] for r in sh], [r['pred'] for r in sh])[0, 1])
    c = res['clip']; S_['clip_true_mean'] = float(np.mean([r['true'] for r in c])); S_['clip_mae'] = float(np.mean([abs(r['pred'] - r['true']) for r in c])); S_['ac_limit_mae'] = float(np.mean([abs(r['ac_limit_pred'] - 0.7) for r in c]))
    ou = res['outage']; S_['outage_trip_mae'] = float(np.mean([abs(r['pred3'] - r['true3']) for r in ou])); S_['outage_dead_day_pred'] = float(np.mean([r['pred9'] for r in ou])); S_['outage_false_mean'] = float(np.mean([r['false'] for r in ou]))
    S_['n_windows'] = len(W)
    print(json.dumps({k: round(v, 3) for k, v in S_.items()}, indent=1))
    json.dump({'summary': S_, 'cases': res}, open(ckpt.replace('.pt', '_inject.json'), 'w'))
    return S_

if __name__ == '__main__':
    main(sys.argv[1])
