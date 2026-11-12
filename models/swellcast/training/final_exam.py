"""Final exam for the incumbent rung (last accepted in ratchet.json). Nothing here feeds back into any choice.
1. sigma factors fitted on DEV only (calib_sigma.fit, widen-only), detection threshold from the frozen real null set (2024);
2. network, classical ridge fit and dumb median guess on dev / exam (confirmed = headline, visible, all pairs);
3. coverage of the 68% error bars; per-basin and per-distance breakdown; false alarms on quiet 2025 windows.
Writes final.json."""
import os, sys, json, numpy as np
import evaluate as E, calib_sigma as CS, baseline as BL

def by(mask_fn, ev, P, D, t0, dr, hd):
    out = {}
    for k, m in mask_fn(ev).items():
        m = np.asarray(m)
        if m.sum() < 3: continue
        Q = {a: (b[m] if hasattr(b, '__len__') and len(b) == len(m) else b) for a, b in P.items()}
        out[k] = E.metrics(Q, D[m], t0[m], dr[m], None, hd[m])
    return out

if __name__ == '__main__':
    log = json.load(open(E.ROOT + '/ratchet.json')); inc = [r for r in log if r['accepted']][-1]; ck = inc['ckpt']; print('incumbent', ck)
    net = E.load_net(ck); sf = CS.fit(ck); thr = E.det_threshold(net); sf['det_thr'] = thr; json.dump(sf, open(E.SIGF, 'w'), indent=1)
    base = json.load(open(E.ROOT + '/baseline.json')); bdet = json.load(open(E.ROOT + '/baseline_det.json'))
    out = dict(ckpt=ck, sigma_factor=sf, det_thr=thr, quiet=E.quiet(net, thr), classical_det=bdet)
    for split, vis in (('dev', 'confirmed'), ('exam', 'confirmed'), ('exam', True), ('exam', None)):
        r, P, ev = E.real(net, split, vis, sigf=sf, return_pred=True); key = f'{split}_{vis}'
        D, t0, dr, hd, w = BL.truth(ev)
        r_raw = E.real(net, split, vis, sigf=dict(det_thr=thr))
        out[key] = dict(network=r, network_uncalibrated_coverage={k: r_raw[k] for k in ('logD_in_1sig', 't0_in_1sig', 'dir_in_1sig') if k in r_raw},
                        classical=base[key]['classical'], dumb=base[key]['dumb'], classical_detect_rate=bdet.get(f'{key}_detect_rate'))
        out[key]['by_basin'] = by(lambda ev: {b: [e['basin'] == b for e in ev] for b in ('EP', 'NA', 'WP', 'SP')}, ev, P, D, t0, dr, hd)
        out[key]['by_distance'] = by(lambda ev: {'1000-2500 km': [1e6 <= e['D'] < 2.5e6 for e in ev], '2500-5000 km': [2.5e6 <= e['D'] < 5e6 for e in ev], '5000-8000 km': [5e6 <= e['D'] <= 8e6 for e in ev]}, ev, P, D, t0, dr, hd)
        out[key]['by_contrast'] = by(lambda ev: {'faint (contrast < 0.3 dex)': [not (e['contrast'] is not None and e['contrast'] >= 0.3) for e in ev], '0.3-0.6 dex': [e['contrast'] is not None and 0.3 <= e['contrast'] < 0.6 for e in ev], 'strong (>= 0.6 dex)': [e['contrast'] is not None and e['contrast'] >= 0.6 for e in ev]}, ev, P, D, t0, dr, hd)
        if key == 'exam_confirmed':
            out['exam_pred'] = [dict(sid=e['sid'], name=e['name'], buoy=e['buoy'], D=e['D'] / 1e3, t0=e['t0_rel'], dir=e['dir'], pD=float(P['D_km'][i]), pt0=float(P['t0'][i]), pdir=float(P['dir'][i]), pdet=float(P['pdet'][i])) for i, e in enumerate(ev)]
        print(key, 'n', r['n'], {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    print('quiet', out['quiet']); json.dump(out, open(E.ROOT + '/final.json', 'w'), indent=1)
