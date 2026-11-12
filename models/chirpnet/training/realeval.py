"""Final exam: real LIGO strain around four events, plus time-slid real background (no true coincidences)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np, torch
sys.path.insert(0, _P.SRC)
import gwphys as G
from evaluate import load, run

CACHE = _P.WORK + '/real_whitened.npz'

def prepare():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True); return z['data'].item()
    out = {}
    for name, ev in G.events().items():
        h, t0 = G.load_strain(os.path.join(G.ROOT, 'data_raw', ev['fn_H1']))
        l, _ = G.load_strain(os.path.join(G.ROOT, 'data_raw', ev['fn_L1']))
        out[name] = {'H1': G.whiten(h).astype(np.float32), 'L1': G.whiten(l).astype(np.float32), 't0': t0, 'tev': ev['tevent'],
                     'm1': ev['m1'], 'm2': ev['m2']}
    np.savez(CACHE, data=np.array(out, dtype=object))
    return out

def windows(h, l, starts):
    return np.stack([np.stack([h[s:s + G.WIN], l[s:s + G.WIN]]) for s in starts]).astype(np.float32)

def scan(net, h, l, stride=128, lo=2.0, hi=30.0, tta=False):
    starts = np.arange(int(lo * G.FS), int(hi * G.FS) - G.WIN, stride)
    o = run(net, windows(h, l, starts), tta=tta)
    return starts, o

def hard_background(net, slides=range(3, 26)):
    """time slides WITHOUT removing the events: a real chirp in one detector against noise in the other must not count"""
    D = prepare(); out = []
    for name, d in D.items():
        core = slice(int(2 * G.FS), int(30 * G.FS)); hc, lc = d['H1'][core], d['L1'][core]; n = len(hc)
        for k_s in slides:
            ls = np.roll(lc, int(k_s * G.FS))
            out.append(run(net, windows(hc, ls, list(range(0, n - G.WIN, 256))))[:, 0])
    return np.concatenate(out)

def exam(net, tta=False, slides=range(3, 26)):
    D = prepare(); res = {'events': {}, 'background': {}}
    bg = []
    for name, d in D.items():
        h, l = d['H1'], d['L1']; tev = d['tev'] - d['t0']
        st, o = scan(net, h, l, stride=64, tta=tta)
        tm = st / G.FS + o[:, 5]
        near = np.abs(st / G.FS + 1.25 - tev) < 0.7          # windows whose merger slot covers the event
        k = np.argmax(np.where(near, o[:, 0], -1e9))
        good = near & (o[:, 0] > o[k, 0] - 2.0)
        w = np.exp(-2 * np.clip(o[good, 2], -7, 3))
        lmc = np.sum(w * o[good, 1]) / np.sum(w); slmc = np.sqrt(1 / np.mean(w))
        q = float(np.clip(np.mean(o[good, 3]), 0.05, 1))
        mc = float(np.exp(lmc)); m1, m2 = G.masses_from_mc_q(mc, q)
        res['events'][name] = {'score': float(o[k, 0]), 'mc': mc, 'mc_lo': float(np.exp(lmc - slmc)), 'mc_hi': float(np.exp(lmc + slmc)),
                               'q': q, 'm1': float(m1), 'm2': float(m2), 't_pred': float(tm[k] - tev),
                               'mc_ref': float(G.chirp_mass(d['m1'], d['m2'])), 'q_ref': float(d['m2'] / d['m1']),
                               'trace_t': (st / G.FS + 1.25 - tev).tolist(), 'trace_s': o[:, 0].tolist()}
        # background: slide L1 against H1 by whole seconds (circular within the clean core)
        core = slice(int(2 * G.FS), int(30 * G.FS))
        hc, lc = h[core], l[core]; n = len(hc); te = tev - 2
        for k_s in slides:
            sh = int(k_s * G.FS); ls = np.roll(lc, sh)
            starts = np.arange(0, n - G.WIN, 256)
            # drop windows containing the event in either detector
            tl = (te + k_s) % (n / G.FS)
            keep = [s for s in starts if not (s / G.FS - 0.5 < te < s / G.FS + 2.5) and not (s / G.FS - 0.5 < tl < s / G.FS + 2.5)]
            ob = run(net, windows(hc, ls, keep), tta=tta)
            bg.append(ob[:, 0])
    bg = np.concatenate(bg)
    live_s = len(bg) * 256 / G.FS
    res['background'] = {'n_windows': int(len(bg)), 'livetime_s': float(live_s), 'max': float(bg.max()),
                         'q999': float(np.quantile(bg, 0.999)), 'scores_hist': np.histogram(bg, bins=60, range=(-20, 20))[0].tolist()}
    for name, e in res['events'].items():
        e['n_bg_louder'] = int(np.sum(bg >= e['score']))
        e['far_per_hour_ub'] = float((e['n_bg_louder'] + 1) / (live_s / 3600))
    return res

if __name__ == '__main__':
    net = load(sys.argv[1]); r = exam(net, tta='--tta' in sys.argv)
    for k, e in r['events'].items():
        print(f"{k:10s} score {e['score']:6.2f}  Mc {e['mc']:5.1f} [{e['mc_lo']:.1f},{e['mc_hi']:.1f}] ref {e['mc_ref']:5.1f}  q {e['q']:.2f} ref {e['q_ref']:.2f}  dt {1000*e['t_pred']:5.1f} ms  bg louder {e['n_bg_louder']}")
    b = r['background']; print('background windows', b['n_windows'], 'livetime %.0f s' % b['livetime_s'], 'max %.2f' % b['max'], 'q999 %.2f' % b['q999'])
    json.dump(r, open(sys.argv[1].replace('.pt', '_real.json'), 'w'))
