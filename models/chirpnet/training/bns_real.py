"""GW170817 final exam for the neutron-star model: 2048 s of real O2 data from Hanford and Livingston (LIGO DCC P1700349),
a scan around the merger, and a time-slide background from the same 34 minutes with the event cut out."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC)
import bns as BN, gwphys as G
from bns_train import load_bns, run

GPS_EVENT = 1187008882.43          # GW170817 merger (GWTC-1)
MC_DET = 1.1975                     # detector-frame chirp mass, +0.0001 / -0.0001 (GWTC-1)
F = {'H1': _P.WORK + '/data_bns/H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5', 'L1': _P.WORK + '/data_bns/L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5',
     'L1_raw': _P.WORK + '/data_bns/L-L1_LOSC_C00_4_V1-1187006834-4096.hdf5'}
CACHE = _P.WORK + '/bns_real_whitened.npz'

def gate(w, thresh=25.0, half=0.25, taper=0.1, fs=BN.FSB):
    """zero out (with a Tukey edge) any excursion beyond thresh sigma, as LIGO's searches do"""
    w = w.copy(); bad = np.where(np.abs(w) > thresh)[0]; spans = []
    if len(bad):
        h = int(half * fs); tp = int(taper * fs); n = len(w); win = np.ones(n, np.float32)
        groups = np.split(bad, np.where(np.diff(bad) > h)[0] + 1)
        for g in groups:
            a, b = max(0, g[0] - h), min(n, g[-1] + h); spans.append((a / fs, b / fs))
            win[a:b] = 0
            if a - tp > 0: win[a - tp:a] = np.minimum(win[a - tp:a], 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, tp)))
            if b + tp < n: win[b:b + tp] = np.minimum(win[b:b + tp], 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, tp)))
        w *= win
    return w, spans

def whiten_file(path, t_start=None, dur=None):
    x, t0 = G.load_strain(path); t0 = float(t0)
    if t_start is not None:
        i0 = int(round((t_start - t0) * 4096)); x = x[i0:i0 + int(dur * 4096)]; t0 = t_start
    fp, pp = G.psd(x, 4096, seg=8.0)
    return BN.whiten_b(x, 4096, (fp, pp)), t0, (fp, pp)

def prepare():
    try:
        z = np.load(CACHE); return {k: z[k] for k in z.files}
    except FileNotFoundError:
        pass
    H, t0, ph = whiten_file(F['H1']); L, _, pl = whiten_file(F['L1'])
    Lr, _, _ = whiten_file(F['L1_raw'], t_start=t0, dur=2048)
    out = dict(H=H, L=L, Lraw=Lr, t0=np.float64(t0), psdH=np.stack(ph), psdL=np.stack(pl))
    np.savez(CACHE, **out); return out

def windows(H, L, starts):
    W = BN.WINB; return np.stack([np.stack([H[s:s + W], L[s:s + W]]) for s in starts]).astype(np.float32)

def scan(net, H, L, t0, lo, hi, stride=0.25):
    fs = BN.FSB; starts = np.arange(int((lo - t0) * fs), int((hi - t0) * fs) - BN.WINB, int(stride * fs))
    o = run(net, windows(H, L, starts)); return starts / fs + t0, o

def background(net, H, L, t0, slides=(29, 61, 97, 131, 173, 211, 263, 307, 353, 401, 457, 503), stride=2.0, guard=40.0):
    """time slides: Livingston is shifted circularly by many seconds against Hanford, so nothing astrophysical can line up.
    Windows that hold the real event in either detector are removed."""
    fs = BN.FSB; n = len(H); W = BN.WINB; ev = (GPS_EVENT - t0) * fs
    starts = np.arange(4 * fs, n - W - 4 * fs, int(stride * fs)); scores = []
    for k in slides:
        Ls = np.roll(L, k * fs)
        keep = [s for s in starts if not (s - guard * fs < ev < s + W + guard * fs) and not (s - guard * fs < (ev + k * fs) % n < s + W + guard * fs)]
        for i in range(0, len(keep), 256):
            scores.append(run(net, windows(H, Ls, keep[i:i + 256]))[:, 0])
    s = np.concatenate(scores); return s, len(s) * stride

def single_detector_test(net, H, L, t0, k=211):
    """the one-detector check: keep Hanford's real chirp, replace Livingston with slid noise (and vice versa)"""
    ev = GPS_EVENT; fs = BN.FSB
    out = {}
    for name, (A, B) in {'h1_alone': (H, np.roll(L, k * fs)), 'l1_alone': (np.roll(H, k * fs), L)}.items():
        t, o = scan(net, A, B, t0, ev - 18, ev + 6, 0.5); out[name] = float(o[:, 0].max())
    return out

def exam(ckpt):
    net = load_bns(ckpt); D = prepare(); H, L, Lr, t0 = D['H'], D['L'], D['Lraw'], float(D['t0'])
    gH, sH = gate(H); gL, sL = gate(L); gLr, sLr = gate(Lr)
    res = {'gates_H1': sH, 'gates_L1': sL, 'gates_L1_raw': sLr, 'std_H1': float(np.std(H[100000:200000])), 'std_L1': float(np.std(L[100000:200000]))}
    t, o = scan(net, gH, gL, t0, GPS_EVENT - 40, GPS_EVENT + 20)
    i = int(np.argmax(o[:, 0])); mu, ls = o[i, 1], np.clip(o[i, 2], -7, 3)
    res['scan_t'] = (t - GPS_EVENT).round(3).tolist(); res['scan_s'] = o[:, 0].round(3).tolist()
    res['peak'] = {'score': float(o[i, 0]), 'win_start_rel': float(t[i] - GPS_EVENT), 't_merger_rel': float(t[i] + o[i, 5] - GPS_EVENT),
                   'mc': float(np.exp(mu)), 'mc_lo': float(np.exp(mu - np.exp(ls))), 'mc_hi': float(np.exp(mu + np.exp(ls))), 'q': float(o[i, 3])}
    # ensemble of windows with merger in the trained range: average Mc estimate from the confident ones
    sel = o[:, 0] > o[i, 0] - 3
    w = np.exp(-2 * np.clip(o[sel, 2], -7, 3)); res['mc_combined'] = float(np.exp(np.sum(w * o[sel, 1]) / np.sum(w)))
    t2, o2 = scan(net, gH, gLr, t0, GPS_EVENT - 40, GPS_EVENT + 20)
    j = int(np.argmax(o2[:, 0])); res['raw_glitch'] = {'score': float(o2[j, 0]), 'mc': float(np.exp(o2[j, 1])), 'scan_s': o2[:, 0].round(3).tolist()}
    t3, o3 = scan(net, H, Lr, t0, GPS_EVENT - 40, GPS_EVENT + 20)     # no gating at all
    res['raw_nogate'] = {'score': float(o3[:, 0].max()), 'scan_s': o3[:, 0].round(3).tolist()}
    bg, live = background(net, gH, gL, t0)
    res['bg_n'] = int(len(bg)); res['bg_live_s'] = float(live); res['bg_max'] = float(bg.max()); res['bg_q'] = np.quantile(bg, [0.5, 0.9, 0.99, 0.999]).round(3).tolist()
    res['bg_hist'] = np.histogram(bg, bins=50, range=(-20, 10))[0].tolist()
    res['n_bg_above_event'] = int(np.sum(bg >= o[i, 0]))
    res['single'] = single_detector_test(net, gH, gL, t0)
    return res, bg

if __name__ == '__main__':
    r, bg = exam(sys.argv[1])
    np.save(sys.argv[1].replace('.pt', '_bg.npy'), bg)
    json.dump(r, open(sys.argv[1].replace('.pt', '_real.json'), 'w'))
    print(json.dumps({k: v for k, v in r.items() if not k.startswith('scan') and k not in ('raw_glitch', 'raw_nogate', 'bg_hist')}))
    print('raw+gated', r['raw_glitch']['score'], r['raw_glitch']['mc'], ' raw ungated', r['raw_nogate']['score'])
