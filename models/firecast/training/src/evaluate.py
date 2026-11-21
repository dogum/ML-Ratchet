"""Scoring helpers: sim frozen validation, real DEV gate, B's baselines on the same fire-days (sanity check).
CLI: python3 evaluate.py <rung> [sim|dev|both] [--land]   -> out/eval_<rung>.json (merged)"""
import pickle
import zlib
import time
from common import *  # noqa
import torch
from scipy import ndimage
import features as FT

torch.set_num_threads(1)
_DEV = None


def dev_days():
    global _DEV
    if _DEV is None:
        _DEV = pickle.load(open(f'{OUT}/dev_days.pkz', 'rb'))
    return _DEV


def dev_day(i):
    return pickle.loads(zlib.decompress(dev_days()[i]))


def load_net(rung):
    from train import build
    st = torch.load(f'{ROOT}/ckpt/{rung}.pt', weights_only=False)
    net = build(st['cfg'])
    net.load_state_dict(st['model'])
    net.eval()
    return net


def logits_full(net, X, s):
    """Fully convolutional inference on any size (padded to a multiple of 16)."""
    C, H, W = X.shape
    Hp, Wp = int(np.ceil(H / 16) * 16), int(np.ceil(W / 16) * 16)
    Xp = np.zeros((C, Hp, Wp), np.float32)
    Xp[:, :H, :W] = X
    with torch.no_grad():
        o = net(torch.from_numpy(Xp)[None], torch.from_numpy(s)[None])[0].numpy()
    return o[:H, :W]


def real_inputs(d, land=False):
    return FT.make_inputs(d['now'], d['prev'], d['fl'], d['pix_pm'], d['pix_am'],
                          d['land'] if (land and d['land'] is not None) else None, d['wx'], d['has_prev'], d['first'])


def score_real(d, prob, thr=0.5):
    fd_ = fd()
    win = fd_.metrics.score_window(d['now'], RES, WIN_KM)
    return fd_.metrics.nextday(d['now'], d['nxt'], prob, CELL_KM2, thr=thr, window=win)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def dev_gate(net, land=False, temp=1.0, bias=0.0, thr=0.5, keep_logits=False):
    rows, logit_store = [], []
    for i in range(len(dev_days())):
        d = dev_day(i)
        X, s = real_inputs(d, land)
        lg = logits_full(net, X, s)
        p = sig(lg / temp + bias)
        m = score_real(d, p, thr)
        m.update(fire_id=d['fire_id'], k=d['k'], area_km2=d['area_km2'])
        rows.append(m)
        if keep_logits:
            win = fd().metrics.score_window(d['now'], RES, WIN_KM) & ~d['now']
            logit_store.append((lg[win].astype(np.float16), (d['nxt'] & ~d['now'])[win]))
    return rows, logit_store


def summarize(rows):
    S = fd().metrics.summarize(rows)
    # mean over fires of per-fire means (secondary)
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[df.new_obs_km2 > 0]
    S['ap_mean_of_fires'] = float(df.groupby('fire_id').ap.mean().mean())
    S['f1_mean_of_fires'] = float(df.groupby('fire_id').f1.mean().mean())
    return S


def sim_val(net, ver, land=False, max_n=None):
    from dataset import Bank
    fd_ = fd()
    b = Bank(ver, 'val')
    n = len(b) if max_n is None else min(max_n, len(b))
    rows = []
    for i in range(n):
        X, s, d = b.full_item(i, land)
        p = sig(logits_full(net, X, s))
        win = fd_.metrics.score_window(d['now'], RES, WIN_KM)
        m = fd_.metrics.nextday(d['now'], d['nxt'], p, CELL_KM2, window=win)
        m.update(fire_id=f'sim{d["fire"]}', kind=d['kind'])
        rows.append(m)
    return rows


def twin_gate(net, ver, land=False):
    """Score the same predictions against the real next-day perimeter and against the twin's (member 0)."""
    tdir = f'{OUT}/twin_{ver}_targets'
    rows_r, rows_t = [], []
    for i in range(len(dev_days())):
        fn = f'{tdir}/{i:04d}.bin'
        if not os.path.exists(fn):
            continue
        d = dev_day(i)
        n = d['now'].size
        tw = np.unpackbits(np.frombuffer(zlib.decompress(open(fn, 'rb').read()), np.uint8))[:n].reshape(d['now'].shape).astype(bool)
        X, s = real_inputs(d, land)
        p = sig(logits_full(net, X, s))
        a = score_real(d, p)
        a.update(fire_id=d['fire_id'], k=d['k'])
        d2 = dict(d, nxt=tw | d['now'])
        b = score_real(d2, p)
        b.update(fire_id=d['fire_id'], k=d['k'])
        rows_r.append(a)
        rows_t.append(b)
    return summarize(rows_r), summarize(rows_t)


def baselines_dev():
    """B's fl_prev / grow_prev / clim recomputed on the cached dev days (checks our use of the metric)."""
    P = json.load(open(f'{PROC}/baselines/spread_params.json'))
    out = {m: [] for m in ('persist', 'clim', 'grow_prev', 'fl_prev')}
    for i in range(len(dev_days())):
        d = dev_day(i)
        now = d['now']
        dist = np.maximum(ndimage.distance_transform_edt(~now, sampling=RES) - RES / 2, 0)
        sb = int(np.digitize(d['area_km2'], P['size_bins_km2']) - 1)
        dcl = P['clim_d_m'][sb]
        dgp = max(d['d_prev_m'], 0.0) if np.isfinite(d['d_prev_m']) else dcl
        sc = lambda dd, r: 1 / (1 + (dd / max(r, 1.0)) ** 4) if r > 0 else np.zeros(now.shape)
        if d['fl'] is not None and d['fl'].any():
            dfl = np.maximum(ndimage.distance_transform_edt(~d['fl'], sampling=RES) - RES / 2, 0)
        else:
            dfl = dist
        for m, s in (('persist', np.zeros(now.shape)), ('clim', sc(dist, dcl)), ('grow_prev', sc(dist, dgp)),
                     ('fl_prev', sc(dfl, dgp))):
            r = score_real(d, s)
            r.update(fire_id=d['fire_id'], k=d['k'])
            out[m].append(r)
    return {m: summarize(v) for m, v in out.items()}, out


def main():
    rung = sys.argv[1]
    what = sys.argv[2] if len(sys.argv) > 2 else 'both'
    fn = f'{OUT}/eval_{rung}.json'
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    t0 = time.time()
    if rung == 'baselines':
        S, rows = baselines_dev()
        jdump(dict(dev=S), fn)
        print(json.dumps(S, indent=1))
        return
    net = load_net(rung)
    st = torch.load(f'{ROOT}/ckpt/{rung}.pt', weights_only=False)
    ver = os.environ.get('VAL_VER') or st['cfg'].get('val_bank', st['cfg']['bank'])   # VAL_VER: re-score on a newer frozen set
    if what in ('sim', 'both'):
        for land in (False, True):
            rows = sim_val(net, ver, land)
            res[f'sim_{ver}_{"L" if land else "F"}'] = summarize(rows)
            print('sim', ver, land, res[f'sim_{ver}_{"L" if land else "F"}'], f'{time.time() - t0:.0f}s', flush=True)
    if what in ('dev', 'both'):
        for land in (False, True):
            rows, _ = dev_gate(net, land)
            res[f'dev_{"L" if land else "F"}'] = summarize(rows)
            print('dev', land, res[f'dev_{"L" if land else "F"}'], f'{time.time() - t0:.0f}s', flush=True)
            pickle.dump(rows, open(f'{OUT}/devrows_{rung}_{"L" if land else "F"}.pkl', 'wb'))
    if what == 'twin':
        ver2 = sys.argv[3]
        sr, stw = twin_gate(net, ver2)
        res[f'twin_{ver2}'] = dict(real=sr, twin=stw)
        print('twin', ver2, 'real', sr, 'twin', stw)
    jdump(res, fn)


if __name__ == '__main__':
    main()
