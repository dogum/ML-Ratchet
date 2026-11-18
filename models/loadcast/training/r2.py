"""Round 2 rungs (17 Nov 2026; plan in NOTEBOOK.md): bank twins for r8, the r7 magnitude head, and 3-seed scoring.

    python r2.py twins <bank> [procs]          # metadata-only dev-calibrated twin of every bank building -> twin_XXXX.npz
    python r2.py mag <run> <bank> <seed>       # r7 magnitude head: log target-year mean from the 38 metadata features
    python r2.py score <frozen> <out> <spec> [<spec> ...]
        spec = name=kind:cold[,cold2]:warm, kind in cold | twin | magshape | twinref; ckpt names under WORK/ckpt,
        or R3 for the released numpy weights. -> WORK/r2/<out>.json (+ per-unit CSVs)

The twin is the one `twin_anchor.py` and the exam's calibrated-twin baseline use (twin_knobs.json 'med'): for a
simulated building it is built from the metadata the network sees (use, area, year built if shown, meters, site), not
from the generator spec that made its truth. Physics 1.0 (the default) unless $BUILDINGSIM_PHYSICS says otherwise.
"""
import sys, os, json, glob, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, evaluate as E, lcnet as N
_pred, _load = N.predict, N.load
import rescore_seen as R          # noqa: E402  (patches N.predict / N.load: undone on the next line)
N.predict, N.load = _pred, _load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{C.ROOT}/r2'


# ------------------------------------------------------------------ bank twins (r8)
def _site_info():
    m = C.bd.bdg2_meta()
    g = m.groupby('site_id')
    return {s: dict(lat=float(g.lat.median()[s]), lon=float(g.lon.median()[s]), cz=g.climate_zone.agg(lambda x: x.mode()[0])[s])
            for s in g.groups}


def _twin_part(args):
    fn, TK = args
    import lcsim as L
    out = fn.replace('part_', 'twin_')
    if os.path.exists(out): return out, 0
    z = np.load(fn); recs = json.loads(str(z['meta'])); S = _site_info(); Y = {}
    for r in recs:
        si = S[r['site']]
        spec = dict(primary_use=r['use'], floor_area_m2=r['area'], year_built=r['year_built'] if r['yb_shown'] else None,
                    climate_zone=si['cz'], lat=si['lat'], lon=si['lon'])
        k = R.knobs_for(TK, r['use'], r['meters'], 'med')
        raw, _ = L.simulate_site(L.make_spec(spec, r['meters'], k), r['site'])
        y = L.post(raw, k, r['area'])
        for m in r['meters']:
            Y[f"{r['i']}_{m}"] = (y[m] / (r['area'] * C.IREF[m] / 1000.0)).astype(np.float32)
    np.savez(out + '.tmp.npz', **Y); os.replace(out + '.tmp.npz', out)
    return out, len(recs)


def twins(bank, procs=3):
    TK = json.load(open(f'{HERE}/twin_knobs.json'))
    files = sorted(glob.glob(f'{C.ROOT}/data/bank_{bank}/part_*.npz'))
    t0 = time.time()
    with mp.Pool(procs, maxtasksperchild=2) as p:
        for out, n in p.imap_unordered(_twin_part, [(f, TK) for f in files]):
            print(out, n, round(time.time() - t0), 's', flush=True)


# ------------------------------------------------------------------ r7 magnitude head
def mag(run, bank, seed, epochs=30, width=192, depth=3, reps=64, jitter=1.0, dropout=0.1, wd=1e-3):
    import torch
    import train as T
    torch.set_num_threads(1)
    seed = int(seed); t0 = time.time()
    units = D.load_bank(bank, 'real', seed=seed)
    for u in units: D.prepare(u, ('full',))
    units = [u for u in units if np.isfinite(u['y'][u['tidx']]).sum() >= 200]   # as train.py: >= 200 valid target hours
    St = np.array([u['static'] for u in units], np.float32)
    Yt = np.array([np.log(max(float(np.nanmean(u['y'][u['tidx']])), 1e-3)) for u in units], np.float32)
    print(len(units), 'units', round(time.time() - t0), 's', flush=True)
    net = N.MLP(St.shape[1], width, depth, dropout)
    net.mu_x.copy_(torch.from_numpy(St.mean(0))); net.sd_x.copy_(torch.from_numpy(St.std(0) + 1e-3))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=wd)
    for ep in range(epochs):
        rng = np.random.default_rng([seed, ep, 77])
        X = T.jitter_static(np.repeat(St, reps, 0).copy(), rng, jitter); Y = np.repeat(Yt, reps)
        perm = rng.permutation(len(X)); lr = 2e-3 * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        for g in opt.param_groups: g['lr'] = lr
        net.train(); Xt, Ytt = torch.from_numpy(X), torch.from_numpy(Y); zero = torch.zeros(len(X))
        tl = 0.0; nb = 0
        for i in range(0, len(X), 1024):
            j = torch.from_numpy(perm[i:i + 1024])
            mu, ls = net(Xt[j], zero[j]); y = Ytt[j]
            loss = ((y - mu) ** 2).mean() + 0.1 * (((y - mu.detach()) ** 2) * torch.exp(-2 * ls) + 2 * ls).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tl += float(((y - mu) ** 2).mean()); nb += 1
        print(f'{run} mag epoch {ep} mse(log) {tl / nb:.4f}', flush=True)
    net.eval()
    torch.save(dict(state=net.state_dict(), nin=St.shape[1], width=width, depth=depth, mode='mag', bank=bank, noise='real',
                    epochs=epochs, dropout=dropout, jitter=jitter, wd=wd, seed=seed), f'{C.ROOT}/ckpt/{run}_mag.pt')
    print('saved', run, round(time.time() - t0), 's', flush=True)


# ------------------------------------------------------------------ scoring
def _net(name, mode):
    if name == 'R3': return R.np_nets()[mode]
    return N.load(f'{C.ROOT}/ckpt/{name}.pt')[0]


def _predict(net, X, base):
    if isinstance(net, R.NPNet): return R.np_predict(net, X, base)
    return N.predict(net, X, base)


def cold_predict(kind, nets, u):
    t = u['tidx']; y = u['y'][t].astype(float)
    if kind == 'cold':
        X, b, _, _ = D.rows(u, t, 'cold'); mu, sd = _predict(nets[0], X, b)
    elif kind == 'twin':
        X, b, _, _ = D.rows(u, t, 'twin'); mu, sd = _predict(nets[0], X, b)
    elif kind == 'magshape':
        lm, sl = _predict(nets[0], u['static'][None].astype(np.float32), np.zeros(1, np.float32))
        X, b, _, _ = D.rows(u, t, 'shape'); ms, ss = _predict(nets[1], X, b)
        M = float(np.exp(lm[0])); s = float(sl[0]); ms = np.clip(ms, 0, None)
        mu, sd = M * ms, M * np.sqrt(ss ** 2 + (ms * s) ** 2)
    elif kind == 'twinref':
        mu = np.nan_to_num(u['tw'][t].astype(float)); sd = np.full(len(t), np.nan)
    return np.clip(mu, 0, None), sd, y


def unit_metrics(kind, cnets, wnet, u):
    r = dict(id=u['id'], meter=u['meter'])
    mu, sd, y = cold_predict(kind, cnets, u)
    cv, nm, _ = C.g14(y, mu); r.update(cold_cv=cv, cold_nmbe=nm, cold_cov=E.cover(y, mu, sd) if np.isfinite(sd).any() else np.nan)
    X, b, yw, S = D.rows(u, u['tidx'], 'warm', 'full'); m, s = _predict(wnet, X, b)
    mu, sd = np.clip(m, 0, None) * S, s * S
    cv, nm, _ = C.g14(yw * S, mu); r.update(warm_cv=cv, warm_nmbe=nm, warm_cov=E.cover(yw * S, mu, sd))
    return r


def load_sets(frozen):
    t0 = time.time()
    fz = D.load_bank(frozen, 'real', seed=12345)
    if glob.glob(f'{C.ROOT}/data/bank_{frozen}/twin_*.npz'): D.attach_bank_twins(fz, frozen)
    for u in fz: D.prepare(u, ('full',))
    ref = E.ref_cvs(fz)
    dev = D.real_units('dev')
    z = np.load(f'{C.ROOT}/exam/twins.npz')
    for u in dev:
        D.prepare(u, ('full',))
        k = f"med|{u['bid']}|{u['meter']}"
        u['tw'] = (z[k] / (u['area'] * C.IREF[u['meter']] / 1000.0)).astype(np.float32)
    print('sets ready', len(fz), len(dev), round(time.time() - t0), 's', flush=True)
    return fz, ref, dev


def score(frozen, out, *specs):
    os.makedirs(OUT, exist_ok=True)
    fz, ref, dev = load_sets(frozen)
    res = {}
    fn = f'{OUT}/{out}.json'
    if os.path.exists(fn): res = json.load(open(fn))
    for sp in specs:
        name, _, rest = sp.partition('=')
        kind, cold, warm = rest.split(':')
        cn = cold.split(',') if cold else []
        modes = {'cold': ['cold'], 'twin': ['twin'], 'magshape': ['mag', 'shape'], 'twinref': []}[kind]
        cnets = [_net(c, m if c == 'R3' else None) for c, m in zip(cn, modes)]
        wnet = _net(warm, 'warm')
        t0 = time.time()
        df = pd.DataFrame([unit_metrics(kind, cnets, wnet, u) for u in fz])
        dv = pd.DataFrame([unit_metrics(kind, cnets, wnet, u) for u in dev])
        sk = E.skill(df, ref)
        res[name] = dict(spec=sp, frozen=frozen, objective=sk['objective'], cold_skill=sk['cold_skill'], warm_skill=sk['warm_skill'],
                         per_meter=sk['per_meter'], dev_score=E.dev_score(dv), dev_cold=float(dv.cold_cv.median()),
                         dev_warm=float(dv.warm_cv.median()),
                         dev_by_meter={m: dict(n=len(g), cold=float(g.cold_cv.median()), warm=float(g.warm_cv.median()),
                                               cold_nmbe=float(g.cold_nmbe.median())) for m, g in dv.groupby('meter')})
        df.to_csv(f'{OUT}/{out}_{name}_frozen.csv', index=False); dv.to_csv(f'{OUT}/{out}_{name}_dev.csv', index=False)
        C.jdump(res, fn)
        r = res[name]
        print(f"{name:14s} objective {r['objective']:+.4f} (cold {r['cold_skill']:+.3f}, warm {r['warm_skill']:+.3f})  dev score "
              f"{r['dev_score']:.2f} (cold {r['dev_cold']:.1f}, warm {r['dev_warm']:.1f})  {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'twins': twins(a[1], int(a[2]) if len(a) > 2 else 3)
    elif a[0] == 'mag': mag(a[1], a[2], a[3])
    elif a[0] == 'score': score(a[1], a[2], *a[3:])
