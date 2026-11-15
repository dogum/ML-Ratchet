"""HANDOFF P0.5 check: does GridSim physics 1.0 vs 1.1 (outage defaults) change what PlantDetective's ratchet chose?

Two bank v2 copies are built with the release's own scripts, one per physics version, in separate work roots:
    GRIDSIM_PHYSICS=<v> ML_RATCHET_WORK=<root_v> python simbank_gs.py <seed> 1      (seeds 0-4, the build's 5 systems)
    GRIDSIM_PHYSICS=<v> ML_RATCHET_WORK=<root_v> python bank_dp.py v2 <first> <n>   (shards 0-109, 16 units each)
    GRIDSIM_PHYSICS=<v> ML_RATCHET_WORK=<root_v> python bank_dp.py v2 gs
    GRIDSIM_PHYSICS=<v> ML_RATCHET_WORK=<root_v> python trainset.py v2
This script then compares the two (prices, targets, features) and scores the shipped R4 ensemble and its three members
on both frozen validation sets.  Only simulated data; no dev or exam unit is read.

Results of the 14 Nov 2026 run: checks/outage_physics_1_1.json (with the dev gate under both versions).

usage: python outage_check.py <root_v1_0> <root_v1_1> <models_dir> <out.json>
"""
from common import *
import json, glob
import numpy as np
from evaluate import skill
from model import np_forward, ens_forward
from pdcore import REG

GS_JOBS = range(5)
N_SHARDS = 110


def gs_bg_shards():
    """DP shards whose price background is a GridSim job (bank_dp.shard's first two draws; 5 job files present)."""
    out = {}
    for sid in range(N_SHARDS):
        rng = np.random.default_rng(10_000 + sid)
        if rng.random() < 0.3:
            out[f'shard_{sid:04d}'] = int(rng.integers(len(GS_JOBS)))
    return out


def group(keys, gsb):
    sh = np.array([k.split(':')[0] for k in keys])
    g = np.where(np.char.startswith(sh, 'shard_gs_'), 'gs_unit', np.where(np.isin(sh, list(gsb)), 'dp_on_gs_price',
                                                                           'dp_on_ercot_price'))
    return g


def natural(Y):
    """Transformed targets (trainset.tgt_vector) back to natural units."""
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa
    d = dict(pmin_frac=sig(Y[:, 0]), ramp_up=np.exp(Y[:, 1]), ramp_dn=np.exp(Y[:, 2]), starts_yr=np.expm1(Y[:, 3]),
             min_up=np.exp(Y[:, 4]), med_up=np.exp(Y[:, 5]), min_dn=np.exp(Y[:, 6]), hot=sig(Y[:, 7]),
             warm=sig(Y[:, 8]), cold=sig(Y[:, 9]), hr_full=np.exp(Y[:, 10]), hr_part_ratio=np.exp(Y[:, 11]),
             cf=sig(Y[:, 12]))
    return d


def ks(a, b):
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if len(a) < 3 or len(b) < 3:
        return np.nan
    x = np.r_[a, b]
    return float(np.max(np.abs(np.searchsorted(a, x, 'right') / len(a) - np.searchsorted(b, x, 'right') / len(b))))


def prices(root):
    out = {}
    for j in GS_JOBS:
        z = np.load(f'{root}/plantdetective/data/bank_gs/job_{j:05d}.npz')
        m = json.loads(str(z['meta']))
        p = z['price'].astype(float)
        on = z['u'] > 0.5
        out[j] = dict(ba=m['ba'], year=m['year'], peak=round(m['peak']), units=len(m['specs']),
                      mean=float(p.mean()), median=float(np.median(p)), p95=float(np.percentile(p, 95)),
                      h_over_1000=int((p > 1000).sum()), starts=int(z['su'].sum()), on_share=float(on.mean()),
                      relaxed=m['n_relaxed'], wall_s=round(m['wall']))
    return out


def per_unit(z, gsb):
    K = z['K']; _, first = np.unique(K, return_index=True)
    return K[first], z['Y'][first], z['C'][first], group(K[first], gsb), z['V'][first]


def main(r0, r1, mdir, out):
    gsb = gs_bg_shards()
    res = dict(gs_bg_shards=gsb, prices={'1.0': prices(r0), '1.1': prices(r1)})
    S = {v: np.load(f'{r}/plantdetective/data/set_v2.npz') for v, r in (('1.0', r0), ('1.1', r1))}
    # composition
    comp = {}
    for v, z in S.items():
        K, Y, C, G, V = per_unit(z, gsb)
        g = group(z['K'], gsb)
        comp[v] = dict(units=int(len(K)), windows=int(len(z['K'])), val_units=int(V.sum()), val_windows=int(z['V'].sum()),
                       **{f'units_{a}': int((G == a).sum()) for a in np.unique(G)},
                       **{f'val_units_{a}': int(((G == a) & V).sum()) for a in np.unique(G)},
                       **{f'windows_{a}': int((g == a).sum()) for a in np.unique(g)},
                       **{f'val_windows_{a}': int(((g == a) & z['V']).sum()) for a in np.unique(g)})
    res['composition'] = comp
    # target distributions per group (one row per unit)
    td = {}
    U = {v: per_unit(z, gsb) for v, z in S.items()}
    for a in ('gs_unit', 'dp_on_gs_price', 'dp_on_ercot_price'):
        d = {}
        n0 = natural(U['1.0'][1][U['1.0'][3] == a]); n1 = natural(U['1.1'][1][U['1.1'][3] == a])
        for k in n0:
            d[k] = dict(median_1_0=float(np.nanmedian(n0[k])), median_1_1=float(np.nanmedian(n1[k])),
                        mean_1_0=float(np.nanmean(n0[k])), mean_1_1=float(np.nanmean(n1[k])), ks=ks(n0[k], n1[k]))
        # paired, same unit key (GridSim units keep their parameters; DP units only until a shard's draws diverge)
        k0 = dict(zip(U['1.0'][0], U['1.0'][1])); k1 = dict(zip(U['1.1'][0], U['1.1'][1]))
        both = [k for k in k0 if k in k1 and group([k], gsb)[0] == a]
        if both:
            A = natural(np.array([k0[k] for k in both])); B = natural(np.array([k1[k] for k in both]))
            for k in A:
                m = np.isfinite(A[k]) & np.isfinite(B[k])
                d[k]['paired_identical_share'] = float(np.mean(np.isclose(A[k][m], B[k][m], rtol=1e-6)))
                d[k]['paired_median_abs_rel_change'] = float(np.median(np.abs(B[k][m] / np.where(A[k][m] == 0, np.nan, A[k][m]) - 1)[np.isfinite(A[k][m]) & (A[k][m] != 0)])) if m.any() else None
        d['n_units'] = [int((U['1.0'][3] == a).sum()), int((U['1.1'][3] == a).sum())]
        td[a] = d
    res['targets'] = td
    # input features: standardised mean difference per feature, per group (all windows)
    sd = np.nanstd(S['1.0']['X'], 0) + 1e-6
    fd = {}
    for a in ('gs_unit', 'dp_on_gs_price', 'dp_on_ercot_price'):
        X0 = S['1.0']['X'][group(S['1.0']['K'], gsb) == a]; X1 = S['1.1']['X'][group(S['1.1']['K'], gsb) == a]
        smd = (np.nanmean(X1, 0) - np.nanmean(X0, 0)) / sd
        top = np.argsort(-np.abs(smd))[:8]
        fd[a] = dict(max_abs_smd=float(np.max(np.abs(smd))), mean_abs_smd=float(np.mean(np.abs(smd))),
                     top=[(int(i), float(smd[i])) for i in top])
    res['features'] = fd
    # scores of the shipped networks on both frozen sets
    Ws = [dict(np.load(f)) for f in sorted(glob.glob(f'{mdir}/R4_ens_s[0-9].npz'))]
    nets = {'R4_ens': lambda X: ens_forward(Ws, X), **{f'member_s{i}': (lambda W: (lambda X: np_forward(W, X)))(W)
                                                         for i, W in enumerate(Ws)}}
    sc = {}
    for v, z in S.items():
        Vm = z['V']; tr = ~Vm
        dumb = np.array([[np.nanmedian(z['Y'][tr & (z['C'] == c), j]) for j in range(z['Y'].shape[1])] for c in range(4)])
        g = group(z['K'], gsb)
        sv = {}
        for name, f in nets.items():
            p, m, s = f(z['X'][Vm])
            tot, acc, sk = skill(p, m, z['Y'][Vm], z['C'][Vm], dumb)
            sv[name] = dict(val=tot, val_acc=acc, per_target=dict(zip(REG, map(float, sk))))
            for a in np.unique(g):
                mm = g[Vm] == a
                sv[name][f'val_{a}'] = skill(p[mm], m[mm], z['Y'][Vm][mm], z['C'][Vm][mm], dumb)[0]
        sc[v] = sv
    res['scores'] = sc
    res['deltas'] = {v: dict(R4_minus_s0=sc[v]['R4_ens']['val'] - sc[v]['member_s0']['val'],
                             R4_minus_best_member=sc[v]['R4_ens']['val'] - max(sc[v][f'member_s{i}']['val'] for i in range(3)))
                     for v in sc}
    json.dump(res, open(out, 'w'), indent=1, default=float)
    print(json.dumps(dict(composition=comp, scores={v: {n: round(s['val'], 4) for n, s in d.items()} for v, d in sc.items()},
                          deltas=res['deltas'], prices=res['prices']), indent=1, default=float))


if __name__ == '__main__':
    main(*sys.argv[1:5])
