"""Rungs on top of the released network (added 15 Nov 2026): post-processing of the forecast outputs, scored with
the ratchet's own objective (the rebuilt frozen set p2r) and dev gate, from the shipped weights (numpy; no torch checkpoint).

Variants (applied to the network's outputs for a window truncated after N s):
  r5          the release, unchanged
  r7_floor    nadir forecast = min(network nadir, lowest value already seen in 0..N s): the nadir of 0-60 s cannot sit
              above a value the trace has already reached
  r7_held     settling forecast = value at N held (mean of the last 3 s seen): the persistence baseline as a rung
  r7_inertN10 inertia read from the first 10 s even when more of the trace is available (N > 10 -> N = 10)

    python rungs_pp.py check            # r5 on the rebuilt set and dev, next to the logged r5 rung
    python rungs_pp.py add <variant>    # score, decide against the incumbent with ratchet.decide, append to ratchet.json
    python rungs_pp.py dev <variant>    # dev table only (no decision)

The frozen set p2s was rebuilt on 15 Nov 2026 from the same scripts and seeds (bank.py val_p2 p2 3000 98 0.6;
valset.py p2s val_p2 mean): 1,387 events as logged, saved as val_p2r.npz. It is not bit-identical: the release (r5)
scores 0.5593 on it against 0.5553 logged on p2s (inertia skill 0.364 vs 0.336; the other skills within 0.003), while
the dev numbers reproduce exactly. So p2r is a new frozen set, and r5 is re-scored on it as the incumbent."""
import sys, json
from common import *
import measure as M, gridpulse_np as GPN, dataset as DS
import griddata as gd
from _paths import MODEL

REL = dict(np.load(f'{MODEL}/gridpulse_weights.npz'))
REL['widen'] = np.ones_like(REL['widen'])          # the ratchet scores raw sigmas, as on the torch checkpoints
LOG = f'{SRC}/ratchet.json'
NAMES = DS.TARGETS
DEV_LOSS = {'GB_20190809T155233': 1131.0, 'GB_20190701T072734': 1000.0, 'GB_20190711T075329': 1153.0,
            'GB_20191203T200030': 854.0}       # evaluate.DEV_LOSS (evaluate.py imports torch)
DEV_DEMAND = {'GB_20190809T155233': 33664.0, 'GB_20190701T072734': 32737.0, 'GB_20190711T075329': 32998.0,
              'GB_20191203T200030': 43733.0}


def predict(variant, w, N, is_gb, f0, demand):
    w = np.asarray(w, np.float64); N = np.asarray(N); f0 = np.asarray(f0, np.float64)
    mu, sg = GPN.forward(REL, w, N, is_gb, f0, demand)
    sc = (50.0 / f0) * 10                                       # target units of nadir / settle (0.1 Hz at 50 Hz -> 1)
    seen = np.array([w[i, M.PRE:M.PRE + int(N[i]) + 1].min() for i in range(len(w))]) * sc
    held = np.array([w[i, M.PRE + int(N[i]) - 2:M.PRE + int(N[i]) + 1].mean() for i in range(len(w))]) * sc
    if variant == 'r7_floor':
        mu[:, 5] = np.minimum(mu[:, 5], seen)
    elif variant == 'r7_held':
        mu[:, 6] = held
    elif variant == 'r7_inertN10':
        m10, _ = GPN.forward(REL, w, np.minimum(N, 10), is_gb, f0, demand)
        mu[:, 1] = np.where(N > 10, m10[:, 1], mu[:, 1])
    elif variant != 'r5':
        raise SystemExit(f'unknown variant {variant}')
    return mu, sg


def val_scores(variant, valname='p2r'):
    """evaluate.val_scores with `predict` in place of the torch checkpoint."""
    z = np.load(f'{DATA}/val_{valname}.npz', allow_pickle=True)
    gb = (z['system'] == 'GB').astype(np.float32); out = {}
    for N in (5, 10, 60):
        mu, sg = predict(variant, z['w'], np.full(len(gb), N), gb, z['f0'], z['demand'])
        y = z['y']
        mae = np.abs(mu - y).mean(0); mae0 = np.abs(np.median(y, 0) - y).mean(0)
        cov = (np.abs(mu - y) <= sg).mean(0)
        out[N] = {n: dict(mae=float(mae[i]), skill=float(max(0, 1 - mae[i] / mae0[i])), cov1=float(cov[i]))
                  for i, n in enumerate(NAMES)}
        for sysn in ('GB', 'ERCO'):
            s = z['system'] == sysn
            out[N][sysn] = {n: float(max(0, 1 - np.abs(mu[s, i] - y[s, i]).mean() / np.abs(np.median(y[s, i]) - y[s, i]).mean()))
                            for i, n in enumerate(NAMES)}
    obj = np.mean([out[60]['log_loss']['skill'], out[60]['log_E']['skill'], out[60]['log_beta']['skill'],
                   out[10]['nadir']['skill'], out[10]['settle']['skill']])
    return dict(objective=float(obj), by_N=out)


def real_gb(eid):
    meta, t, f = gd.event_trace(eid)
    off = int(round((pd.Timestamp(meta['event_t0_utc']) - pd.Timestamp(meta['start_utc'])).total_seconds()))
    x = (f - 50.0)[None, :].astype(np.float64)
    if np.isnan(x).any():
        x = pd.Series(x[0]).interpolate(limit_direction='both').values[None, :]
    return M.window(x, np.array([int(np.where(np.round(t) == off)[0][0])]))


def dev_gb(variant):
    """evaluate.dev_gb with `predict`: errors on the 4 GB dev events."""
    ev = gd.events('GB', split='dev').set_index('event_id'); rows = []
    for eid in GB_DEV:
        r = real_gb(eid)
        for N in (10, 60):
            mu, sg = predict(variant, r['w'], np.array([N]), np.ones(1), np.full(1, 50.0), np.array([DEV_DEMAND[eid]]))
            rows.append(dict(event=eid, N=N, loss=float(np.exp(mu[0, 0])), loss_true=DEV_LOSS[eid],
                             E=float(np.exp(mu[0, 1]) / 1e3), E_true=float(ev.loc[eid].inertia_outturn_GVAs),
                             nadir=float(mu[0, 5] / 10), nadir_true=float(r['nadir'][0]),
                             settle=float(mu[0, 6] / 10), settle_true=float(r['settle'][0]),
                             seen_min=float(r['w'][0, M.PRE:M.PRE + N + 1].min()),
                             held=float(r['w'][0, M.PRE + N - 2:M.PRE + N + 1].mean()),
                             t_nadir=float(ev.loc[eid].t_nadir_s)))
    d = pd.DataFrame(rows); a = d[d.N == 60]; b = d[d.N == 10]
    summ = dict(loss_mdape=float(np.median(np.abs(a.loss / a.loss_true - 1))),
                E_mdape=float(np.median(np.abs(a.E / a.E_true - 1))),
                nadir10_mae=float(np.abs(b.nadir - b.nadir_true).mean()),
                settle10_mae=float(np.abs(b.settle - b.settle_true).mean()))
    return summ, d


def skills(v):
    b = v['by_N']
    return {'loss@60': b[60]['log_loss']['skill'], 'E@60': b[60]['log_E']['skill'], 'beta@60': b[60]['log_beta']['skill'],
            'head@60': b[60]['log_head']['skill'], 'ffr@60': b[60]['log_ffr']['skill'],
            'nadir@5': b[5]['nadir']['skill'], 'nadir@10': b[10]['nadir']['skill'], 'settle@10': b[10]['settle']['skill'],
            'loss@10': b[10]['log_loss']['skill'], 'E@10': b[10]['log_E']['skill'],
            'GB_E@60': b[60]['GB']['log_E'], 'GB_loss@60': b[60]['GB']['log_loss'], 'cov_loss@60': b[60]['log_loss']['cov1'],
            'cov_E@60': b[60]['log_E']['cov1']}


def decide(rec, inc):   # ratchet.decide, verbatim rule (keep if +0.01 on the objective AND dev <= +10 %)
    KEYS = ('loss_mdape', 'E_mdape', 'nadir10_mae', 'settle10_mae')
    s_ok = rec['objective'] > inc['objective'] + 0.01
    rel = float(np.mean([rec['dev'][k] / max(inc['dev'][k], 1e-9) for k in KEYS])); d_ok = rel <= 1.10
    rec['dev_rel'] = rel; rec['accepted'] = bool(s_ok and d_ok); rec['vs'] = inc['rung']
    rec['why'] = (f"score {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') +
                  f"; dev errors {100 * (rel - 1):+.0f}% vs the incumbent" + ('' if d_ok else ' (gate: <= +10%)'))


if __name__ == '__main__':
    cmd, var = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'r5')
    if cmd == 'check':
        v = val_scores('r5'); d, tab = dev_gb('r5')
        old = next(r for r in json.load(open(LOG)) if r['rung'] == 'r5')
        print('objective rebuilt p2s', round(v['objective'], 4), '| logged', round(old['objective'], 4))
        print('dev rebuilt', {k: round(x, 4) for k, x in d.items()}, '\ndev logged ', {k: round(x, 4) for k, x in old['dev'].items()})
        print({k: round(x, 3) for k, x in skills(v).items()})
    elif cmd == 'dev':
        d, tab = dev_gb(var); print(d); print(tab.round(4).to_string())
    elif cmd == 'add':
        note = sys.argv[3]
        # incumbent = the release (r5), re-scored on the rebuilt frozen set p2r from the shipped weights; earlier
        # torch checkpoints (r1-r4, r6) are not in the repository, so the ladder before r5 is not re-scored on p2r
        log = json.load(open(LOG))
        inc = dict(rung='r5 (release, re-scored on p2r)', objective=val_scores('r5')['objective'], dev=dev_gb('r5')[0])
        v = val_scores(var); d, tab = dev_gb(var)
        rec = dict(rung=var, note=note, val='p2r', objective=v['objective'], skills=skills(v), dev=d,
                   dev_table=tab.round(4).to_dict('records'), date='2026-11-15', scored_from='gridpulse_weights.npz')
        decide(rec, inc); log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
        print(f"{var}: objective {rec['objective']:.4f} | " + ' '.join(f'{k} {x:.3f}' for k, x in rec['skills'].items()))
        print('  dev', {k: round(x, 4) for k, x in d.items()}, '->', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])
