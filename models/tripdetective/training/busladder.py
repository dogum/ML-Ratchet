"""TripDetective bus ladder (Phase D): mass of a city bus from its power log, judged against ZTBus passengers.
Objective: mean skill on the frozen simulated bus set (busbank.py) over log mass, log CdA and log Crr, single 1,024 s
window and pooled over a bus's windows (skill = 1 - MAE / MAE of the median guess).
Dev gate (ZTBus dev missions, 'gate' half; truth 19,000 kg + 70 kg x mean passengers per leg): the level-corrected
mean absolute mass error per leg (estimate x one factor, median truth / estimate on these dev legs) may not get more
than 5 % worse, and the slope of estimated mass on passenger mass may not drop by more than 0.1. The exam (1,115 exam
missions) is never computed here.
Rule changed in the open on 14 Nov 2026, after rungs pb1/pb2 and before any other: the first rule gated on the raw
error. But the level of the truth is a convention (RoadSim's ZTBus twin uses 19 t curb; the bus's real curb is not
published in the data), and on 40 dev route-half legs both the logged power and the logged traction force regress to
0.67-0.73 x (19 t + passengers). So the gate judges how well the estimate follows passengers, with the level fitted on
dev; the raw bias is still logged.
usage: python busladder.py <ckpt> <val_tag> "<note>"   |   python busladder.py --rescore <val_tag>"""
from common import *
import evaluate as E, simbank, busbank, features as F
from model import REG, from_reg, reg_targets

LOG = os.environ.get('TD_BUS_RATCHET', f'{ROOT}/ratchet_bus.json')
S = busbank.BUS_SCALE


def sim_scores(net, bank):
    meta = {m['vid']: m for m in bank['meta']}; vids = bank['veh']; cars = sorted(meta)
    inp = E.bank_inputs(bank, np.arange(len(vids)))
    Y = np.stack([reg_targets(meta[v]) for v in cars]); cmap = {v: i for i, v in enumerate(cars)}
    res = {}
    for mode in ['single', 'pooled']:
        if mode == 'single':
            P = E.predict(net, inp, np.arange(len(vids))); y = Y[[cmap[v] for v in vids]]
        else:
            P = E.predict(net, inp, np.array([cmap[v] for v in vids])); y = Y
        med = np.nanmedian(Y, 0); sk = {}; r = {}
        for i, k in enumerate(REG[:4]):
            ok = np.isfinite(y[:, i])
            if ok.sum() < 10: continue
            e = np.abs(P['mu'][ok, i] - y[ok, i]); e0 = np.abs(med[i] - y[ok, i])
            sk[k] = max(0.0, 1 - e.mean() / e0.mean()); r[f'{k}_mdape'] = float(np.median(np.abs(np.exp(P['mu'][ok, i] - y[ok, i]) - 1)))
            r[f'{k}_in1s'] = float((e < P['sig'][ok, i]).mean())
        r['skill'] = {k: round(float(v), 4) for k, v in sk.items()}; r['objective'] = float(np.mean(list(sk.values())))
        res[mode] = r
    res['objective'] = 0.5 * (res['single']['objective'] + res['pooled']['objective'])
    return res


def real_predict(net, rb):
    o = rb['offs']
    inp = [F.net_input(rb['raw'][o[j]:o[j + 1]].T.astype(np.float32), rb['flags'][j]) for j in range(len(rb['trip']))]
    legs = list(dict.fromkeys(rb['trip'].tolist())); lmap = {t: i for i, t in enumerate(legs)}
    P = E.predict(net, inp, np.array([lmap[t] for t in rb['trip']]))
    return legs, from_reg(0, P['mu'][:, 0]) * S, P['sig'][:, 0]


def leg_metrics(mhat, meta, sig=None, sigk=1.0):
    pax = np.array([m['pax'] for m in meta]); truth = np.array([m['mass_kg'] for m in meta]); ok = np.isfinite(truth) & np.isfinite(mhat)
    e = mhat[ok] - truth[ok]
    A = np.stack([np.ones(ok.sum()), busbank.PAX_KG * pax[ok]], 1)
    slope = float(np.linalg.lstsq(A, mhat[ok], rcond=None)[0][1]) if np.std(mhat[ok]) > 0 else 0.0
    out = dict(n=int(ok.sum()), mae=float(np.mean(np.abs(e))), med_ae=float(np.median(np.abs(e))), bias=float(np.median(e)),
               mdape=float(np.median(np.abs(e) / truth[ok])), slope=slope,
               r=float(np.corrcoef(mhat[ok], pax[ok])[0, 1]) if np.std(mhat[ok]) > 0 else 0.0)
    if sig is not None:
        z = np.log(mhat[ok] / truth[ok]) / (sig[ok] * sigk); out['in1s'] = float(np.mean(np.abs(z) < 1)); out['in2s'] = float(np.mean(np.abs(z) < 2))
    return out


def dev_scores(net):
    rb = busbank.load_real('dev')
    legs, m, s = real_predict(net, rb)
    meta = {x['trip_id']: x for x in rb['meta']}
    M = [meta[t] for t in legs]
    out = leg_metrics(m, M, s)
    truth = np.array([x['mass_kg'] for x in M]); ok = np.isfinite(truth)
    k = float(np.median(truth[ok] / m[ok])); lc = leg_metrics(m * k, M)
    out.update(level_factor=k, mae_lc=lc['mae'], med_ae_lc=lc['med_ae'])
    return out


def decide(rec, inc):
    if inc is None:
        rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01
    rel = rec['dev']['mae_lc'] / inc['dev']['mae_lc']; sl_ok = rec['dev']['slope'] >= inc['dev']['slope'] - 0.1
    rec['dev_rel'] = rel; rec['accepted'] = bool(s_ok and rel <= 1.05 and sl_ok); rec['vs'] = inc['ckpt']
    rec['why'] = (f"sim objective {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') +
                  f"; dev level-corrected mass error {100 * (rel - 1):+.0f}%" + ('' if rel <= 1.05 else ' (gate fails)') +
                  f"; passenger slope {rec['dev']['slope']:.2f} vs {inc['dev']['slope']:.2f}" + ('' if sl_ok else ' (gate fails)'))


def score(ckpt, val_tag):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    return sim_scores(net, simbank.load_bank(val_tag)), dev_scores(net)


def main(ckpt, val_tag, note):
    v, dv = score(ckpt, val_tag)
    log = jload(LOG, []); inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = dict(ckpt=ckpt, note=note, val_tag=val_tag, val=v, objective=v['objective'], dev=dv)
    decide(rec, inc); log.append(rec); jsave(log, LOG)
    print(f"{ckpt}: objective {v['objective']:.3f} single {v['single']['skill']} pooled {v['pooled']['skill']}")
    print('  dev', {k: round(x, 3) for k, x in dv.items()})
    print(' ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])


def rescore(val_tag):
    log = jload(LOG, []); inc = None
    for rec in log:
        v, dv = score(rec['ckpt'], val_tag)
        rec.update(val=v, objective=v['objective'], dev=dv, val_tag=val_tag)
        decide(rec, inc); inc = rec if rec['accepted'] else inc
        print(rec['ckpt'], round(rec['objective'], 3), rec['accepted'], rec['why'], flush=True)
    jsave(log, LOG)


if __name__ == '__main__':
    if sys.argv[1] == '--rescore': rescore(sys.argv[2])
    else: main(sys.argv[1], sys.argv[2], sys.argv[3])
