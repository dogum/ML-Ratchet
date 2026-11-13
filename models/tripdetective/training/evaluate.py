"""Scoring: frozen simulated validation set (single trips and whole-car pools) and real VED dev/exam cars.
Skill = 1 - MAE / MAE of guessing the training median (floored at 0); architecture skill = (acc - majority)/(1 - majority)."""
from common import *
import torch
torch.set_num_threads(1)
import features as F
from model import TDNet, REG, from_reg, LED_SCALE, reg_targets
import realdata as rd


def load_net(path):
    st = torch.load(path, weights_only=False)
    o = st.get('opts', {})
    width = tuple(int(x) for x in o.get('width', '32,48,64,96,128,160').split(','))
    feat = int(o.get('feat', 1)); F.FEAT = feat          # inputs built after loading use this network's feature version
    net = TDNet(width=width, feat=feat); net.load_state_dict(st['net']); net.eval()
    return net


@torch.no_grad()
def predict(net, inputs, groups, bs=64):
    """inputs: list of (x (NIN, L), mask (L,)); groups: car/pool index per input (ints, contiguous 0..G-1).
    Pools are formed by groups; trips are encoded in batches, then pooled."""
    groups = np.asarray(groups); G = int(groups.max()) + 1
    E = []
    for i in range(0, len(inputs), bs):
        xs = inputs[i:i + bs]
        T = int(max(it[1].sum() for it in xs)); T = min(L, max(32, int(np.ceil(T / 32) * 32)))
        x = torch.tensor(np.stack([it[0][:, :T] for it in xs])); m = torch.tensor(np.stack([it[1][:T] for it in xs]))
        sc = torch.tensor(np.stack([it[2] for it in xs]))
        E.append(net.encode(x, m, sc))
    e = torch.cat(E)
    g = torch.tensor(groups)
    cnt = torch.zeros(G).index_add_(0, g, torch.ones(len(g)))
    z = torch.zeros(G, e.shape[1]).index_add_(0, g, e) / cnt[:, None]
    z = net.car(torch.cat([z, torch.log(cnt)[:, None]], 1))
    r = net.h_reg(z)
    led = net.h_led(torch.cat([e, z[g]], 1))
    return dict(arch=torch.softmax(net.h_arch(z), 1).numpy(), strat=torch.softmax(net.h_strat(z), 1).numpy(),
                mu=r[:, :len(REG)].numpy(), sig=torch.exp(r[:, len(REG):].clamp(-6, 3)).numpy(),
                led_mu=(led[:, :6].numpy() * LED_SCALE), led_sig=(torch.exp(led[:, 6:].clamp(-6, 3)).numpy() * LED_SCALE))


def bank_inputs(bank, idx):
    o = bank['offs']
    return [F.net_input(bank['raw'][o[j]:o[j + 1]].T.astype(np.float32), bank['flags'][j]) for j in idx]


# ---------------- simulated validation ----------------
PRIOR_MED = None


def sim_scores(net, bank, prior=None):
    meta = {m['vid']: m for m in bank['meta']}
    vids = bank['veh']; cars = sorted(set(vids.tolist()))
    idx = np.arange(len(vids))
    inp = bank_inputs(bank, idx)
    Y = np.stack([reg_targets(meta[v]) for v in cars]); A = np.array([ARCHS.index(meta[v]['arch']) for v in cars])
    Sg = np.array([STRATS.index(meta[v]['strategy']) for v in cars])
    cmap = {v: i for i, v in enumerate(cars)}
    res = {}
    for mode in ['single', 'pooled']:
        if mode == 'single':
            P = predict(net, inp, np.arange(len(idx)))
            y = Y[[cmap[v] for v in vids]]; a = A[[cmap[v] for v in vids]]; sg = Sg[[cmap[v] for v in vids]]
        else:
            P = predict(net, inp, np.array([cmap[v] for v in vids]))
            y, a, sg = Y, A, Sg
        r = {}
        pa = P['arch'].argmax(1)
        r['arch_acc'] = float((pa == a).mean())
        vt = np.array([VEDTYPES.index(VEDTYPE[ARCHS[i]]) for i in a]); vp = np.array([VEDTYPES.index(VEDTYPE[ARCHS[i]]) for i in pa])
        r['vedtype_acc'] = float((vt == vp).mean())
        r['strat_acc'] = float((P['strat'].argmax(1) == sg).mean())
        maj = np.bincount(a).max() / len(a)
        sk = {'arch': max(0.0, (r['arch_acc'] - maj) / (1 - maj))}
        med = prior if prior is not None else np.nanmedian(Y, 0)
        for i, k in enumerate(REG):
            ok = np.isfinite(y[:, i])
            if ok.sum() < 10: continue
            e = np.abs(P['mu'][ok, i] - y[ok, i]); e0 = np.abs(med[i] - y[ok, i])
            r[f'{k}_mae'] = float(e.mean()); r[f'{k}_in1s'] = float((e < P['sig'][ok, i]).mean())
            if k in ('mass', 'disp', 'CdA', 'Crr'):
                r[f'{k}_mape'] = float(np.median(np.abs(np.exp(P['mu'][ok, i] - y[ok, i]) - 1)))
            sk[k] = max(0.0, 1 - e.mean() / e0.mean())
        sk['strat'] = max(0.0, (r['strat_acc'] - np.bincount(sg).max() / len(sg)) / (1 - np.bincount(sg).max() / len(sg)))
        if mode == 'single':
            led = bank['ledger']; ok = bank['km'] > 0.3
            e = np.abs(P['led_mu'][ok] - led[ok]); e0 = np.abs(np.median(led[ok], 0) - led[ok])
            r['led_mae'] = e.mean(0).round(1).tolist(); r['led_in1s'] = float((e < P['led_sig'][ok]).mean())
            sk['ledger'] = float(np.mean(np.maximum(0, 1 - e.mean(0) / e0.mean(0))))
        r['skill'] = {k: round(float(v), 4) for k, v in sk.items()}
        r['objective'] = float(np.mean(list(sk.values())))
        res[mode] = r
    res['objective'] = 0.5 * (res['single']['objective'] + res['pooled']['objective'])
    return res


# ---------------- real VED cars ----------------
def real_truth():
    vt = rd.ved_vehicles()
    return vt


def real_predict(net, rb):
    """returns per-trip predictions (windows of a trip pooled) and per-car predictions (all windows pooled)"""
    o = rb['offs']
    inp = [F.net_input(rb['raw'][o[j]:o[j + 1]].T.astype(np.float32), rb['flags'][j]) for j in range(len(rb['trip']))]
    trips = list(dict.fromkeys(rb['trip'].tolist())); tmap = {t: i for i, t in enumerate(trips)}
    cars = list(dict.fromkeys(rb['vid'].tolist())); cmap = {c: i for i, c in enumerate(cars)}
    Pt = predict(net, inp, np.array([tmap[t] for t in rb['trip']]))
    Pc = predict(net, inp, np.array([cmap[c] for c in rb['vid']]))
    tvid = {t: v for t, v in zip(rb['trip'], rb['vid'])}
    return dict(trips=trips, trip_vid=[tvid[t] for t in trips], Pt=Pt, cars=cars, Pc=Pc)


def real_scores(net, split='dev', rb=None, sigk=None):
    import realbank
    rb = rb or realbank.load(split)
    R = real_predict(net, rb)
    vt = real_truth()
    out = {}
    curb = lambda v: vt.loc[v, 'gen_weight_lb'] * LB
    # architecture: VED type is the published truth; finer architecture is RoadData's inference
    t_true = np.array([vt.loc[v, 'ved_type'] for v in R['trip_vid']])
    t_pred = np.array([VEDTYPE[ARCHS[i]] for i in R['Pt']['arch'].argmax(1)])
    out['vedtype_acc_trip'] = float((t_true == t_pred).mean())
    c_true = np.array([vt.loc[v, 'ved_type'] for v in R['cars']])
    c_pred = np.array([VEDTYPE[ARCHS[i]] for i in R['Pc']['arch'].argmax(1)])
    out['vedtype_acc_car'] = float((c_true == c_pred).mean())
    a_true = np.array([vt.loc[v, 'architecture'] for v in R['cars']])
    out['arch_acc_car'] = float((a_true == np.array([ARCHS[i] for i in R['Pc']['arch'].argmax(1)])).mean())
    # mass (curb = predicted total - 136 kg) and displacement, per car pooled and per trip
    for mode, P, vids in [('car', R['Pc'], R['cars']), ('trip', R['Pt'], R['trip_vid'])]:
        cb = np.array([curb(v) for v in vids]); ok = np.isfinite(cb)
        mpred = from_reg(0, P['mu'][:, 0]) - CURB_OFFSET
        e = mpred[ok] - cb[ok]
        out[f'mass_mae_{mode}'] = float(np.mean(np.abs(e))); out[f'mass_bias_{mode}'] = float(np.median(e))
        out[f'mass_mape_{mode}'] = float(np.median(np.abs(e) / cb[ok]))
        out[f'mass_within_bin_{mode}'] = float(np.mean(np.abs(e) <= 250 * LB))
        ls = (np.log(mpred[ok] + CURB_OFFSET) - np.log(cb[ok] + CURB_OFFSET)); s = P['sig'][ok, 0] * (sigk or {}).get('mass', 1)
        out[f'mass_in1s_{mode}'] = float(np.mean(np.abs(ls) < np.sqrt(s ** 2 + (113 / 1700) ** 2 / 3)))
        dt = np.array([vt.loc[v, 'displacement_L'] for v in vids]); ok = np.isfinite(dt)
        if ok.sum():
            dp = from_reg(1, P['mu'][ok, 1])
            out[f'disp_mae_{mode}'] = float(np.mean(np.abs(dp - dt[ok]))); out[f'disp_mape_{mode}'] = float(np.median(np.abs(dp / dt[ok] - 1)))
            out[f'disp_in1s_{mode}'] = float(np.mean(np.abs(np.log(dp / dt[ok])) < P['sig'][ok, 1] * (sigk or {}).get('disp', 1)))
    out['n_cars'] = len(R['cars']); out['n_trips'] = len(R['trips'])
    return out, R
