"""Train one FireDetective rung on simulation banks only (resumable checkpoints) and score it on the frozen sim
validation set and the dev real records.

python3 train.py <rung_name> --banks train_v1[,more] [--epochs 60] [--hidden-p 0.5] [--seed 0] [--h 48 --c 64 --d 128]
Writes models/<rung>.pt, models/<rung>.json (sim + dev metrics).

27 Nov 2026 (audit follow-up): --flag-sd1 stores the blind-wind flag (wind_hidden) with scale 1 instead of the 1e-3
floor (the flag is 0 on every wind-visible row the scaler sees, so the floor made a blind row enter at z = 1000).
Checkpoints now also hold the torch and numpy RNG states, so a resumed run continues the same random stream.
"""
from common import *
import argparse, glob, time
import torch
from scipy.stats import spearmanr
import model as M
import feats

torch.set_num_threads(1)
MOD = f'{ROOT}/models'
os.makedirs(MOD, exist_ok=True)


def load_bank(names, drop_edge=False):
    S, Sh, St, Lb = [], [], [], []
    labels = None
    for nm in names.split(','):
        nm, _, upto = nm.partition(':')         # 'train_v1:30' = shards 0-29 only (27 Nov 2026, to rebuild R6's bank)
        for fn in sorted(glob.glob(f'{BANK}/{nm}/shard_*.npz'))[:int(upto) if upto else None]:
            z = np.load(fn)
            if len(z['lab']) == 0:
                continue
            labels = list(z['labels'])
            S.append(z['seq']); Sh.append(z['seq_h']); St.append(z['static']); Lb.append(z['lab'])
    w = min(l.shape[1] for l in Lb)          # v2/v3 banks append extra label columns: keep the common ones
    Lb = [l[:, :w] for l in Lb]
    labels = labels[:w] if len(labels) > w else labels
    S, Sh, St, Lb = (np.concatenate(a) for a in (S, Sh, St, Lb))
    keep = np.isfinite(Lb).all(1)
    if drop_edge:
        keep &= Lb[:, labels.index('edge')] < 0.5
    return S[keep].astype(np.float32), Sh[keep].astype(np.float32), St[keep], Lb[keep], labels


def norm_stats(S, St):
    m = S[..., 0] > 0
    v = S[m]
    mu, sd = v.mean(0), v.std(0) + 1e-3
    mu[0] = 0; sd[0] = 1
    wc = feats.WIND_COLS + [feats.SEQ_NAMES.index('wind_hidden')]
    mu[wc] = 0  # keep hidden wind exactly zero after normalisation
    return dict(seq_mu=mu, seq_sd=sd, st_mu=St.mean(0), st_sd=St.std(0) + 1e-3)


def prep(S, St, ns):
    X = (S - ns['seq_mu']) / ns['seq_sd']
    mask = (S[..., 0] > 0).astype(np.float32)
    X = X * mask[..., None]
    T = (St - ns['st_mu']) / ns['st_sd']
    return X.astype(np.float32), T.astype(np.float32), mask


def predict(net, ns, S, St, bs=1024):
    X, T, mk = prep(S, St, ns)
    net.eval()
    out_mu, out_ls, out_b = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            mu, ls, b = net(torch.from_numpy(X[i:i + bs]), torch.from_numpy(T[i:i + bs]), torch.from_numpy(mk[i:i + bs]))
            out_mu.append(mu.numpy()); out_ls.append(ls.numpy()); out_b.append(b.numpy())
    return np.concatenate(out_mu), np.exp(np.concatenate(out_ls)), 1 / (1 + np.exp(-np.concatenate(out_b)))


def sim_metrics(net, ns, val):
    S, Sh, St, Lb, labels = val
    Y, B = M.targets(Lb, labels)
    mu, sd, pb = predict(net, ns, S, St)
    muh, sdh, _ = predict(net, ns, Sh, St)
    out = {}
    for j, k in enumerate(M.REG):
        for tag, m_, s_ in (('', mu, sd), ('_hidden', muh, sdh)):
            y = Y[:, j]
            r2 = 1 - np.mean((m_[:, j] - y) ** 2) / np.var(y)
            z = (m_[:, j] - y) / s_[:, j]
            out[k + tag] = dict(r2=float(r2), rho=float(spearmanr(m_[:, j], y)[0]), cov80=float(np.mean(np.abs(z) < 1.2816)))
    for j, k in enumerate(M.BIN):
        from sklearn.metrics import roc_auc_score
        out[k] = dict(auc=float(roc_auc_score(B[:, j], pb[:, j])))
    # primary sim skill: mean clipped R2 over the three targets graded on real fires (crown, wind with wind hidden, lfmc)
    out['primary'] = float(np.mean([max(out['crown']['r2'], 0), max(out['wind_hidden']['r2'], 0), max(out['lfmc']['r2'], 0)]))
    return out


def real_arrays(recs):
    S = np.stack([r['seq'] for r in recs]).astype(np.float32)
    Sh = np.stack([r['seq_hidden'] for r in recs]).astype(np.float32)
    St = np.stack([r['static'] for r in recs]).astype(np.float32)
    return S, Sh, St


def real_predict(net, ns, recs):
    S, Sh, St = real_arrays(recs)
    mu, sd, pb = predict(net, ns, S, St)
    muh, sdh, _ = predict(net, ns, Sh, St)
    return dict(crown=M.i_crown(mu[:, 0]), crown_y=mu[:, 0], crown_sd=sd[:, 0], wind=np.exp(muh[:, 1]),
                wind_y=muh[:, 1], wind_sd=sdh[:, 1], lfmc=mu[:, 2] * 100, lfmc_sd=sd[:, 2] * 100,
                ros=np.exp(mu[:, 3]), m1=np.exp(mu[:, 4]), regime=mu[:, 6], spot=pb[:, 0], ia=pb[:, 1],
                wind_full=np.exp(mu[:, 1]))


def dev_metrics(net, ns, recs):
    p = real_predict(net, ns, recs)
    hs = np.array([r['hs'] for r in recs]); wind = np.array([r['wind'] for r in recs]); lf = np.array([r['lfmc'] for r in recs])
    ff = np.array([r['forest_frac'] for r in recs])
    out = {}
    ok = np.isfinite(hs)
    out['hs_rho_all'] = float(spearmanr(p['crown'][ok], hs[ok])[0]); out['hs_n_all'] = int(ok.sum())
    okf = ok & (ff >= 0.5)
    out['hs_rho_forest'] = float(spearmanr(p['crown'][okf], hs[okf])[0]); out['hs_n_forest'] = int(okf.sum())
    ok = np.isfinite(wind)
    out['wind_rho'] = float(spearmanr(p['wind'][ok], wind[ok])[0]); out['wind_n'] = int(ok.sum())
    out['wind_mae'] = float(np.mean(np.abs(p['wind'][ok] - wind[ok])))
    ok = np.isfinite(lf)
    out['lfmc_rho'] = float(spearmanr(p['lfmc'][ok], lf[ok])[0]) if ok.sum() > 3 else np.nan; out['lfmc_n'] = int(ok.sum())
    out['composite'] = float(np.nanmean([out['hs_rho_all'], out['wind_rho']]))
    return out


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    S, Sh, St, Lb, labels = load_bank(args.banks, args.drop_edge)
    val = load_bank(args.val)
    ns = norm_stats(S, St)
    for nm in [x for x in args.drop_feats.split(',') if x]:
        j = feats.SEQ_NAMES.index(nm)          # dropped input: normalised to ~0 everywhere (train, numpy, JS)
        ns['seq_mu'][j] = 0.0; ns['seq_sd'][j] = 1e9
    if args.flag_sd1:
        ns['seq_sd'][feats.SEQ_NAMES.index('wind_hidden')] = 1.0
    Y, B = M.targets(Lb, labels)
    X, T, mk = prep(S, St, ns)
    Xh, _, _ = prep(Sh, St, ns)
    n = len(X)
    print('train n', n, 'val n', len(val[0]), flush=True)
    net = M.Net(X.shape[2], T.shape[1], args.h, args.c, args.d)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    ck = f'{MOD}/{args.rung}.ckpt'
    ep0 = 0
    if os.path.exists(ck):
        c = torch.load(ck, weights_only=False)
        net.load_state_dict(c['net']); opt.load_state_dict(c['opt']); ep0 = c['epoch'] + 1
        if 'torch_rng' in c:                           # checkpoints written from 27 Nov 2026 on
            torch.set_rng_state(c['torch_rng']); np.random.set_state(c['np_rng'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, last_epoch=ep0 - 1) if ep0 else \
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    wts = np.ones(n, np.float32)
    if args.prior:
        # dev-informed prior (importance weights vs the uniform sampling priors): ROS factor lognormal around the
        # dev replay calibration (median 0.45, log-sd 0.45); LFMC normal(100, 30) % (dev Globe-LFMC fire-season sites)
        L_ = {k: Lb[:, i] for i, k in enumerate(labels)}
        wts = np.exp(-0.5 * ((L_['log_ros_adj'] - np.log(0.45)) / 0.45) ** 2) * np.exp(-0.5 * ((L_['lfmc'] - 100) / 30) ** 2)
        wts = np.clip(wts / wts.mean(), 0, 5).astype(np.float32); wts /= wts.mean()
        print('prior weights: effective n', float(wts.sum() ** 2 / (wts ** 2).sum()), flush=True)
    tW = torch.from_numpy(wts)
    tX, tXh, tT, tm, tY, tB = (torch.from_numpy(a) for a in (X, Xh, T, mk, Y, B))
    w_reg = torch.tensor([float(x) for x in args.wreg.split(',')]) if args.wreg else torch.ones(len(M.REG))
    for ep in range(ep0, args.epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            hid = (torch.rand(len(idx)) < args.hidden_p)[:, None, None]
            xb = torch.where(hid, tXh[idx], tX[idx])
            mu, ls, bl = net(xb, tT[idx], tm[idx])
            nll = 0.5 * ((mu - tY[idx]) / ls.exp()) ** 2 + ls
            wb = tW[idx][:, None]
            loss = (nll * w_reg * wb).mean() + 0.3 * (torch.nn.functional.binary_cross_entropy_with_logits(
                bl, tB[idx], reduction='none') * wb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        if ep % 5 == 4 or ep == args.epochs - 1:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), epoch=ep, torch_rng=torch.get_rng_state(),
                            np_rng=np.random.get_state()), ck)
            sm = sim_metrics(net, ns, val)
            print(f'ep {ep} loss {tot / n:.4f} primary {sm["primary"]:.4f} crown {sm["crown"]["r2"]:.3f} '
                  f'windH {sm["wind_hidden"]["r2"]:.3f} lfmc {sm["lfmc"]["r2"]:.3f} ros {sm["ros"]["r2"]:.3f}', flush=True)
    sm = sim_metrics(net, ns, val)
    from real import load_records
    recs = [r for r in load_records(f'{REAL}/dev_records_package.pkl') if r is not None]
    dm = dev_metrics(net, ns, recs)
    torch.save(dict(net=net.state_dict(), ns=ns, args=vars(args), n_seq=X.shape[2], n_stat=T.shape[1]), f'{MOD}/{args.rung}.pt')
    jdump(dict(rung=args.rung, args=vars(args), n_train=n, sim=sm, dev=dm), f'{MOD}/{args.rung}.json')
    print('SIM', json.dumps({k: v for k, v in sm.items() if k in ('primary',)}), 'DEV', json.dumps(dm), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('rung'); ap.add_argument('--banks', default='train_v1'); ap.add_argument('--val', default='val_v1')
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--hidden-p', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--h', type=int, default=48)
    ap.add_argument('--c', type=int, default=64); ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--drop-edge', action='store_true'); ap.add_argument('--wreg', default='')
    ap.add_argument('--drop-feats', default='')
    ap.add_argument('--prior', action='store_true')
    ap.add_argument('--flag-sd1', action='store_true')
    train(ap.parse_args())
