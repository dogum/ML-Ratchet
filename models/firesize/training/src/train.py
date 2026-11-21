"""Train one FireSize rung on simulated ignitions and score it (frozen sim set + real dev gate half).
usage: train.py <rung> '<json config>'
config keys: physics (bank version), theta (calib json name or 'lit'), feats ('base'), hidden, epochs, lr, seed,
             frozen (frozen-set version tag = '<physics>|<theta>'; default = this rung's own)
-> models/<rung>.npz, models/<rung>.json"""
import sys, json, time, os
from common import *
import torch
torch.set_num_threads(1)
import supp, calib, net
import features as F
from firedata import metrics as M

MOD = f'{ROOT}/models'
os.makedirs(MOD, exist_ok=True)


def theta_of(name):
    if name == 'lit':
        return calib.theta_from(calib.V0)
    p = f'{DATA}/calib_{name}.json'
    if not os.path.exists(p):                       # import: fall back to the shipped fit
        p = f'{TRAIN}/results/calibration/calib_{name}.json'
    return json.load(open(p))['theta']


def sim_set(physics, theta, seed=0, n_ign=None):
    """Simulated ignitions with features, soft labels, importance weights, hard draw."""
    B = calib.load_bank(physics, 'train')
    if n_ign:
        m = B['ign'] < n_ign
        B = {k: v[m] for k, v in B.items()}
    T = calib.Twins(physics, kind='train', B=B)
    th = theta_of(theta)
    P, w, hard = T.sim_probs(th)
    X = pd.read_parquet(f'{DATA}/feat_sim.parquet').set_index('ign').loc[T.B['ign']]
    I = pd.read_parquet(f'{DATA}/ignitions.parquet').set_index('ign').loc[T.B['ign']]
    Wn = pd.read_parquet(f'{DATA}/windows.parquet').set_index('win')
    frozen = Wn.loc[I.win.values, 'frozen'].values.astype(bool)
    return dict(X=X.reset_index(drop=True), P=P, w=w / np.mean(w), hard=hard, frozen=frozen, win=I.win.values,
                ign=T.B['ign'])


def real_dev(half):
    d = fod_records('dev')
    d = d[[dev_half(i) == half for i in d.FOD_ID]].reset_index(drop=True)
    X = pd.read_parquet(f'{DATA}/feat_dev.parquet').set_index('FOD_ID').loc[d.FOD_ID].reset_index(drop=True)
    return d, X


def rps(Pf, Po):
    return np.sum((np.cumsum(Pf, 1) - np.cumsum(Po, 1)) ** 2, 1)


def score_sim(W, S):
    m = S['frozen']
    Pf = net.forward_np(W, S['X'][list(W['feats'])].values[m])
    Po = S['P'][m]; w = S['w'][m]
    clim = np.sum(w[:, None] * Po, 0) / w.sum()
    r_f = np.sum(w * rps(Pf, Po)) / w.sum()
    r_c = np.sum(w * rps(np.repeat(clim[None], len(Po), 0), Po)) / w.sum()
    from sklearn.metrics import roc_auc_score
    y = (S['hard'][m] >= supp.ESC_BIN).astype(int)
    pe = Pf[:, supp.ESC_BIN:].sum(1)
    return dict(n=int(m.sum()), rpss=float(1 - r_f / r_c), esc_auc=float(roc_auc_score(y, pe, sample_weight=w)),
                esc_rate_sim=float(np.sum(w * Po[:, supp.ESC_BIN:].sum(1)) / w.sum()),
                esc_rate_pred=float(np.sum(w * pe) / w.sum()))


def score_real(W, d, X):
    Pf = net.forward_np(W, X[list(W['feats'])].values)
    pe = Pf[:, supp.ESC_BIN:].sum(1)
    pl = Pf[:, supp.LARGE_BIN:].sum(1)
    med = 10 ** supp.median_log10(Pf)
    return dict(escape=M.binary_scores(d.escape.values, pe), large=M.binary_scores(d.large.values, pl),
                size=M.size_scores(d.size_ha.values, med), mean_p_escape=float(pe.mean())), Pf


def fit(S, cfg, feats):
    rs = np.random.default_rng(cfg.get('seed', 0))
    tr = ~S['frozen']
    wins = np.unique(S['win'][tr])
    vw = set(rs.choice(wins, max(1, len(wins) // 10), replace=False))
    va = tr & np.isin(S['win'], list(vw))
    tr = tr & ~va
    X = S['X'][feats].values.astype(np.float64)
    nf = ~S['frozen']                      # normalisation from all non-frozen rows (same for every ensemble member)
    fill = np.nanmedian(X[nf], 0)
    fill = np.where(np.isfinite(fill), fill, 0)
    Xf = np.where(np.isfinite(X), X, fill[None])
    mu = Xf[nf].mean(0); sd = Xf[nf].std(0) + 1e-6
    Z = torch.tensor((Xf - mu) / sd, dtype=torch.float32)
    Y = torch.tensor(S['P'], dtype=torch.float32)
    Wt = torch.tensor(S['w'], dtype=torch.float32)
    model = net.torch_mlp(len(feats), cfg.get('hidden', [128, 128]), supp.NB, cfg.get('seed', 0))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 2e-3), weight_decay=cfg.get('wd', 1e-4))
    itr, iva = np.nonzero(tr)[0], np.nonzero(va)[0]

    def loss_on(idx):
        lp = torch.log_softmax(model(Z[idx]), 1)
        return -(Wt[idx] * (Y[idx] * lp).sum(1)).sum() / Wt[idx].sum()
    best, best_state, bad = 1e9, None, 0
    for ep in range(cfg.get('epochs', 60)):
        model.train()
        perm = rs.permutation(itr)
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            l = loss_on(b)
            l.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lv = float(loss_on(iva))
        if lv < best - 1e-4:
            best, bad = lv, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best_state)
    return model, mu, sd, fill, dict(val_loss=best, epochs=ep + 1, n_train=int(tr.sum()), n_val=int(va.sum()))


def feature_list(kind):
    X = pd.read_parquet(f'{DATA}/feat_sim.parquet')
    base = [c for c in X.columns if c != 'ign' and not c.startswith('ros')]
    if kind == 'base':
        return base
    if kind == 'phys':
        return base + [c for c in X.columns if c.startswith('ros')]
    if kind == 'noregion':
        return [c for c in base if not c.startswith('r_')]
    raise ValueError(kind)


def main():
    rung = sys.argv[1]
    cfg = json.loads(sys.argv[2])
    t0 = time.time()
    feats = feature_list(cfg.get('feats', 'base'))
    S = sim_set(cfg['physics'], cfg['theta'], n_ign=cfg.get('n_ign'))
    models, infos = [], []
    for e in range(cfg.get('n_ens', 3)):
        c = dict(cfg, seed=cfg.get('seed', 0) + 100 * e)
        model, mu, sd, fill, info = fit(S, c, feats)
        models.append(model); infos.append(info)
    info = dict(infos[0], val_loss=[i['val_loss'] for i in infos], epochs=[i['epochs'] for i in infos])
    W = net.export(models, mu, sd, fill, feats)
    np.savez(f'{MOD}/{rung}.npz', **W)
    res = dict(rung=rung, cfg=cfg, info=info, frozen_version=f"{cfg['physics']}|{cfg['theta']}",
               sim=score_sim(W, S), wall_s=time.time() - t0)
    d, X = real_dev('gate')
    res['dev_gate'], _ = score_real(W, d, X)
    d, X = real_dev('calib')
    res['dev_calib'], _ = score_real(W, d, X)
    json.dump(res, open(f'{MOD}/{rung}.json', 'w'), indent=1, default=float)
    print(json.dumps(dict(rung=rung, sim=res['sim'], gate_esc=res['dev_gate']['escape'], gate_size=res['dev_gate']['size'],
                          info=info), default=float), flush=True)


if __name__ == '__main__':
    main()
