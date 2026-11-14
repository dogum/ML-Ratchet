"""GridPulse-Summary for ERCOT (no public ERCOT frequency traces exist; ERCOT's operations reports give delta f, MW
loss, load, inertia and the renewable share). Three small sim-trained MLPs on the ERCOT part of a bank:
  loss  <- (delta f, inertia, load, IBR share)       'MW lost from the reported drop and inertia'
  dfnad <- (loss, inertia, load, IBR share)           'forecast the frequency drop' (tests the simulator itself)
  E     <- (delta f, loss, load, IBR share)           'inertia from the drop and loss' (weakly identifiable)
delta f = pre-event frequency minus the 1 s nadir (1 s mean measurement, random phase, noise-free).
Usage: python3 ercot_head.py <name> <bank>"""
from common import *
import torch, torch.nn as nn, bank as BK, measure as M, dataset as DS
torch.set_num_threads(1)
SPECS = {'loss': (['ldf', 'lE', 'lload', 'ibr'], 'lloss'), 'dfnad': (['lloss', 'lE', 'lload', 'ibr'], 'ldf'),
         'E': (['ldf', 'lloss', 'lload', 'ibr'], 'lE')}


def table(dev, L, rng):
    d1 = M.to_1s(dev, rng.uniform(0, 1, len(dev)))
    df = -d1.min(1)
    return pd.DataFrame(dict(ldf=np.log(np.maximum(df, 0.005)), lE=np.log(L.inertia_MWs.values / 1e3),
                             lload=np.log(L.demand_MW.values / 1e3), ibr=L.ibr_share.values,
                             lloss=np.log(L.loss_MW.values)))


class MLP(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.b = nn.Sequential(nn.Linear(nin, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.o = nn.Linear(64, 2)

    def forward(self, x):
        h = self.o(self.b(x)); return h[:, 0], h[:, 1].clamp(-6, 3)


def fit(T, xs, y, epochs=300, seed=0):
    torch.manual_seed(seed)
    X = T[xs].values.astype(np.float32); Y = T[y].values.astype(np.float32)
    xm, xsd, ym, ysd = X.mean(0), X.std(0) + 1e-6, Y.mean(), Y.std()
    Xt = torch.tensor((X - xm) / xsd); Yt = torch.tensor((Y - ym) / ysd)
    net = MLP(len(xs)); opt = torch.optim.Adam(net.parameters(), 3e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]; mu, ls = net(Xt[idx])
            l = (ls + 0.5 * ((Yt[idx] - mu) / ls.exp()) ** 2).mean(); opt.zero_grad(); l.backward(); opt.step()
    W = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    W.update(xm=xm, xs=xsd, ym=np.float32(ym), ys=np.float32(ysd))
    return W


def np_pred(W, X):
    from scipy.special import erf
    g = lambda h: 0.5 * h * (1 + erf(h / np.sqrt(2)))
    h = (np.asarray(X, np.float32) - W['xm']) / W['xs']
    h = g(h @ W['b.0.weight'].T + W['b.0.bias']); h = g(h @ W['b.2.weight'].T + W['b.2.bias'])
    o = h @ W['o.weight'].T + W['o.bias']
    return o[:, 0] * W['ys'] + W['ym'], np.exp(np.clip(o[:, 1], -6, 3)) * W['ys']


def features_from_events(e):
    """ERCOT ops-report rows -> feature table (same names)."""
    return pd.DataFrame(dict(ldf=np.log(np.maximum(e.delta_f_Hz.values, 0.005)), lE=np.log(e.inertia_MWs.values / 1e3),
                             lload=np.log(e.load_MW.values / 1e3), ibr=e.renew_share_pct.values / 100,
                             lloss=np.log(e.mw_loss.values)))


if __name__ == '__main__':
    name, bk = sys.argv[1], sys.argv[2]
    dev, L = BK.load(bk); s = (L.system == 'ERCO').values; dev, L = dev[s], L[s].reset_index(drop=True)
    keep = DS.selected(dev, L); dev, L = dev[keep], L[keep].reset_index(drop=True)
    rng = np.random.default_rng(0)
    T = table(dev, L, rng); n = len(T); tr = np.arange(n) % 10 != 0
    out = {}
    for k, (xs, y) in SPECS.items():
        W = fit(T[tr], xs, y)
        mu, sg = np_pred(W, T[~tr][xs].values)
        err = np.abs(mu - T[~tr][y].values)
        print(k, 'sim holdout MdAE (log)', float(np.median(err)), 'cov1', float((err <= sg).mean()), flush=True)
        out.update({f'{k}__{a}': b for a, b in W.items()})
    np.savez(f'{CKPT}/ercot_head_{name}.npz', **out)
    print('n ERCOT sims', n)
