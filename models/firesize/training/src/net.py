"""FireSize network: a small MLP from per-ignition features to a 12-class distribution over FPA-FOD size classes.
numpy inference (release) + torch training."""
import numpy as np

SIZE_EDGES_AC = [0.095, 0.255, 0.995, 2.995, 9.995, 29.995, 99.995, 299.995, 999.995, 4999.995, 19999.995]


def forward_np(W, X):
    """W: dict with mu, sd, fill and layers [e<i>_]L<k>_w/_b (an ensemble averages member probabilities);
    X: n x f raw features (NaN allowed). Returns n x 12 probabilities."""
    Z = np.where(np.isfinite(X), X, W['fill'][None, :])
    z = (Z - W['mu']) / W['sd']
    ne = int(W.get('n_ens', 0))
    pre = [f'e{i}_' for i in range(ne)] if ne else ['']
    out = 0
    for pf in pre:
        h = z
        k = 0
        while f'{pf}L{k}_w' in W:
            h = h @ W[f'{pf}L{k}_w'].T + W[f'{pf}L{k}_b']
            if f'{pf}L{k + 1}_w' in W:
                h = np.maximum(h, 0.0)
            k += 1
        h = h / float(W.get('temp', 1.0))
        h = h - h.max(1, keepdims=True)
        p = np.exp(h)
        out = out + p / p.sum(1, keepdims=True)
    return out / len(pre)


def torch_mlp(nin, hidden, nout, seed=0):
    import torch
    torch.manual_seed(seed)
    layers = []
    d = nin
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    layers += [torch.nn.Linear(d, nout)]
    return torch.nn.Sequential(*layers)


def export(models, mu, sd, fill, feats, extra=None):
    import torch
    models = models if isinstance(models, list) else [models]
    W = dict(mu=mu.astype(np.float64), sd=sd.astype(np.float64), fill=fill.astype(np.float64),
             feats=np.array(feats), temp=1.0, n_ens=len(models))
    for i, model in enumerate(models):
        k = 0
        for m in model:
            if isinstance(m, torch.nn.Linear):
                W[f'e{i}_L{k}_w'] = m.weight.detach().numpy().astype(np.float64)
                W[f'e{i}_L{k}_b'] = m.bias.detach().numpy().astype(np.float64)
                k += 1
    if extra:
        W.update(extra)
    return W
