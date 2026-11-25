"""Torch version of the formula arm's parameter mapping, so the formula CONSTANTS can be tuned through the
differentiable simulator (hbv_torch.py). Trees are lawsearch JSON trees over the 18 raw static values; the operator
semantics follow lawsearch._ev, except that invalid domains (power or log of a non-positive base, sqrt of a negative)
are clamped instead of giving NaN; the numpy evaluation at inference is checked to stay finite on the train bank."""
import torch

import lawsearch as ls


def _ev(t, X, c):
    k = t[0]
    if k == 'x':
        return X[:, t[1]]
    if k == 'c':
        return c[t[1]].expand(X.shape[0])
    a = _ev(t[1], X, c)
    if k == '+': return a + _ev(t[2], X, c)
    if k == '-': return a - _ev(t[2], X, c)
    if k == '*': return a * _ev(t[2], X, c)
    if k == '/': return a / _ev(t[2], X, c)
    if k == '^': return torch.clamp(a, min=1e-12) ** _ev(t[2], X, c)
    if k == 'exp': return torch.exp(torch.clamp(a, -50, 50))
    if k == 'log': return torch.log(torch.clamp(a, min=1e-300))
    if k == 'sqrt': return torch.sqrt(torch.clamp(a, min=0.0) + 1e-30)
    if k == 'inv': return 1.0 / a
    if k == 'sq': return a * a
    raise ValueError(k)


class FormulaNet(torch.nn.Module):
    """forward(x_std) with x_std = (static - mu) / sd (as the MLP gets it); returns z (N, 17) in PARAMS order."""

    def __init__(self, F, names, mu, sd):
        super().__init__()
        self.trees = [ls.from_json(F['params'][n]['tree']) for n in names]
        self.consts = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(F['params'][n]['consts'], dtype=torch.float64))
                                              for n in names])
        self.fallback = torch.tensor([F['params'][n]['fallback'] for n in names], dtype=torch.float64)
        self.mu = torch.tensor(mu, dtype=torch.float64); self.sd = torch.tensor(sd, dtype=torch.float64)

    def forward(self, x):
        X = x * self.sd + self.mu
        cols = []
        for j, (t, c) in enumerate(zip(self.trees, self.consts)):
            v = _ev(t, X, c)
            v = torch.where(torch.isfinite(v), v, self.fallback[j])
            cols.append(torch.clamp(v, -12.0, 12.0))
        return torch.stack(cols, 1)
