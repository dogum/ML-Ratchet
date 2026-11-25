"""The MLP of train_dpl.py (same definition), importable without starting a training run."""
import torch

import hbv_torch as H


class Net(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        dims = [18] + list(hidden)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = torch.nn.Linear(dims[-1], len(H.PARAMS))
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.copy_(torch.tensor(H.from_params(H.DEFAULTS)))

    def forward(self, x):
        for L in self.layers:
            x = torch.nn.functional.silu(L(x))
        return self.out(x)
