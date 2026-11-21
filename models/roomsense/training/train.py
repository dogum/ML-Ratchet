"""Train a small residual 1D network entirely on the synthetic feature bank."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import common
from _paths import work
from pipeline import FALLBACK
from simbank import load


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 12, 7, stride=4, padding=3)
        self.conv2 = nn.Conv1d(12, 16, 5, stride=4, padding=2)
        self.hidden = nn.Linear(16 * 16 + 6, 32)
        self.output = nn.Linear(32, 4)

    def forward(self, curves, context):
        x = self.conv2(self.conv1(curves).relu()).relu().flatten(1)
        raw = self.output(self.hidden(torch.cat((x, context), 1)).relu())
        mask = context[:, 3:] != 0
        ordered = torch.sort(torch.where(mask, context[:, :3], torch.inf), dim=1).values
        count = mask.sum(1)
        anchor = torch.where(count == 3, ordered[:, 1], torch.where(
            count == 2, (ordered[:, 0] + ordered[:, 1]) * 0.5, np.log(FALLBACK)))
        mean = raw[:, :2] + torch.stack((anchor, torch.full_like(anchor, np.log(150))), dim=1)
        return torch.cat((mean, F.softplus(raw[:, 2:]) + 0.035), dim=1)


def train(name, bank, steps, seed, seconds_loss):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    sources = [load(name) for name in bank.split(',')]
    arrays = {key: np.concatenate([data[key][i::len(sources)] for i, data in enumerate(sources)])
              for key in sources[0]}
    x = torch.as_tensor(arrays['x'])
    z = torch.as_tensor(arrays['z'])
    y = torch.as_tensor(np.log(arrays['y']))
    model = Net()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012)
    loss_weight = torch.tensor([1., 0.35])
    generator = torch.Generator().manual_seed(seed)
    path = Path(work('checkpoints', f'{name}.pt', mkdir=True))
    start = 0
    if path.exists():
        state = torch.load(path, weights_only=True)
        if (state['bank'] != bank or state['seed'] != seed or state['step'] > steps
                or state.get('seconds_loss', 0.) != seconds_loss):
            raise ValueError(f'{name} already has different training settings')
        model.load_state_dict(state['weights'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['generator'])
        start = state['step']
    for step in range(start + 1, steps + 1):
        ids = torch.randint(len(y), (96,), generator=generator)
        prediction = model(x[ids].float(), z[ids])
        sigma = prediction[:, 2:]
        objective = (0.5 * ((prediction[:, :2] - y[ids]) / sigma).square() + sigma.log())
        loss = (objective * loss_weight).mean()
        if seconds_loss:
            rt_mae = (prediction[:, 0].exp() - y[ids, 0].exp()).abs().mean()
            loss = loss + seconds_loss * rt_mae
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()
        if step % 400 == 0 or step == steps:
            tmp = path.with_suffix(f'.{os.getpid()}.tmp')
            torch.save(dict(weights=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                            bank=bank, seed=seed, seconds_loss=seconds_loss, generator=generator.get_state()), tmp)
            tmp.replace(path)
            print(f'{name}: {step}/{steps}, loss {loss.item():.3f}', flush=True)


def load_weights(name):
    torch.set_num_threads(1)
    state = torch.load(work('checkpoints', f'{name}.pt'), weights_only=True)
    return {key: value.numpy() for key, value in state['weights'].items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('bank')
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--seed', type=int, default=541)
    parser.add_argument('--seconds-loss', type=float, default=0.)
    args = parser.parse_args()
    train(args.name, args.bank, args.steps, args.seed, args.seconds_loss)

