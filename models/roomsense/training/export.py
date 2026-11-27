"""Verify the selected checkpoint and write a release lock before opening the exam."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import common
from _paths import MODEL, work
from data import MANIFEST
from exam import LOCK_PATH, calibrate, cases, fit_baselines, weight_digest
from roomsense import forward
from train import Net, load_weights


def export(output):
    if LOCK_PATH.exists():
        raise ValueError('A release is already locked in this work directory.')
    card = yaml.safe_load(Path(MODEL, 'ratchet.yaml').read_text())['ladders'][0]
    rung = next(row['id'] for row in card['rungs'] if row['decision'] == 'released')
    dev = cases('dev')
    weights = load_weights(rung)
    net = Net()
    net.load_state_dict({key: torch.from_numpy(a) for key, a in weights.items()})
    features = [row['features'] for row in dev if row['features'] is not None]
    x, z = np.stack([f.curves for f in features]), np.stack([f.context for f in features])
    with torch.no_grad():
        reference = net(torch.from_numpy(x), torch.from_numpy(z)).numpy()
    difference = float(np.max(np.abs(reference - forward(weights, x, z))))
    if difference > 1e-5:
        raise ValueError(f'NumPy parity error {difference:g}')
    factors = calibrate(dev, weights)
    meta = dict(rung=rung, widen=factors, parameters=sum(w.size for w in weights.values()),
                pytorch_to_numpy_max_abs=difference, target='midband RT60 and experimental room volume')
    output = Path(output)
    with output.open('wb') as f:
        np.savez_compressed(f, **weights, meta=json.dumps(meta))
    lock = dict(rung=rung, weights_sha256=weight_digest(output), baselines=fit_baselines(dev),
                split_sha256=weight_digest(MANIFEST))
    Path(work('logs', 'release_lock.json', mkdir=True)).write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps(dict(**meta, **{k: v for k, v in lock.items() if k != 'rung'}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=Path(MODEL, 'roomsense_weights.npz'))
    export(parser.parse_args().out)
