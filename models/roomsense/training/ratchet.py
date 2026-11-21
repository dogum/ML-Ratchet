"""Judge RoomSense candidates using frozen simulations and measured dev rooms."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import common
from _paths import MODEL, work
from exam import cases, fit_baselines, score
from mlratchet.ratchet import Ratchet, gates
from roomsense import forward
from simbank import load
from train import load_weights

CHANGES = {
    'r1': 'Image-source reflections and Eyring tails. Learn a correction to the classical estimate.',
    'r2': 'Reduce diffuse pressure from 0.390 to 0.316 using the measured dev twin gap.',
    'r3': 'Train on equal numbers of textbook and dev-calibrated simulations.',
    'r4': 'Add absolute RT60 error in seconds to the training objective on the mixed bank.',
}
FROZEN = ('val-1.0', 'val-1.1')
DEV_GATE = 'Room-weighted dev RT60 MAE may grow by at most 5%.'


def skill(weights, bank, dev_median):
    prediction = np.concatenate([forward(weights, bank['x'][i:i+250], bank['z'][i:i+250])
                                 for i in range(0, len(bank['y']), 250)])
    mae = np.mean(np.abs(np.exp(prediction[:, 0]) - bank['y'][:, 0]))
    denominator = np.mean(np.abs(dev_median - bank['y'][:, 0]))
    return max(0., float(1 - mae / denominator))


def evaluate(names, *, advance, final):
    dev = cases('dev')
    settings = fit_baselines(dev)
    banks = {key: load(key) for key in FROZEN[:2 if advance else 1]}
    ladder = Ratchet('roomsense', margin=0.01, dev_gate=gates.no_worse_than(1.05, keys=['rt60_mae_s']),
                     objective='RT60 MAE skill against the dev-median guess on frozen simulations',
                     rule='Keep when simulated RT60 skill gains more than 0.01 and the measured dev gate passes.',
                     frozen_set=FROZEN[0], name='RoomSense')
    scores = {}
    for name in names:
        weights = load_weights(name)
        metrics = score(dev, settings, weights)['metrics']
        marks = {key: skill(weights, v, settings['seconds']) for key, v in banks.items()}
        scores[name] = dict(sim=marks, dev=metrics)
        if name == 'r2':
            ladder.rescore({r.id: scores[r.id]['sim'][FROZEN[1]] for r in ladder.rungs}, FROZEN[1])
        ladder.judge(name, CHANGES[name], marks[ladder.frozen_set], dev={'rt60_mae_s': metrics['model_rt']},
                     metrics=[round(metrics['model_rt'], 5), round(metrics['model_volume'], 5)])
    if final:
        ladder.release()
    doc = ladder.to_dict()
    doc.update(frozen_sets=list(banks), dev_gate=DEV_GATE,
               columns_source='Measured dev experiments. Exam never used.',
               columns=['Dev RT60 MAE (s)', 'Dev volume MAE (ln m³)'])
    if final:
        Path(MODEL, 'ratchet.yaml').write_text(yaml.safe_dump({'model': 'roomsense', 'ladders': [doc]},
                                                            sort_keys=False, allow_unicode=True))
        Path(work('logs', 'ratchet.json', mkdir=True)).write_text(json.dumps(scores, indent=2) + '\n')
    return doc, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', action='store_true', help='Preview r1 on the original freeze')
    args = parser.parse_args()
    print(json.dumps(evaluate(('r1',) if args.first else tuple(CHANGES), advance=not args.first,
                              final=not args.first), indent=2))
