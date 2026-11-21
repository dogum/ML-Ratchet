"""Build deterministic, resumable feature banks with disjoint real backgrounds."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import common
from _paths import work
from data import noise_pool
from pipeline import RATE, preprocess
from simulator import clap, clean_response, record, reference_seconds, sample_room


def build(name, count, seed, role, physics, chunk=250):
    directory = Path(work('banks', name))
    directory.mkdir(parents=True, exist_ok=True)
    spec = dict(count=count, seed=seed, role=role, physics=physics, chunk=chunk)
    manifest = directory / 'bank.json'
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError(f'{name} already has different bank settings')
    manifest.write_text(json.dumps(spec, indent=2) + '\n')
    pool = noise_pool(role)
    for start in range(0, count, chunk):
        path = directory / f'{start:06d}.npz'
        if path.exists():
            continue
        curves, context, labels, baseline = [], [], [], []
        for index in range(start, min(start + chunk, count)):
            rng = np.random.default_rng([seed, index])
            noise = pool[rng.integers(len(pool))]
            for _ in range(80):
                room = sample_room(rng)
                ir = clean_response(room, rng, physics)
                seconds = reference_seconds(ir)
                if seconds is None or not 0.13 <= seconds <= 2.5:
                    continue
                try:
                    reading = preprocess(record(ir, clap(rng), noise, rng), RATE)
                except ValueError:
                    continue
                curves.append(reading.curves)
                context.append(reading.context)
                labels.append([seconds, room.volume])
                baseline.append(np.nan if reading.classical_seconds is None else reading.classical_seconds)
                break
            else:
                raise RuntimeError(f'Unable to build simulation case {index}')
        tmp = path.with_suffix(f'.{os.getpid()}.npz')
        np.savez(tmp, x=np.asarray(curves, np.float16), z=np.asarray(context, np.float32),
                 y=np.asarray(labels, np.float32), baseline=np.asarray(baseline, np.float32))
        tmp.replace(path)
        print(f'{name}: {min(start + chunk, count)}/{count}', flush=True)


def load(name):
    directory = Path(work('banks', name))
    spec = json.loads((directory / 'bank.json').read_text())
    parts = [directory / f'{start:06d}.npz' for start in range(0, spec['count'], spec['chunk'])]
    result = {key: [] for key in ('x', 'z', 'y', 'baseline')}
    for part in parts:
        with np.load(part, allow_pickle=False) as arrays:
            for key in result:
                result[key].append(arrays[key])
    return {key: np.concatenate(arrays) for key, arrays in result.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('count', type=int)
    parser.add_argument('seed', type=int)
    parser.add_argument('--role', choices=['train', 'validation'], required=True)
    parser.add_argument('--physics', default='1.0', choices=['1.0', '1.1'])
    args = parser.parse_args()
    build(args.name, args.count, args.seed, args.role, args.physics)

