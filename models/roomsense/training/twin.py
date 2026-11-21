"""Fit the diffuse-to-early energy ratio from dev measurements with known geometry."""
from collections import defaultdict
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import common
from _paths import work
from data import measured_responses, reference_measurement
from pipeline import RATE, resample
from simulator import Room, clean_response, eyring

BASE_DENSITY = 0.39


def signature(response):
    peak = int(np.argmax(np.abs(response)))
    response = np.pad(response[peak:], (0, RATE // 4))[:RATE // 4]
    energy = response**2
    early = energy[:int(.045 * RATE)].sum()
    late = energy[int(.045 * RATE):].sum()
    return 10 * np.log10(max(late, 1e-16) / max(early, 1e-16))


def geometry(dimensions, source, position):
    dims = np.asarray(dimensions)
    receiver = dims * np.array([0.52, 0.52, 0.45])
    if source == 'surrey':
        angle = np.deg2rad(int(position.split('-')[-1]) * 5 - 90)
        emitter = receiver + 1.5 * np.array([np.cos(angle), np.sin(angle), 0.])
    else:
        # Source placement is undocumented numerically for these files.
        # Hold one plausible source distance per setup.
        setup = ('2A', '2B', '2C', '3A', '3B').index(position)
        emitter = receiver + np.array([1.4 + 0.3*setup, 0.6, 0.])
    return dims, emitter, receiver


def diagnostics():
    rows = defaultdict(list)
    for meta, position, measured, rate in measured_responses('dev'):
        if not meta['dimensions_m']:
            continue
        measured = resample(measured, rate)
        target = reference_measurement(measured)
        dims, emitter, receiver = geometry(meta['dimensions_m'], meta['source'], position)
        def make(alpha):
            return Room(dims, emitter, receiver, np.full((6, 3), alpha))
        alpha = brentq(lambda a: eyring(make(a))[1] - target, 0.003, 0.995)
        room = make(alpha)
        seeds = [int.from_bytes(sha256(f'{meta["name"]}/{position}/{n}'.encode()).digest()[:8], 'little')
                 for n in range(6)]
        twins = [signature(clean_response(room, np.random.default_rng(seed))) for seed in seeds]
        rows[f'{meta["source"]}/{meta["name"]}'].append(dict(position=position, rt60_s=target,
                absorption=alpha, measured_late_to_early_db=signature(measured),
                twin_late_to_early_db=float(np.median(twins))))
    gaps = {room: float(np.median([r['measured_late_to_early_db'] - r['twin_late_to_early_db']
                                   for r in readings])) for room, readings in rows.items()}
    median_gap = float(np.median(list(gaps.values())))
    # Take a partial step in dB because two rooms give limited prior evidence.
    proposed = BASE_DENSITY * float(np.clip(10**(0.6 * median_gap / 20), 0.4, 2.5))
    return dict(metric='10 log10 energy ratio, 45 to 250 ms over first 45 ms after peak',
                geometry='Measured dimensions and Surrey 1.5 m arc. Approximate receiver and Kitamura positions.',
                reference='Dev sweep-derived midband RT60 used to match Eyring absorption.',
                room_gaps_db=gaps, median_gap_db=median_gap,
                v1_density=BASE_DENSITY, proposed_density=proposed, rooms=rows)


if __name__ == '__main__':
    result = diagnostics()
    target = Path(work('logs', 'twins.json', mkdir=True))
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rooms'}, indent=2))
