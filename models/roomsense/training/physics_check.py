"""Independent path geometry, diffuse decay and classical fit checks."""
import argparse
import json

import numpy as np

import common  # adds the standalone model to the import path
from simulator import Room, SOUND_SPEED, eyring, image_paths
from pipeline import RATE, preprocess


def check():
    room = Room(np.array([6., 5., 3.]), np.array([2., 1., 1.]), np.array([3., 2., 1.5]),
                np.full((6, 3), 0.2))
    direct = np.sqrt(1**2 + 1**2 + 0.5**2)
    expected = np.sqrt([5**2 + 1**2 + 0.5**2, 7**2 + 1**2 + 0.5**2,
                        1**2 + 3**2 + 0.5**2, 1**2 + 7**2 + 0.5**2,
                        1**2 + 1**2 + 2.5**2, 1**2 + 1**2 + 3.5**2])
    paths = image_paths(room)
    actual_direct = [distance for distance, order, _ in paths if order == 0][0]
    first = sorted((distance, coeff[0]) for distance, order, coeff in paths if order == 1)
    expected_gain = np.sqrt(0.8)
    arrival_error = max(abs(direct - actual_direct),
                        *abs(np.sort(expected) - np.array([distance for distance, _ in first])))
    gain_error = max(abs(gain - expected_gain) for _, gain in first)
    equivalent_absorption = 126 * -np.log(0.8)
    decay_expected = 0.1611 * 90 / equivalent_absorption
    decay_error = abs(eyring(room)[0] / decay_expected - 1)
    assert arrival_error / SOUND_SPEED < 1e-10 and gain_error < 1e-10
    assert decay_error < 0.002

    rng = np.random.default_rng(305)
    reference = [0.30, 0.75, 1.50]
    estimates = []
    times = np.arange(3 * RATE) / RATE
    ramp = np.ones_like(times)
    ramp[:RATE // 100] = np.linspace(1.5, 1, RATE // 100)
    for seconds in reference:
        tail = ramp * rng.normal(size=len(times)) * 10**(-3 * times / seconds)
        wave = np.r_[np.zeros(RATE // 5), tail] + rng.normal(0, 2e-5, len(tail) + RATE // 5)
        wave *= 0.7 / np.max(np.abs(wave))
        estimates.append(preprocess(wave, RATE).classical_seconds)
    errors = np.abs(np.asarray(estimates) / reference - 1)
    assert errors.max() < 0.12
    return {'max_path_arrival_error_ms': 1000 * arrival_error / SOUND_SPEED,
            'max_first_reflection_pressure_error': gain_error,
            'eyring_reference_s': decay_expected, 'eyring_relative_error': decay_error,
            'known_decay_s': reference, 'estimated_decay_s': estimates,
            'max_relative_decay_error': float(errors.max())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=argparse.FileType('w'))
    args = parser.parse_args()
    result = json.dumps(check(), indent=2)
    print(result)
    if args.out:
        args.out.write(result + '\n')

