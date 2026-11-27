"""Dev-only baselines, room-weighted metrics, and a once-scored measured exam."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

import numpy as np

import common
from _paths import work
from data import MANIFEST, measurements
from pipeline import BANDS, HOP, RATE, band_filter, frame_power, preprocess
from roomsense import RoomSense, forward
from mlratchet.calibrate import widen_factor

EXAM_PATH = Path(work('logs', 'measured_exam.json'))
LOCK_PATH = Path(work('logs', 'release_lock.json'))
PUBLISHED = Path(__file__).resolve().parent


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def room_mean(rows, key):
    averages = [np.mean([r[key] for r in group if r[key] is not None])
                for group in grouped([r for r in rows if r[key] is not None], 'room').values()]
    return float(np.mean(averages)) if averages else None


def tail_fit(wave, onset):
    """An additional classical fit to the late band-power envelope."""
    start = round(onset * RATE)
    results = []
    for frequency in BANDS:
        filtered = band_filter(wave, frequency)
        noise = np.mean(filtered[:start - HOP]**2)
        tail = filtered[start:]
        energy = frame_power(tail, HOP, len(tail) // HOP)
        smooth = np.convolve(energy, np.ones(3) / 3, mode='same')
        significant = (smooth > 10 * noise) & (smooth > smooth.max() * 1e-5)
        # Find a continuous fit after the direct/clap burst. Isolated late
        # noise spikes should not extend the fitting range.
        weak = ~significant
        stops = np.flatnonzero(weak[3:-2] & weak[4:-1] & weak[5:])
        end = 3 + int(stops[0]) if len(stops) else len(smooth) - 1
        indices = np.arange(3, end)
        if len(indices) < 7:
            continue
        x = indices * HOP / RATE
        db = 10 * np.log10(np.maximum(1e-20, smooth[indices] - noise))
        slope, offset = np.polyfit(x, db, 1)
        denom = np.sum((db - db.mean())**2)
        if slope >= -10 or denom <= 0 or np.ptp(db) < 12:
            continue
        fit = 1 - np.sum((db - (slope*x + offset))**2) / denom
        if fit >= 0.70 and 0.1 <= -60 / slope <= 6:
            results.append(-60 / slope)
    return float(np.median(results)) if len(results) >= 2 else None


def cases(role):
    seen = set()
    rows = []
    for r in measurements(role):
        if r['case'] in seen:
            raise ValueError(f"Duplicate case {r['case']}")
        seen.add(r['case'])
        try:
            r['features'] = preprocess(r['wave'], RATE)
        except ValueError as e:
            r.update(features=None, input_failure=str(e), t20=None, tail=None)
        else:
            r['t20'] = r['features'].classical_seconds
            r['tail'] = tail_fit(r['wave'], r['features'].onset_seconds)
        rows.append(r)
    return rows


def fit_baselines(dev):
    rooms = grouped(dev, 'room')
    seconds = float(np.median([np.median([r['seconds'] for r in rows]) for rows in rooms.values()]))
    known = [rows for rows in rooms.values() if rows[0]['volume'] is not None]
    volume = float(np.exp(np.median(np.log([rows[0]['volume'] for rows in known]))))
    x = np.log([np.median([r['t20'] or r['tail'] or seconds for r in rows]) for rows in known])
    y = np.log([rows[0]['volume'] for rows in known])
    x_center = float(x.mean())
    dx = x - x_center
    slope = np.clip(dx @ (y - y.mean()) / (dx @ dx + 0.04), 0, 3)
    return dict(seconds=seconds, volume=volume, volume_center=x_center,
                volume_intercept=float(y.mean()), volume_slope=float(slope))


def predictions(rows, weights):
    accepted = [r for r in rows if r['features'] is not None]
    if not accepted:
        return {}
    result = forward(weights, np.stack([r['features'].curves for r in accepted]),
                     np.stack([r['features'].context for r in accepted]))
    return {r['case']: p.astype(float) for r, p in zip(accepted, result)}


def calibrate(dev, weights):
    predicted = predictions(dev, weights)
    scores = [[], []]
    for r in dev:
        p = predicted.get(r['case'])
        if p is None:
            continue
        for i, truth in enumerate((r['seconds'], r['volume'])):
            if truth:
                scores[i].append((np.log(truth) - p[i], p[2 + i]))
    return [widen_factor(*np.asarray(matrix).T) for matrix in scores]


def score(rows, settings, weights=None, widen=(1., 1.)):
    predicted = predictions(rows, weights) if weights is not None else {}
    table = []
    for r in rows:
        p = predicted.get(r['case'])
        s, v = r['seconds'], r['volume']
        seconds = settings['seconds']
        fitted = r['t20'] or r['tail'] or seconds
        volume_rt = np.exp(settings['volume_intercept'] +
                           settings['volume_slope'] * (np.log(fitted) - settings['volume_center']))
        model = np.exp(p[:2]) if p is not None else (seconds, settings['volume'])
        table.append(dict(case=r['case'], room=r['room'], snr=r['snr'], seconds=s, volume=v,
                          t20=r['t20'], tail=r['tail'], accepted=r['features'] is not None,
                          rt60_pred=float(model[0]) if p is not None else None,
                          volume_pred=float(model[1]) if p is not None else None,
                          model_rt=abs(float(model[0]) - s),
                          t20_rt=abs((r['t20'] or seconds) - s),
                          combined_rt=abs(fitted - s), median_rt=abs(seconds - s),
                          model_volume=abs(float(np.log(model[1] / v))) if v else None,
                          decay_volume=abs(float(np.log(volume_rt / v))) if v else None,
                          median_volume=abs(float(np.log(settings['volume'] / v))) if v else None,
                          rt_covered=bool(abs(p[0] - np.log(s)) <= p[2] * widen[0]) if p is not None else None,
                          vol_covered=bool(abs(p[1] - np.log(v)) <= p[3] * widen[1])
                          if p is not None and v else None))
    keys = ['t20_rt', 'combined_rt', 'median_rt', 'decay_volume', 'median_volume']
    if weights is not None:
        keys += ['model_rt', 'model_volume', 'rt_covered', 'vol_covered']

    def summary(group):
        return dict(cases=len(group), accepted=sum(r['accepted'] for r in group),
                    t20_valid=sum(r['t20'] is not None for r in group),
                    combined_valid=sum(r['t20'] is not None or r['tail'] is not None for r in group),
                    volume_rooms=len({r['room'] for r in group if r['volume'] is not None}),
                    **({'matched_model_rt': room_mean([r for r in group if r['t20'] is not None], 'model_rt'),
                        'matched_t20_rt': room_mean([r for r in group if r['t20'] is not None], 't20_rt')}
                       if weights is not None else {}),
                    **{key: room_mean(group, key) for key in keys})

    return dict(metrics=summary(table), per_room={k: summary(v) for k, v in grouped(table, 'room').items()},
                per_snr={k: summary(v) for k, v in grouped(table, 'snr').items()}, cases=table)


def weight_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_lock(weights_path, *, allow_published=False):
    path = LOCK_PATH if LOCK_PATH.exists() or not allow_published else PUBLISHED / 'release_lock.json'
    lock = json.loads(path.read_text())
    if lock['weights_sha256'] != weight_digest(weights_path):
        raise ValueError('These weights do not match the locked release.')
    return lock


def same_results(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(same_results(v, expected[k]) for k, v in actual.items())
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(same_results(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, float):
        return isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)
    return actual == expected


def run_exam(weights_path, *, reproduce=False):
    model = RoomSense(weights_path)
    lock = load_lock(weights_path, allow_published=reproduce)
    if lock['split_sha256'] != weight_digest(MANIFEST):
        raise ValueError('The data manifest does not match the locked release.')
    if lock['rung'] != model.meta['rung']:
        raise ValueError('These weights do not match the locked release.')
    if EXAM_PATH.exists() and not reproduce:
        raise ValueError('Exam already scored. Use --reproduce to compare the locked run.')
    result = score(cases('exam'), lock['baselines'], model.weights, model.meta['widen'])
    result.update(locked_release=lock)
    result = json.loads(json.dumps(result))
    if reproduce:
        reference = EXAM_PATH if EXAM_PATH.exists() else PUBLISHED / 'measured_exam.json'
        if not same_results(result, json.loads(reference.read_text())):
            raise ValueError('The locked exam failed to reproduce the recorded scores.')
    else:
        EXAM_PATH.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exam', metavar='LOCKED_WEIGHTS')
    p.add_argument('--reproduce', action='store_true')
    args = p.parse_args()
    if args.exam:
        result = run_exam(args.exam, reproduce=args.reproduce)
    else:
        dev = cases('dev')
        settings = fit_baselines(dev)
        result = score(dev, settings)
        path = Path(work('logs', 'dev_baselines.json', mkdir=True))
        path.write_text(json.dumps(dict(settings=settings, **result), indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('metrics', 'per_room', 'per_snr')}, indent=2))

