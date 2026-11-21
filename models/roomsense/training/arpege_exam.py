"""Freeze and score RoomSense on the documented ARPEGE handclap recordings."""
import argparse
from contextlib import ExitStack, contextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import zipfile
from tempfile import mkstemp

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

import common
from _paths import MODEL, SRC, work
from data import MANIFEST, reference_measurement
from exam import load_lock, same_results, score, tail_fit, weight_digest
from pipeline import RATE, preprocess, read_wav, resample
from roomsense import RoomSense

PLAN = Path(SRC) / 'arpege_exam_plan.json'
WEIGHTS = Path(MODEL) / 'roomsense_weights.npz'
RELEASE = Path(SRC) / 'release_lock.json'
FROZEN = Path(work('logs', 'arpege_frozen.json'))
RESULT = Path(work('logs', 'arpege_exam.json'))


def sha(path):
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def new_file(path):
    # Publish a complete result atomically and refuse to overwrite a freeze.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = mkstemp(dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        path.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_once(path, value):
    with new_file(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def planned_pairs(plan):
    for room in plan['rooms']:
        dimensions = room['envelope_dimensions_m']
        for pair in room['pairs']:
            yield dict(pair, room=room['id'], case=f"{room['id']}/{pair['position']}",
                       volume=float(np.prod(dimensions)) if dimensions else None)


def isolate_clap(samples, rate):
    """Identify one loud, temporally isolated transient without model or sweep readings."""
    hop = max(1, round(rate * 0.005))
    count = len(samples) // hop
    if not count or not np.isfinite(samples).all():
        return None
    frames = samples[:count * hop].reshape(count, hop)
    power = np.mean((frames - np.median(samples))**2, axis=1)
    peak = float(power.max())
    threshold = max(np.median(power) * 10**1.5, peak * 0.04, 1e-12)
    peaks, _ = find_peaks(np.pad(power, 1), height=threshold, prominence=0.07 * peak,
                          distance=max(1, round(0.24 * rate / hop)))
    peaks -= 1
    windows = []
    for i, frame in enumerate(peaks):
        center = int(frame * hop + hop // 2)
        start = max(0, center - round(0.18 * rate))
        end = min(len(samples), center + round(2.6 * rate))
        if i > 0:
            start = max(start, int(peaks[i - 1] * hop + hop // 2) + round(0.10 * rate))
        if i + 1 < len(peaks):
            end = min(end, int(peaks[i + 1] * hop + hop // 2) - round(0.05 * rate))
        if center - start >= round(0.075 * rate) and end - center >= round(0.40 * rate):
            windows.append((float(power[frame]), start, end))
    if not windows:
        return None
    _, start, end = max(windows, key=lambda item: item[0])
    return [start, end]


def channel(wave, index):
    if wave.ndim == 1:
        return wave
    if wave.ndim == 2 and wave.shape[1] > index:
        return wave[:, index]
    raise ValueError(f'WAV does not contain channel {index}.')


def freeze(root, plan):
    # A provider-prepared mono copy uses the same designated channel as the raw release.
    if FROZEN.exists():
        raise ValueError(f'Inputs are already frozen in {FROZEN}.')
    pairs, missing = [], []
    for pair in planned_pairs(plan):
        clap_path = root / pair['claps']
        sweep_path = root / pair['sweep_response']
        absent = [str(path.relative_to(root)) for path in (clap_path, sweep_path) if not path.is_file()]
        if absent:
            missing.append({'case': pair['case'], 'missing': absent})
            continue
        wave, rate = read_wav(clap_path)
        samples = channel(wave, plan['channel_zero_based'])
        pairs.append(dict(pair, claps_sha256=sha(clap_path), sweep_sha256=sha(sweep_path),
                          sample_rate_hz=int(rate), clap_window=isolate_clap(samples, rate)))
    if not pairs:
        choices = missing[0]['missing'] if missing else []
        raise ValueError(f'No complete ARPEGE pairs in {root}. Expected files such as {choices}.')
    if len({r['claps_sha256'] for r in pairs}) != len(pairs):
        raise ValueError('Duplicate clap WAV content in the candidate pairs.')
    frozen = dict(protocol=plan['protocol'], plan_sha256=sha(PLAN), release_lock_sha256=sha(RELEASE),
                  rooms=sorted({r['room'] for r in pairs}), pairs=pairs, missing_pairs=missing)
    write_json_once(FROZEN, frozen)
    return frozen


def measure(root, frozen, channel_index):
    rows, missing_truth, abstentions = [], [], {}
    for pair in frozen['pairs']:
        clap_path = root / pair['claps']
        sweep_path = root / pair['sweep_response']
        if sha(clap_path) != pair['claps_sha256'] or sha(sweep_path) != pair['sweep_sha256']:
            raise ValueError(f"Audio differs from the frozen files for {pair['case']}.")
        sweep, sweep_rate = read_wav(sweep_path)
        response = resample(np.asarray(channel(sweep, channel_index), dtype=float), sweep_rate)
        reference = None
        if response.size and np.isfinite(response).all() and np.any(response):
            start = max(0, int(np.argmax(np.abs(response))) - RATE // 100)
            response = response[start:start + 3 * RATE]
            if response.size >= RATE // 20:
                reference = reference_measurement(response)
        if reference is None or not np.isfinite(reference):
            missing_truth.append(dict(case=pair['case'], room=pair['room'],
                                      reason='The independent sweep lacks a usable band-decay reference.'))
            continue
        row = dict(case=pair['case'], room=pair['room'], snr=None, seconds=reference,
                   volume=pair['volume'], features=None, t20=None, tail=None)
        if pair['clap_window'] is None:
            abstentions[pair['case']] = 'No isolated transient with a recorded lead-in and decay.'
        else:
            wave, rate = read_wav(clap_path)
            start, end = pair['clap_window']
            clip = channel(wave, channel_index)[start:end]
            try:
                features = preprocess(clip, rate)
            except ValueError as error:
                abstentions[pair['case']] = str(error)
            else:
                # The secondary classical fit uses 16 kHz, the same DC removal,
                # resampler and onset as the network frontend.
                clip16 = resample(clip - clip.mean(), rate)
                row.update(features=features, t20=features.classical_seconds,
                           tail=tail_fit(clip16, features.onset_seconds))
        rows.append(row)
    return rows, missing_truth, abstentions


def evaluate(root, plan, *, reproduce=False):
    if RESULT.exists() != reproduce:
        raise ValueError('Use --reproduce only after scoring the frozen audio once.')
    frozen = json.loads(FROZEN.read_text())
    if frozen['plan_sha256'] != sha(PLAN) or frozen['release_lock_sha256'] != sha(RELEASE):
        raise ValueError('The audio freeze does not match the current protocol and release lock.')
    lock = load_lock(WEIGHTS, allow_published=True)
    model = RoomSense(WEIGHTS)
    if lock != json.loads(RELEASE.read_text()) or lock['split_sha256'] != weight_digest(MANIFEST):
        raise ValueError('The original development split or release lock has changed.')
    rows, missing_truth, abstentions = measure(root, frozen, plan['channel_zero_based'])
    result = score(rows, lock['baselines'], model.weights, model.meta['widen'])
    result.pop('per_snr')
    scored_rooms = sorted({r['room'] for r in rows})
    result.update(protocol=plan['protocol'], freeze_sha256=sha(FROZEN), weights_sha256=lock['weights_sha256'],
                  room_count=len(scored_rooms), scored_rooms=scored_rooms,
                  target_room_count=3, room_target_met=len(scored_rooms) >= 3,
                  missing_pairs=frozen['missing_pairs'], unavailable_reference=missing_truth,
                  input_abstentions=abstentions)
    if reproduce:
        if not same_results(result, json.loads(RESULT.read_text())):
            raise ValueError('The recorded-clap exam did not reproduce.')
    else:
        write_json_once(RESULT, result)
    return result


def archive_wavs(archive):
    """Index EM32 recordings by the basename used in the annotations."""
    matches = {}
    for member in archive.namelist():
        path = PurePosixPath(member)
        if path.parent.name != 'EM32' or path.suffix.lower() != '.wav':
            continue
        stem, _, take = path.stem.rpartition('_')
        key, number = (stem, int(take)) if take.isascii() and take.isdigit() else (path.stem, -1)
        matches.setdefault(key, []).append((number, member))
    index = {}
    for key, candidates in matches.items():
        candidates.sort()
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise ValueError(f'Ambiguous EM32 WAV matching {key} in {archive.filename}.')
        index[key] = candidates[0][1]
    return index


def pack(root, plan, output, rooms=None):
    """Save a compact transfer archive using the exact samples of EM32 channel 0."""
    rows = [pair for pair in planned_pairs(plan) if not rooms or pair['room'].split('/')[-1] in rooms]
    with ExitStack() as stack:
        sources = {}
        for name in ('ARPEGE_01.2026_claps', *sorted({r['room'].split('/')[-1] + '_RIR' for r in rows})):
            path = root / f'{name}.zip'
            if path.is_file():
                archive = stack.enter_context(zipfile.ZipFile(path))
                sources[name] = archive, archive_wavs(archive)

        def locate(name, archive_name):
            path = root / name
            if path.is_file():
                return lambda: path.open('rb')
            if archive_name in sources:
                archive, index = sources[archive_name]
                member = index.get(PurePosixPath(name).stem)
                if member is not None:
                    return lambda: archive.open(member)
            return None

        count = 0
        with new_file(output) as temporary:
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as transfer:
                for pair in rows:
                    room = pair['room'].split('/')[-1]
                    clap = locate(pair['claps'], 'ARPEGE_01.2026_claps')
                    sweep = locate(pair['sweep_response'], f'{room}_RIR')
                    if clap is None or sweep is None:
                        continue
                    for name, open_input in ((pair['claps'], clap), (pair['sweep_response'], sweep)):
                        with open_input() as stream:
                            rate, wave = wavfile.read(stream)
                        buf = io.BytesIO()
                        wavfile.write(buf, rate, np.ascontiguousarray(channel(wave, plan['channel_zero_based'])))
                        transfer.writestr(name, buf.getvalue())
                    count += 1
            if not count:
                raise ValueError(f'No complete ARPEGE pairs in {root}.')
    print(f'Packed {count} pairs into {output} ({output.stat().st_size / 2**20:.1f} MiB).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('files', help='List required archive-relative WAV paths')
    for command in ('pack', 'freeze', 'score'):
        cmd = commands.add_parser(command)
        cmd.add_argument('root', type=Path, help='Directory containing ARPEGE_01.2026/ or, for pack, the Zenodo ZIPs')
        if command == 'pack':
            cmd.add_argument('output', type=Path)
            cmd.add_argument('--room', action='append', choices=[f'room{i}' for i in range(1, 7)],
                             help='Select a room. Repeat to include several rooms.')
        if command == 'score':
            cmd.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if args.command == 'files':
        print('\n'.join(pair[key] for pair in planned_pairs(plan) for key in ('claps', 'sweep_response')))
    elif args.command == 'pack':
        pack(args.root, plan, args.output, args.room)
    elif args.command == 'freeze':
        frozen = freeze(args.root, plan)
        print(f"Froze {len(frozen['pairs'])} pairs from {len(frozen['rooms'])} rooms in {FROZEN}.")
    else:
        result = evaluate(args.root, plan, reproduce=args.reproduce)
        print(json.dumps({'room_count': result['room_count'], **result['metrics']}, indent=2))


if __name__ == '__main__':
    main()
