"""Fetch verified public files and construct the frozen dev and exam recordings."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile
from tempfile import NamedTemporaryFile

import numpy as np
from scipy import signal
from scipy.io import wavfile

import common
from _paths import SRC, work
from pipeline import BANDS, RATE, read_wav, resample
from simulator import clap, record

MANIFEST = Path(SRC) / 'split.json'


def manifest():
    return json.loads(MANIFEST.read_text())


def download(name, url, digest):
    path = Path(work('raw', name, mkdir=True))
    if not path.exists():
        with urlopen(Request(url, headers={'User-Agent': 'ML-Ratchet RoomSense'}), timeout=60) as response:
            data = response.read()
        if sha256(data).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch downloading {name}')
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(data)
        Path(temporary.name).replace(path)
    if sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Checksum mismatch in {path}')
    return path


def fetch(source):
    cfg = manifest()
    if source == 'noise':
        for row in cfg['noise']:
            download(f'esc10/{row["file"]}', cfg['esc_url'] + row['file'], row['sha256'])
    else:
        row = cfg['archives'][source]
        return download(f'{source}-pinned.zip', row['url'], row['sha256'])


def noise_pool(role, *, filenames=False):
    cfg = manifest()
    pool, names = [], []
    for row in cfg['noise']:
        if row['role'] == role:
            path = download(f'esc10/{row["file"]}', cfg['esc_url'] + row['file'], row['sha256'])
            samples, rate = read_wav(path)
            pool.append(resample(np.asarray(samples, dtype=float), rate))
            names.append(row['file'])
    if not pool:
        raise ValueError(f'Unknown or empty background split {role}')
    return (pool, names) if filenames else pool


def measured_responses(role):
    """Yield stable case IDs and one fixed channel per source position."""
    cfg = manifest()
    for source in ('surrey', 'kitamura'):
        with zipfile.ZipFile(fetch(source)) as archive:
            root = archive.namelist()[0].split('/')[0]
            for row in cfg['rooms']:
                if row['source'] != source or row['role'] != role:
                    continue
                room = row['name']
                if source == 'surrey':
                    import h5py
                    member = f'{root}/UniS_Room_{room}_BRIR_16k.sofa'
                    with h5py.File(io.BytesIO(archive.read(member))) as sofa:
                        for index in row['positions']:
                            yield row, f'position-{index:02d}', sofa['Data.IR'][index, 0, :], RATE
                else:
                    for position in row['positions']:
                        member = f'{root}/asynchMulchiArray/impulse_dataset/{room}_{position}_target_ir_1.wav'
                        rate, response = wavfile.read(io.BytesIO(archive.read(member)))
                        yield row, position, response.astype(float), rate


def reference_measurement(response):
    """RT60 from the high-SNR sweep IR using full-rate octave-band integration.

    The clap estimator has to find onset and noise from the noisy input. Here
    the sweep is the reference and a fourth-order bandpass and sample-grid
    fit give a comparable definition on both corpora.
    """
    results = []
    for frequency in BANDS:
        sos = signal.butter(4, [frequency / np.sqrt(2), frequency * np.sqrt(2)],
                            'bandpass', fs=RATE, output='sos')
        power = signal.sosfilt(sos, response)**2
        background = np.median(power[-min(len(power) // 8, RATE // 6):]) / 0.4549364
        edc = np.maximum(1e-9, np.cumsum((power - background)[::-1])[::-1] / power.sum())
        db = 10 * np.log10(edc)
        indices = np.flatnonzero((db <= -5) & (db >= -25) & (np.arange(len(db)) < len(db) - RATE // 100))
        if len(indices) < RATE // 100 or np.min(db[indices]) > -24:
            continue
        slope, _ = np.polyfit(indices / RATE, db[indices], 1)
        if slope < 0:
            results.append(float(-60 / slope))
    return float(np.median(results)) if len(results) >= 2 else None


def measurements(role):
    if role not in ('dev', 'exam'):
        raise ValueError('Choose dev or exam.')
    pool, names = noise_pool(role, filenames=True)
    for room, position, response, sample_rate in measured_responses(role):
        case = f'{room["source"]}/{room["name"]}/{position}'
        response = resample(response, sample_rate)
        start = max(0, int(np.argmax(np.abs(response))) - RATE // 100)
        response = response[start:start + 3 * RATE]
        reference = reference_measurement(response)
        if reference is None:
            raise ValueError(f'Insufficient reference sweep decay for {case}')
        dimensions = room.get('dimensions_m')
        for snr in (27, 44):
            seed = int.from_bytes(sha256(f'{case}:{snr}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            excitation = clap(rng)
            noise_index = rng.integers(len(pool))
            wave = record(response, excitation, pool[noise_index], rng, snr, instrument=False)
            yield dict(case=f'{case}/{snr}dB', room=case.rsplit('/', 1)[0], source=room['source'],
                       seconds=reference, volume=float(np.prod(dimensions)) if dimensions else None,
                       snr=snr, wave=wave, background=names[noise_index])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='+', choices=['noise', 'surrey', 'kitamura', 'claprir'])
    for name in parser.parse_args().source:
        print(name, fetch(name))
