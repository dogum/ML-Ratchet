"""Read the six published ClapRIR demonstration claps after release lock."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile

import common
from _paths import MODEL, work
from data import fetch
from exam import load_lock
from roomsense import RoomSense


def check():
    path = Path(MODEL, 'roomsense_weights.npz')
    load_lock(path, allow_published=True)
    model = RoomSense(path)
    rows = []
    with zipfile.ZipFile(fetch('claprir')) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith('/recorded_clap.wav'):
                continue
            rate, wave = wavfile.read(io.BytesIO(archive.read(member)))
            row = {'recording': member.split('/')[1]}
            try:
                r = model.read(wave, rate)
            except ValueError as error:
                row['input_failure'] = str(error)
            else:
                row['rt60_s'] = r['rt60_s']
                row['rt60_interval_s'] = r['rt60_interval_s']
            rows.append(row)
    return {'reference': 'No paired reference measurements. Input acceptance and repeatability only.',
            'rooms': 2, 'readings': sum('rt60_s' in row for row in rows), 'rows': rows}


if __name__ == '__main__':
    result = json.dumps(check(), indent=2)
    Path(work('logs', 'phone_claps.json', mkdir=True)).write_text(result + '\n')
    print(result)
