"""Export the seeded Surrey exam examples using the already locked exam."""
import json
from pathlib import Path
import zipfile

import numpy as np

import common
from _paths import MODEL
from data import fetch, measurements
from exam import EXAM_PATH


def main():
    exam = json.loads(EXAM_PATH.read_text())
    case_ids = {r['case'] for r in exam['cases'] if r['accepted']}
    by_room = {}
    for row in measurements('exam'):
        if row['source'] == 'surrey' and row['case'] in case_ids:
            by_room.setdefault(row['room'], []).append(row)
    rng = np.random.default_rng(9311)
    out = Path(MODEL, 'examples')
    examples = []
    for room in sorted(by_room):
        for snr in (27, 44):
            choices = [r for r in by_room[room] if r['snr'] == snr]
            selected = choices[int(rng.integers(len(choices)))]
            name = f'room-{room.split("/")[-1].lower()}-{snr}db'
            samples = selected['wave']
            np.savez_compressed(out / f'{name}.npz', audio=samples)
            examples.append(dict(id=name, case=selected['case'], description=f'Surrey {room[-1]} · {snr} dB',
                                 rt60_s=selected['seconds'], volume_m3=selected['volume'], snr_db=snr,
                                 background=selected['background']))
    (out / 'truth.json').write_text(json.dumps(examples, indent=2) + '\n')
    with zipfile.ZipFile(fetch('surrey')) as archive:
        license_file = next(p for p in archive.namelist() if p.endswith('/LICENSE'))
        (out / 'LICENSE-Surrey.txt').write_bytes(archive.read(license_file))
    print('Examples:', ', '.join(r['id'] for r in examples))


if __name__ == '__main__':
    main()
