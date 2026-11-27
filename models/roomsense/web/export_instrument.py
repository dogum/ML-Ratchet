"""Export frozen examples and raw-audio Python parity cases for the website."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.io import wavfile

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack
from pipeline import RATE, preprocess
from roomsense import RoomSense, forward


def export():
    model = RoomSense()
    weights = roundtrip(model.weights, 'f16')
    out = instrument_dir('roomsense')
    write_pack(out / 'weights', model.weights, dtype='f16', meta=model.meta)
    rows = json.loads((MODEL / 'examples' / 'truth.json').read_text())
    waves, meta, cases = {}, [], []
    for i, row in enumerate(rows):
        with np.load(MODEL / 'examples' / f'{row["id"]}.npz', allow_pickle=False) as data:
            wave = data['audio']
        waves[f'wave_{i}'] = wave
        audio_name = f'{row["id"]}.wav'
        wavfile.write(out / audio_name, RATE, np.rint(wave.astype(float) * 32767).astype(np.int16))
        meta.append(dict(label=row['description'], filename=audio_name, rt60_s=row['rt60_s'],
                         volume_m3=row['volume_m3']))
        at_48k = np.interp(np.arange(3 * len(wave)) / 3, np.arange(len(wave)), wave).astype(np.float32)
        for variant, audio, rate in (('raw', wave, RATE), ('short-tail', wave[:int(0.9 * RATE)], RATE),
                                     ('resample48k', at_48k, 48000)):
            f = preprocess(audio, rate)
            prediction = forward(weights, f.curves[None], f.context[None])[0].tolist()
            prediction += [f.classical_seconds or -1, f.onset_seconds, f.snr_db]
            prediction += [v or -1 for v in f.band_seconds]
            cases.append(dict(example=i, variant=variant, outputs=prediction))
    write_pack(out / 'examples', waves, dtype='f32', meta=meta)
    write_json(out / 'reference.json', {'cases': cases, 'tolerance': 2e-5})
    for source in ('LICENSE-Surrey.txt', 'ATTRIBUTION.md'):
        shutil.copyfile(MODEL / 'examples' / source, out / source)
    print(f'Wrote {len(rows)} recordings and {len(cases)} parity cases to {out}')


if __name__ == '__main__':
    export()
