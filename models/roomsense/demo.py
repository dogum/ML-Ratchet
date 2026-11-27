"""Read reproducible Surrey exam clips or a local mono clap WAV."""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline import RATE
from roomsense import RoomSense


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav', nargs='?', help='A local WAV with quiet lead-in and one clap')
    args = parser.parse_args()
    model = RoomSense()
    if args.wav:
        try:
            print(json.dumps(model.read_file(args.wav), indent=2))
        except ValueError as error:
            parser.error(str(error))
        return
    directory = Path(__file__).with_name('examples')
    print('Synthetic claps with measured Surrey room responses')
    for row in json.loads((directory / 'truth.json').read_text()):
        with np.load(directory / f'{row["id"]}.npz', allow_pickle=False) as data:
            result = model.read(data['audio'], RATE)
        lo, hi = result['rt60_interval_s']
        classical = result['classical_rt60_s']
        t20 = f'{classical:.2f} s' if classical is not None else 'unavailable'
        print(f'{row["id"]} | RT60 {result["rt60_s"]:.2f} s [{lo:.2f}, {hi:.2f}]'
              f' | sweep {row["rt60_s"]:.2f} s | T20 {t20}'
              f' | volume {result["volume_m3"]:.0f} m³ (experimental), reference {row["volume_m3"]:.0f} m³')


if __name__ == '__main__':
    main()
