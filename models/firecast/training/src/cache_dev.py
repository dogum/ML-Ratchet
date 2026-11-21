"""Cache the DEV growth fire-days (raw inputs on B's crops) -> out/dev_days.pkz (list of compressed records)."""
import pickle
import zlib
import time
from common import *  # noqa
import realdays as RDY


def main():
    fd_ = fd()
    ids = list(fd_.fires('dev').Event_ID)
    out = []
    t0 = time.time()
    for fid in ids:
        for d in RDY.fire_days(fid, with_land=True):
            out.append(zlib.compress(pickle.dumps(d, protocol=4), 3))
        print(fid, len(out), f'{time.time() - t0:.0f}s', flush=True)
    pickle.dump(out, open(f'{OUT}/dev_days.pkz', 'wb'))
    print('DONE', len(out))


if __name__ == '__main__':
    main()
