"""Worker for a bank + features run (Windows-safe replacement of run_bank.sh): claims shards with lock files, so any
number of workers can run side by side. Frozen-set shards (900-904) first, then training shards in order.
usage: python runbank.py <bank_tag> <feat_tag> [first=0] [last=59] [--w3]
"""
import os
import sys

import common as C
import featurize as FZ
import simbank as SB


def claim(path):
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main(bank, feat, first=0, last=59, w3=False):
    d = os.path.join(C.ROOT, 'bank', bank)
    os.makedirs(d, exist_ok=True)
    for sid in list(range(900, 905)) + list(range(first, last + 1)):
        fn = os.path.join(d, f'shard_{sid:04d}.npz')
        if not claim(fn + '.lock'):
            continue
        SB.build_shard(bank, sid)
        FZ.featurize_shard(bank, feat, fn, 2, True, w3)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 0, int(a[3]) if len(a) > 3 else 59, '--w3' in sys.argv)
