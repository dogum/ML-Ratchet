"""Simulation bank reader: per-fire compressed records -> training crops / full-grid validation inputs."""
import glob
from scipy import ndimage
import pickle
import zlib
from common import *  # noqa
import features as FT

N = 256


def unpack(b, n=N):
    return np.unpackbits(b)[:n * n].reshape(n, n).astype(bool)


def det_dict(det, step):
    q = det[det.step == step]
    return dict(row=q.row.values.astype(float), col=q.col.values.astype(float), ps=q.pix_km_scan.values,
                pt=q.pix_km_track.values, frp=q.frp_mw.values)


class Bank:
    def __init__(self, ver, split, max_fires=None):
        files = [f for v in ver.split('+') for f in sorted(glob.glob(f'{BANK}/{v}/{split}/shard_*.pkz'))]   # 'p0+p1' mixes banks
        self.blobs = []
        for f in files:
            self.blobs += pickle.load(open(f, 'rb'))
            if max_fires and len(self.blobs) >= max_fires:
                break
        if max_fires:
            self.blobs = self.blobs[:max_fires]
        self.index = []
        self._cache = {}
        for i, b in enumerate(self.blobs):
            r = self.rec(i)
            for j in range(len(r['samples'])):
                self.index.append((i, j))
        self._cache = {}

    def rec(self, i):
        if i not in self._cache:
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = pickle.loads(zlib.decompress(self.blobs[i]))
        return self._cache[i]

    def __len__(self):
        return len(self.index)

    def raw(self, n):
        """Raw fire-day (full 256 grid) for index n."""
        i, j = self.index[n]
        r = self.rec(i)
        s = r['samples'][j]
        k = s['k']
        keys = r['keys']
        now = unpack(r['masks'][k])
        prev = unpack(r['masks'][s['kprev']]) if s['kprev'] is not None else None
        fl = unpack(r['lines'][k])
        nxt = unpack(r['masks'][s['k2']]) | now
        pm = det_dict(r['det'], k)
        am = det_dict(r['det'], k - 1) if k >= 1 and keys[k - 1] == (keys[k][0], 'AM') else None
        return dict(now=now, prev=prev, fl=fl, nxt=nxt, pix_pm=pm, pix_am=am, land=r['land'], wx=s['wx'],
                    has_prev=prev is not None, first=s['first'], truth_next=unpack(s['truth_next']),
                    truth_now=unpack(s['truth_now']), params=r['params'], fire=i, kind=r['kind'])

    def train_item(self, n, rng, crop=128, land_p=0.5):
        d = self.raw(n)
        now = d['now']
        bnd = now & ~ndimage.binary_erosion(now, border_value=0)
        u = rng.random()
        src = d['fl'] if (u < 0.6 and d['fl'].any()) else bnd
        if u > 0.9:
            src = ndimage.binary_dilation(now, iterations=40)
        rr, cc = np.nonzero(src)
        q = int(rng.integers(len(rr)))
        cr = int(np.clip(rr[q] + rng.integers(-20, 21) - crop // 2, 0, N - crop))
        cc0 = int(np.clip(cc[q] + rng.integers(-20, 21) - crop // 2, 0, N - crop))
        box = (cr, cr + crop, cc0, cc0 + crop)
        op = int(rng.integers(8))
        land = d['land'] if rng.random() < land_p else None
        X, s = FT.make_inputs(now, d['prev'], d['fl'], d['pix_pm'], d['pix_am'], land, d['wx'], d['has_prev'],
                              d['first'], crop=box, op=op)
        y = (d['nxt'] & ~now)[box[0]:box[1], box[2]:box[3]]
        m = (X[FT.SPATIAL.index('inwin')] > 0) & (X[0] == 0)
        if op:
            # y must follow the same dihedral op (masks only)
            F2, _ = FT.dihedral({'y': y.astype(np.float32)}, np.zeros(16, np.float32), op)
            y = F2['y'] > 0.5
        return X, s, y.astype(np.float32), m.astype(np.float32)

    def full_item(self, n, land=False):
        d = self.raw(n)
        X, s = FT.make_inputs(d['now'], d['prev'], d['fl'], d['pix_pm'], d['pix_am'], d['land'] if land else None,
                              d['wx'], d['has_prev'], d['first'])
        return X, s, d
