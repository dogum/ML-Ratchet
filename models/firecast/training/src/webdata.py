"""Page data: the 12 seeded exam examples (exam.py examples) -> web/data.json (gzip+base64 rasters), the JS weights
pack, and JS-vs-numpy check inputs. Also exam / ratchet / twin / calibration tables for the page.
Usage: python3 webdata.py <rung>"""
import base64
import gzip
import pickle
from common import *  # noqa
import features as FT
import webpack


def gz(a):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()


def q8(v, lo, hi):
    return np.clip(np.round((v - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


def main(rung):
    ex = pickle.load(open(f'{OUT}/exam/examples_{rung}.pkl', 'rb'))
    E = []
    for r in ex:
        F = r['fields']
        H, W = r['now'].shape
        Hc, Wc = H // 16 * 16, W // 16 * 16
        sl = (slice(0, Hc), slice(0, Wc))
        ch = dict(
            masks=gz(np.packbits(np.stack([r['now'][sl], (F['grew'][sl] > 0), (F['fl'][sl] > 0), r['nxt'][sl],
                                           F['inwin'][sl] > 0]).astype(np.uint8), axis=None)),
            pix=gz(np.stack([q8(F['pix_pm'][sl], 0, 2.5), q8(F['pix_am'][sl], 0, 2.5)])),
            dist=gz(np.stack([q8(F['d_burn'][sl], 0, 2), q8(F['d_fl'][sl], 0, 2), q8(F['d_pix'][sl], 0, 2)])),
            off=gz(np.stack([np.round(F['off_e'][sl] * 1000 / 5).astype(np.int16),
                             np.round(F['off_n'][sl] * 1000 / 5).astype(np.int16)])),   # 5 m steps
            prob=gz(q8(r['prob'][sl], 0, 1)), flprev=gz(q8(r['flprev'][sl], 0, 1)))
        E.append(dict(meta=r['meta'], H=Hc, W=Wc, wx=[float(v) for v in r['wx']], sc=[float(v) for v in r['sc']], ch=ch))
    json.dump(E, open(f'{ROOT}/web/examples.json', 'w'))
    print('examples.json', os.path.getsize(f'{ROOT}/web/examples.json') / 1e6, 'MB')
    wfn = f'{ROOT}/release/firecast_weights.npz'   # a re-run's export, else the released weights
    pk = webpack.pack(wfn if os.path.exists(wfn) else f'{MODEL}/firecast_weights.npz')
    json.dump(pk, open(f'{ROOT}/web/weights.json', 'w'))
    print('weights.json', os.path.getsize(f'{ROOT}/web/weights.json') / 1e6, 'MB')


if __name__ == '__main__':
    main(sys.argv[1])
