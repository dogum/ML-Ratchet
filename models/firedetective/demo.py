"""FireDetective demo: four real EXAM fires (a fixed seeded draw within strata, not hand-picked), from raw
observations (FEDS burned-area masks, VIIRS detections, gridMET, LANDFIRE) to the inferred drivers, beside the
truths it was graded on. Runs in a second or two with numpy + scipy.

    python3 demo.py
"""
import os, sys, glob, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline
import firedetective as fdet


def load_fire(fn):
    z = np.load(fn)
    shape = tuple(z['shape'])
    masks = np.unpackbits(z['masks'], axis=-1)[..., :shape[2]].astype(bool)
    meta = json.loads(str(z['meta']))
    dets = [dict(row=np.array(d['row']), col=np.array(d['col']), frp=np.array(d['frp'])) for d in meta['dets']]
    layers = {k: z[k] for k in ('fuel', 'cc', 'ch', 'cbh', 'cbd', 'slope', 'elev')}
    layers['fuel'] = layers['fuel'].astype(int)
    return dict(masks=masks, dets=dets, day=meta['day'], wx=meta['wx'], layers=layers, doy=meta['doy'],
                gamma=meta['gamma']), meta


def fmt(x, nd=2):
    return '  n/a' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


if __name__ == '__main__':
    t0 = time.time()
    model = fdet.load()
    files = sorted(glob.glob(os.path.join(HERE, 'examples', '*.npz')))
    fires, metas = zip(*[load_fire(f) for f in files])
    out = pipeline.run(list(fires), model)
    for i, m in enumerate(metas):
        A = fires[i]['masks'].sum(axis=(1, 2)) * 0.0324
        print(f"\n{m['name']} ({m['year']}, {m['region']}, {m['stratum']}): {len(fires[i]['masks'])} half-day steps, "
              f"burned {A[0]:.1f} -> {A[-1]:.1f} km2")
        T, C = m['truth'], m['classical']
        print(f"  high-severity fraction  predicted {fmt(out['hs'][i])} +- {fmt(out['hs_sd'][i])}   MTBS truth {fmt(T['hs'])}"
              f"   (Van Wagner crown potential {fmt(C['cfp'])}, ERC max {fmt(C['erc_max'], 0)})")
        print(f"  wind (inferred blind)   {fmt(out['wind'][i], 1)} m/s [{fmt(out['wind_lo'][i], 1)}-{fmt(out['wind_hi'][i], 1)}]"
              f"   gridMET truth {fmt(T['wind'], 1)}   (Rothermel back-calc {fmt(C['wind_bc'], 1)})")
        print(f"  live fuel moisture      {fmt(out['lfmc'][i], 0)} +- {fmt(out['lfmc_sd'][i], 0)} %   Globe-LFMC truth {fmt(T['lfmc'], 0)}")
        print(f"  simulator-only reads    crown fraction {fmt(out['crown'][i])}, ROS factor {fmt(out['ros_adj'][i])}, "
              f"1-h moisture {fmt(out['m1'][i], 1)} %, P(spotting) {fmt(out['p_spot'][i])}")
    print(f'\n{len(files)} fires in {time.time() - t0:.2f} s')
