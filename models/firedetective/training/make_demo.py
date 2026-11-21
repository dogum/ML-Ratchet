"""After the exam: package raw inputs of a few exam fires for release/demo.py (fixed seeded draw within strata:
2 forest fires with MTBS severity from the main period, 1 non-forest main fire, 1 recent fire).
Raw inputs = FEDS masks (180 m), VIIRS detections, gridMET rows, LANDFIRE layers, cropped to the fire."""
from common import *
import importlib, pickle
import firedata as fd
import feats
import real as RL
from exam import exam_list, EXD

G = importlib.import_module('firedata.grid')
OUT = f'{ROOT}/release/demo_data'
os.makedirs(OUT, exist_ok=True)


def raw_inputs(fid):
    r = fd._row(fid)
    pr = fd.perimeters(fid, for_exam_scoring=True)
    fp = fd.firepix(fid, for_exam_scoring=True)
    g90 = fd.grid(fid); g180 = G.coarsen(g90, 2)
    x0, y0, x1, y1 = g180['bounds']
    k0 = int(np.argmax(pr.farea_km2.values > 0))
    pw = pr.iloc[k0:k0 + feats.T_STEPS].reset_index(drop=True)
    masks = np.maximum.accumulate(np.stack([G.burn([g], g180) for g in pw.geometry]), axis=0)
    L, _ = RL.landscape180(fid, int(r.year), g180, True, True)
    rr, cc = np.nonzero(masks[-1])
    m = 14
    r0, r1 = max(rr.min() - m, 0), min(rr.max() + m + 1, masks.shape[1])
    c0, c1 = max(cc.min() - m, 0), min(cc.max() + m + 1, masks.shape[2])
    gx, gy = to5070(fp.Lon.values.astype(float), fp.Lat.values.astype(float))
    dets = []
    for t in pw.t:
        d = fp[fp.t == t]
        dets.append(dict(col=((gx[d.index] - x0) / feats.RES - 0.5 - c0).round(3).tolist(),
                         row=((y1 - gy[d.index]) / feats.RES - 0.5 - r0).round(3).tolist(), frp=d.FRP.values.round(2).tolist()))
    W = fd.weather(fid, for_exam_scoring=True)['daily'].copy(); W.index = pd.DatetimeIndex(W.index).normalize()
    days = [RL.step_day(t) for t in pw.t]
    wx = RL.wx_rows(W, days)
    import firesim as fs
    meta = dict(fid=fid, name=r.Incid_Name, year=int(r.year), period=r.exam_period, region=r.region,
                steps=[str(t) for t in pw.t], day=[h == 'PM' for h in pw.half], dets=dets, wx=wx,
                doy=int(pd.Timestamp(pw.t.iloc[0]).dayofyear), gamma=fs.realdata.grid_convergence_deg(float(r.cx), float(r.cy)))
    lay = {k: np.asarray(L[k])[r0:r1, c0:c1] for k in ('fuel', 'cc', 'ch', 'cbh', 'cbd', 'slope', 'elev')}
    return masks[:, r0:r1, c0:c1], lay, meta


if __name__ == '__main__':
    recs = [pickle.load(open(f'{EXD}/{f}.pkl', 'rb')) for f in exam_list() if os.path.exists(f'{EXD}/{f}.pkl')]
    recs = [r for r in recs if r is not None and 'error' not in r]
    rs = np.random.default_rng(7)
    strata = [('forest_main', [r for r in recs if r['forest_frac'] >= 0.5 and r['period'] == 'main' and np.isfinite(r['hs'])], 2),
              ('nonforest_main', [r for r in recs if r['forest_frac'] < 0.5 and r['period'] == 'main'], 1),
              ('recent', [r for r in recs if r['period'] == 'recent'], 1)]
    chosen = []
    for nm, pool, k in strata:
        pool = sorted(pool, key=lambda r: r['fid'])
        for i in rs.choice(len(pool), k, replace=False):
            chosen.append((nm, pool[i]))
    for nm, rec in chosen:
        masks, lay, meta = raw_inputs(rec['fid'])
        meta.update(stratum=nm, truth=dict(hs=rec['hs'], wind=rec['wind'], lfmc=rec['lfmc']),
                    classical=dict(cfp=rec['base']['cfp'], erc_max=rec['base']['erc_max'], wind_bc=rec['base']['wind_bc']))
        np.savez_compressed(f'{OUT}/{rec["fid"]}.npz', masks=np.packbits(masks, axis=-1), shape=np.array(masks.shape),
                            meta=json.dumps(meta, default=float), **{k: v.astype(np.float32) for k, v in lay.items()})
        print(nm, rec['fid'], rec['name'], masks.shape)
