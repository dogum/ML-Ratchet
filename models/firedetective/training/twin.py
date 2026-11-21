"""Twin test on DEV fires: simulate each dev fire's twin (its LANDFIRE landscape at 180 m, its HRRR weather at the
fire, ignited from its first FEDS perimeter, dev-calibrated ROS factors from checks/replay_calibration.json), observe
it with the VIIRS emulator at the real FEDS step times and the FEDS emulator, and compare the model's input
features with the real ones. Physics variants: v1 (bank v1) and v2 (night fine-fuel recovery factor 1.4).

python3 twin.py [v1|v2]  -> out/twin_<v>.json
"""
from common import *
import importlib, warnings
import firesim as fs
import firedata as fd
import feats
from real import load_records, landscape180, wx_rows, step_day

warnings.filterwarnings('ignore')
G = importlib.import_module('firedata.grid')
RES = feats.RES
CAL = json.load(open(f'{CHECKS}/replay_calibration.json'))   # import: was <sandbox>/fire/checks/


def twin(rec, variant='v1', seed=0):
    fid = rec['fid']
    r = fd._row(fid)
    pr = fd.perimeters(fid)
    k0 = int(np.argmax(pr.farea_km2.values > 0))
    pw = pr.iloc[k0:k0 + rec['n_steps']].reset_index(drop=True)
    g180 = G.coarsen(fd.grid(fid), 2)
    x0, y0, x1, y1 = g180['bounds']
    L, _ = landscape180(fid, int(r.year), g180)
    W = fd.weather(fid)
    w = fs.realdata.weather_series(W, float(r.cx), float(r.cy))
    lon, lat = fs.realdata.lonlat_of(float(r.cx), float(r.cy))
    w = fs.realdata.fill_precip_from_daily(w, W['daily'], lon)
    md = fs.moisture.dead_moisture(w, method='nelson')
    w['m1h'], w['m10h'], w['m100h'] = md['m1h'], md['m10h'], md['m100h']
    if variant == 'v2':
        lh = (w.index.tz_convert(None) + pd.Timedelta(hours=lon / 15.0)).hour.to_numpy() + 0.5
        night = 0.5 * (1 - np.cos(2 * np.pi * (lh - 15.0) / 24.0))
        fac = 1.0 + 0.4 * night
        w['m1h'] = w['m1h'] * fac; w['m10h'] = w['m10h'] * np.sqrt(fac)
    ls = fs.Landscape(meta=dict(crs='EPSG:5070', res_m=RES, transform=(x0, RES, 0.0, y1, 0.0, -RES), nodata=-9999,
                                lat=lat, lon=lon), res_m=RES, **L)
    m0 = G.burn([pw.geometry.iloc[0]], g180)
    t_utc = pd.DatetimeIndex(pw.t_utc).tz_localize('UTC')
    start, end = t_utc[0], t_utc[-1] + pd.Timedelta(hours=1)
    params = dict(ros_adjust=dict(CAL['factors']))
    if variant == 'lw':
        params['lw'] = 'anderson1983'
    if variant == 'waf':
        params['waf'] = 0.4
    out = fs.simulate(ls, w, {'mask': m0}, start, end, dt='12h', crown='van_wagner', spotting=False, moisture='given',
                      params=params, rng=seed, return_perimeters=False)
    ops = [(t, 'S-NPP', 0.0) for t in t_utc]
    ob = fs.observe(out, 'viirs', overpasses=ops, rng=seed, lat=lat, lon=lon)
    det = ob['detections']
    det = det[det.true_fire] if len(det) else det
    em_in, dets = [], []
    for t in t_utc:
        d = det[det.time == t] if len(det) else det
        e = dict(x=d.x.values if len(d) else np.zeros(0), y=d.y.values if len(d) else np.zeros(0),
                 ds=d.pix_km_scan.values if len(d) else np.zeros(0), dt=d.pix_km_track.values if len(d) else np.zeros(0))
        em_in.append(e)
        dets.append(dict(col=(e['x'] - x0) / RES - 0.5, row=(y1 - e['y']) / RES - 0.5,
                         frp=d.frp_mw.values if len(d) else np.zeros(0)))
    masks = feats.feds_emulate(m0.shape, x0, y1, RES, em_in)
    masks = masks | m0[None]
    wx = rec['wx']
    seq = feats.seq_features(masks, dets, np.array(pw.half == 'PM'), wx, gamma_deg=rec['gamma'])
    return seq


def summ(S):
    m = S[:, 0] > 0
    g = m & (S[:, 3] > 0.01)
    day = m & (S[:, 1] > 0) & (S[:, 5] > 0); nig = m & (S[:, 1] == 0) & (S[:, 5] > 0)
    return dict(final_logA=float(S[m, 2][-1]), frac_growth=float(g.sum() / m.sum()), mean_dlogA=float(S[m, 3].mean()),
                logN=float(S[m, 5].mean()), frp_day=float(S[day, 7].mean()) if day.any() else np.nan,
                frp_night=float(S[nig, 7].mean()) if nig.any() else np.nan, elong=float(S[m, 9].mean()),
                night_dlogA=float(S[m & (S[:, 1] == 0), 3].mean()) if (m & (S[:, 1] == 0)).any() else np.nan,
                day_dlogA=float(S[m & (S[:, 1] > 0), 3].mean()) if (m & (S[:, 1] > 0)).any() else np.nan)


if __name__ == '__main__':
    v = sys.argv[1] if len(sys.argv) > 1 else 'v1'
    recs = [r for r in load_records(f'{REAL}/dev_records_package.pkl') if r is not None]
    res = {}
    for rec in recs:
        try:
            s = twin(rec, v)
        except Exception as e:
            print(rec['fid'], 'fail', repr(e)[:200], flush=True); continue
        res[rec['fid']] = dict(sim=summ(s), real=summ(rec['seq']), steps_sim=s[:, 3].tolist(), steps_real=rec['seq'][:, 3].tolist(),
                               valid=rec['seq'][:, 0].tolist())
        print(rec['fid'], {k: (round(res[rec['fid']]['sim'][k], 2), round(res[rec['fid']]['real'][k], 2)) for k in res[rec['fid']]['sim']}, flush=True)
    agg = {}
    for k in next(iter(res.values()))['sim']:
        a = np.array([x['sim'][k] for x in res.values()]); b = np.array([x['real'][k] for x in res.values()])
        ok = np.isfinite(a) & np.isfinite(b)
        agg[k] = dict(sim_median=float(np.median(a[ok])), real_median=float(np.median(b[ok])),
                      median_diff=float(np.median(a[ok] - b[ok])), corr=float(np.corrcoef(a[ok], b[ok])[0, 1]))
    from scipy.stats import spearmanr
    a = np.concatenate([np.array(x['steps_sim'])[np.array(x['valid']) > 0][1:] for x in res.values()])
    b = np.concatenate([np.array(x['steps_real'])[np.array(x['valid']) > 0][1:] for x in res.values()])
    agg['step_dlogA_spearman'] = dict(rho=float(spearmanr(a, b)[0]), n=int(len(a)))
    jdump(dict(variant=v, n=len(res), agg=agg, per_fire=res), f'{ROOT}/out/twin_{v}.json')
    for k, x in agg.items():
        print(k, {a: round(b, 3) for a, b in x.items()})
