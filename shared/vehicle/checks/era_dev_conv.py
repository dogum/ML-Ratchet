"""Dev diagnostic (RoadSim v3.1 pass, item C): do the modern benchmark maps flatter older engines? Conventional gasoline
test cars of RoadData's DEV model families, by model-year band, simulated on FTP and HWY with the conventional specs of
checks/epa_dev_pool.py (same map rule, gear sets, road-load handling). Mild hybrids (Honda IMA, GM BAS) run on the
same conventional maps (gas_na_toyota_a25a below 3.0 L), so an era bias of those maps on dev conventional cars is
evidence for the mild twins too. Dev families only (the frozen family split); at most 3 test vehicles per family and
band, drawn with a fixed seed (never by results). Error = sim / EPA - 1 on fuel economy (+ = the sim burns less).

usage: python era_dev_conv.py [--options roadsim-physics-v2] [--per-band 150]
Writes checks/era_dev_conv.json.
"""
import os, sys, json, time, argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(VEH, 'roadsim')); sys.path.insert(0, VEH)
import _opts
from epa_procedures import ftp, hot_cycle, test_spec
import epa_dev_pool as P

BANDS = [(2008, 2011), (2012, 2015), (2016, 2017), (2018, 2026)]
SEED = 20261114


def records(per_band, per_family=3):
    import realdata as rd
    e = rd.epa_testcars()
    d = e[(e.split == 'dev') & (e.architecture == 'conv') & (e.fuel == 'gasoline') & e.FTP_mpg.notna() & e.HWY_mpg.notna()
          & e.A_lbf.notna() & e.C_lbf_mph2.notna() & e.rated_hp.notna() & e.displacement_L.notna() & e.nv_ratio.notna()
          & e.etw_lb.notna()]
    d = d.drop_duplicates(subset=['model_year', 'make', 'model', 'displacement_L', 'trans_type', 'gears', 'etw_lb', 'FTP_mpg'])
    rng = np.random.default_rng(SEED)
    out = []
    for a, b in BANDS:
        g = d[(d.model_year >= a) & (d.model_year <= b)]
        keep = []
        for fam, gg in g.groupby('family', sort=True):
            idx = list(gg.index)
            if len(idx) > per_family:
                idx = list(rng.choice(idx, per_family, replace=False))
            keep += idx
        keep = sorted(keep)
        if len(keep) > per_band:
            keep = sorted(rng.choice(keep, per_band, replace=False))
        for i in keep:
            r = d.loc[i]
            try:
                s = P.spec_of(r)
            except Exception:
                continue
            out.append(dict(row=str(i), band=f'{a}-{b}', my=int(r.model_year), family=r.family, spec=s,
                            map=s['engine']['bsfc_map'], epa={'FTP': float(r.FTP_mpg), 'HWY': float(r.HWY_mpg)}))
    return out


def one(job):
    rec, opt = job
    _opts.set(opt)
    s = test_spec(rec['spec'])
    o = dict(row=rec['row'], band=rec['band'], my=rec['my'], family=rec['family'], map=rec['map'], err={})
    try:
        o['err']['FTP'] = round(100 * (ftp(s)['FE'] / rec['epa']['FTP'] - 1), 2)
        o['err']['HWY'] = round(100 * (hot_cycle(s, 'hwfet')['FE'] / rec['epa']['HWY'] - 1), 2)
    except Exception as ex:
        o['error'] = str(ex)[:200]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-band', type=int, default=150)
    ap.add_argument('--workers', type=int, default=14)
    _opts.init()
    if _opts.get() is None:
        _opts.set('roadsim-physics-v2')
    a = ap.parse_args()
    recs = records(a.per_band)
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = pool.map(one, [(r, _opts.get()) for r in recs], chunksize=2)
    df = pd.DataFrame([dict(band=r['band'], map=r['map'], FTP=r['err'].get('FTP'), HWY=r['err'].get('HWY')) for r in rows if r['err']])
    summ = {}
    for (b, m), g in df.groupby(['band', 'map']):
        summ[f'{b}|{m}'] = dict(n=len(g), FTP=round(float(g.FTP.median()), 2), HWY=round(float(g.HWY.median()), 2))
    for b, g in df.groupby('band'):
        summ[f'{b}|all'] = dict(n=len(g), FTP=round(float(g.FTP.median()), 2), HWY=round(float(g.HWY.median()), 2))
    for k in sorted(summ):
        print(f'{k:40s} n{summ[k]["n"]:4d}  FTP {summ[k]["FTP"]:+6.1f}  HWY {summ[k]["HWY"]:+6.1f}')
    json.dump(dict(about=__doc__.strip(), options=_opts.get(), summary=summ, rows=rows, wall_s=round(time.time() - t0)),
              open(os.path.join(HERE, 'era_dev_conv.json'), 'w', encoding='utf-8', newline='\n'), indent=1)


if __name__ == '__main__':
    main()
