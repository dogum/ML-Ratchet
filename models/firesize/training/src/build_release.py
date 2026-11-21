"""Build the release folder (release/firesize-v1.0/) and the page data (web/page_data.json) AFTER the single exam run.
Chooses nothing about the model: the demo examples and the map points are fixed seeded draws.
usage: build_release.py"""
import sys, json, os, shutil, glob, base64
from common import *
import net, supp, train
import features as F, physfeat as PF
from sim import TG

REL = f'{ROOT}/release/firesize-v1.0'
WEB = f'{ROOT}/web'
FCLS_NAME = dict(GR='grass', GS='grass-shrub', SH='shrub', TU='timber understory', TL='timber litter', SB='slash/blowdown',
                 NBU='urban (non-burnable)', NBA='agriculture', NBO='other non-burnable')
REGION_NAMES = dict(CA='California', NW='Northwest', NR='Northern Rockies', GB='Great Basin', SW='Southwest',
                    RM='Rocky Mountain', SP='Southern Plains', SE='Southeast', NE_MW='Northeast / Midwest')
CLASS_SHORT = ['<.1', '.1', '.25', '1', '3', '10', '30', '100', '300', '1k', '5k', '20k+']


def r5(a, s=5):
    a = np.asarray(a, float)
    return [float(f'{v:.{s}g}') for v in a.ravel()]


def main():
    W = dict(np.load(f'{ROOT}/models/final.npz'))
    feats = [str(f) for f in W['feats']]
    ex = json.load(open(f'{ROOT}/results/exam.json'))
    post = json.load(open(f'{ROOT}/results/post_exam.json'))
    preds = pd.read_parquet(f'{ROOT}/results/exam_preds.parquet', columns=['FOD_ID', 'p_escape', 'med_ha'])
    d = fod_records('exam')
    d = d.merge(preds, on='FOD_ID')
    rs = np.random.default_rng(SEED + 7)
    # ---- 8 demo examples: 2 per size group, seeded draw
    groups = [(0, 1), (1, 100), (100, 1000), (1000, 1e9)]
    pick = []
    for lo, hi in groups:
        idx = np.nonzero(((d.FIRE_SIZE >= lo) & (d.FIRE_SIZE < hi) & d.DISCOVERY_TIME.notna()).values)[0]
        pick += list(rs.choice(idx, 2, replace=False))
    E = d.iloc[pick].reset_index(drop=True)
    E = F.fod_gridmet(E)
    X = F.fod_features(E)
    X = pd.concat([X, PF.phys_features(E.x.values, E.y.values, E.tmmx.values, E.rmin.values, E.vs.values,
                                       E.tmmx_m1.values, E.rmin_m1.values, E.vs_m1.values)], axis=1)
    Xv = X[feats].values.astype(float)
    P = net.forward_np(W, Xv)
    assert np.allclose(P[:, supp.ESC_BIN:].sum(1), E.p_escape.values, atol=1e-4), 'demo != exam predictions'
    # fuel codes around each example (for in-browser ROS recomputation) + LUT rows
    lut = PF.build_lut()
    a = np.load(f'{DATA}/nat/fuel.npz')['a']
    r = np.floor((F.Y1 - E.y.values) / F.R).astype(int); c = np.floor((E.x.values - F.X0) / F.R).astype(int)
    d13 = np.arange(-6, 7)
    codes = a[np.clip(r[:, None, None] + d13[None, :, None], 0, a.shape[0] - 1),
              np.clip(c[:, None, None] + d13[None, None, :], 0, a.shape[1] - 1)].reshape(len(E), -1).astype(int)
    del a
    lutrows = {int(k): r5(lut[k], 4) for k in np.unique(codes)}
    examples = []
    for i, e in E.iterrows():
        cls = [k for k in F.FCLS if X.loc[i, f'pt_{k}'] == 1][0]
        b = int(np.searchsorted(supp.EDGES, np.log10(e.size_ha), side='right'))
        nm = str(e.FIRE_NAME).title() if isinstance(e.FIRE_NAME, str) and e.FIRE_NAME.strip() else 'Unnamed'
        examples.append(dict(label=f'{nm[:14]} ({int(e.FIRE_YEAR)})', name=nm, date=str(e.DISCOVERY_DATE)[:10],
                             state=str(e.STATE), cause=str(e.NWCG_GENERAL_CAUSE), fuel=FCLS_NAME[cls], acres=float(e.FIRE_SIZE),
                             cls=str(e.FIRE_SIZE_CLASS), obs_bin=b, x=r5(Xv[i], 6), codes=codes[i].tolist(), fod_id=int(e.FOD_ID)))
    # ---- release folder
    if os.path.exists(REL):
        shutil.rmtree(REL)
    os.makedirs(f'{REL}/training', exist_ok=True)
    np.savez(f'{REL}/firesize_weights.npz', **W)
    shutil.copy(f'{ROOT}/release/firesize.py', REL)
    shutil.copy(f'{ROOT}/release/demo.py', REL)
    shutil.copy(f'{ROOT}/release/MODEL_CARD.md', REL)
    shutil.copy(f'{DATA}/ros_lut.npz', REL)
    dem = X[feats].copy()
    dem.insert(0, 'FOD_ID', E.FOD_ID.values); dem.insert(1, 'fire_name', [x['name'] for x in examples])
    dem.insert(2, 'year', E.FIRE_YEAR.values); dem.insert(3, 'state', E.STATE.values)
    dem.insert(4, 'real_acres', E.FIRE_SIZE.values); dem.insert(5, 'exam_p_escape', E.p_escape.values)
    dem.to_csv(f'{REL}/demo_examples.csv', index=False)
    for f in glob.glob(f'{ROOT}/src/*.py') + glob.glob(f'{ROOT}/src/*.sh'):
        shutil.copy(f, f'{REL}/training/')
    os.makedirs(f'{REL}/results', exist_ok=True)
    for f in ['exam.json', 'post_exam.json', 'widen.json', 'dev_gate_baselines.json']:
        shutil.copy(f'{ROOT}/results/{f}', f'{REL}/results/')
    for f in ['ratchet.json', 'rungs.json']:
        shutil.copy(f'{ROOT}/{f}', f'{REL}/results/')
    os.makedirs(f'{REL}/results/calibration', exist_ok=True)
    for f in glob.glob(f'{DATA}/calib_*.json'):
        shutil.copy(f, f'{REL}/results/calibration/')
    # ---- page data
    ne = int(W['n_ens'])
    members = [[dict(w=r5(W[f'e{m}_L{k}_w']), b=r5(W[f'e{m}_L{k}_b'])) for k in range(3)] for m in range(ne)]
    netj = dict(feats=feats, fill=r5(W['fill'], 7), mu=r5(W['mu'], 7), sd=r5(W['sd'], 7), temp=float(W['temp']), members=members)
    # ERC coupling (dev records, least squares slopes)
    dv = F.fod_gridmet(fod_records('dev'))
    ok = dv[['erc', 'bi', 'fm100', 'fm1000']].notna().all(1)
    coup = {k: float(np.polyfit(dv.erc[ok], dv[k][ok], 1)[0]) for k in ['bi', 'fm100', 'fm1000']}
    # containment race examples: dev twins (calib half, simulated in their real windows), one per fuel group
    th = train.theta_of('v1rep2')
    import calib
    B = calib.load_bank('v1', 'twin')
    I = pd.read_parquet(f'{DATA}/ignitions.parquet').set_index('ign')
    race = []
    for g, lab in [('GR', 'Grass'), ('SH', 'Shrub'), ('TU', 'Timber understory'), ('TL', 'Timber litter')]:
        idx = [i for i in range(len(B['ign'])) if supp.fgroup(B['fuel0'][i]) == g and B['A'][i, -1] > 50]
        i = int(rs.choice(idx))
        ig = I.loc[int(B['ign'][i])]
        rr = th['r0'][g] * th['r0_mult'] * np.exp(th['r0_fed'] * ig.fed)
        race.append(dict(label=f'{lab}, {ig.region} {int(ig.year)}', A=r5(B['A'][i], 4), lw=r5(B['lw'][i], 3),
                         tau0=float(B['tau0'][i]), lr0=float(np.log10(rr)), real_acres=float(ig.size_ha / ACRE_HA)))
    # map points: seeded draw, escapes over-sampled x10
    u = rs.uniform(size=len(d))
    m = np.where(d.escape.values == 1, u < 0.02, u < 0.002)
    S = d[m]
    px = (S.x.values - F.X0) / 4000.0; py = (F.Y1 - S.y.values) / 4000.0
    mappts = [[round(float(a_), 1), round(float(b_), 1), round(float(p), 4), int(e), float(ac), int(y),
               (str(n).title() if isinstance(n, str) else 'Unnamed')[:22], f'{st} · {cs}', round(float(mh / ACRE_HA), 2)]
              for a_, b_, p, e, ac, y, n, st, cs, mh in zip(px, py, S.p_escape, S.escape, S.FIRE_SIZE, S.FIRE_YEAR,
                                                            S.FIRE_NAME, S.STATE, S.NWCG_GENERAL_CAUSE, S.med_ha)]
    png = base64.b64encode(open(f'{WEB}/conus_fuel_q.png', 'rb').read()).decode()
    R = json.load(open(f'{ROOT}/ratchet.json'))
    ratchet = [dict(rung=x['rung'], change=x['change'], rpss=x['sim_latest']['rpss'], auc=x['dev_gate']['esc_auc'],
                    ap=x['dev_gate']['esc_ap'], mae=x['dev_gate']['mae'], rho=x['dev_gate']['rho'], kept=x['kept'],
                    reason=x['reason']) for x in R['rungs'] if 'sim_latest' in x]
    D = dict(net=netj, edges_log10_ac=r5(np.log10(supp.EDGES_AC), 6), class_short=CLASS_SHORT,
             exam=dict(n=ex['n'], periods=ex['periods'], baselines=ex['baselines'], reliability=ex['reliability']),
             exam_region_base={k: dict(clim=v['clim']['auc'], gbm=v['gbm']['auc']) for k, v in post['region'].items()},
             post=post, examples=examples, lut=lutrows, lut_m1=PF.M1.tolist(), lut_u10=PF.U10.tolist(),
             erc_coupling=coup, race=race, tg=r5(TG, 5), f_perim=float(th['f_perim']), tau_e=float(th['tau_e']),
             map_png='data:image/png;base64,' + png, mappts=mappts, regions=REGIONS, region_names=REGION_NAMES,
             ratchet=ratchet, widen=json.load(open(f'{ROOT}/results/widen.json')),
             devbase=json.load(open(f'{ROOT}/results/dev_gate_baselines.json')))
    json.dump(D, open(f'{WEB}/page_data.json', 'w'), separators=(',', ':'), default=float)
    print('examples', [(x['label'], x['acres'], round(float(p), 3)) for x, p in zip(examples, E.p_escape)])
    print('race', [(x['label'], x['real_acres'], x['tau0'], x['lr0']) for x in race])
    print('mappts', len(mappts), 'coupling', coup, 'page_data KB', os.path.getsize(f'{WEB}/page_data.json') // 1024)


if __name__ == '__main__':
    main()
