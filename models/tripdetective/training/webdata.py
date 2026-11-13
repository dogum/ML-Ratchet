"""Page data: real exam cars for the explorer (fixed random draw within VED-type strata, seed 3; never cherry-picked),
their trips in compact integer form, Python predictions on exactly those decoded arrays (to check the JS port),
plus the exam, ratchet, pooling and twin-ledger summaries. usage: python3 webdata.py <ckpt>"""
from common import *
import realbank, realdata as rd, features as F
import evaluate as E
from model import from_reg

STRATA = {'ICE': 5, 'HEV': 4, 'PHEV': 3, 'EV': 1}
NTRIP = 8
Q = dict(v=100, grade=1000, hw=100, fuel=1000, rpm=1000, batt=10, soc=1000, aux=1000)


def compact(raw):
    """raw (len(RAW), n) -> dict of int lists (and the decoded raw used for predictions)"""
    r = raw.astype(np.float64).copy()
    T = float(np.round(np.mean(r[2]), 1)); rho = float(np.round(np.mean(r[4]), 3))
    d = dict(n=r.shape[1], T=T, rho=rho)
    for k, i in [('v', 0), ('grade', 1), ('hw', 3), ('fuel', 5), ('rpm', 6), ('batt', 7), ('soc', 8), ('aux', 9)]:
        q = np.round(r[i] * Q[k]).astype(int); d[k] = q.tolist(); r[i] = q / Q[k]
    r[2] = T; r[4] = rho
    d['valid'] = ''.join('1' if x > 0.5 else '0' for x in r[10])
    return d, r.astype(np.float32)


def build(ckpt):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    ex = realbank.load('exam'); vt = rd.ved_vehicles()
    exam_cars = list(dict.fromkeys(ex['vid'].tolist()))
    rng = np.random.default_rng(3); pick = []
    for t, k in STRATA.items():
        c = [v for v in exam_cars if vt.loc[v, 'ved_type'] == t and len(set(ex['trip'][ex['vid'] == v])) >= 4]
        pick += [c[i] for i in sorted(rng.choice(len(c), min(k, len(c)), replace=False))]
    o = ex['offs']; cars = []
    for v in pick:
        trips = list(dict.fromkeys(ex['trip'][ex['vid'] == v].tolist()))
        sel = [trips[i] for i in sorted(rng.choice(len(trips), min(NTRIP, len(trips)), replace=False))]
        tr, inp = [], []
        meta = {m['trip_id']: m for m in ex['meta']}
        for t in sel:
            js = np.where(ex['trip'] == t)[0]
            for j in js[:1]:            # first window of each trip (<= 1024 s)
                d, r = compact(ex['raw'][o[j]:o[j + 1]].T.astype(np.float32))
                d['flags'] = ex['flags'][j].astype(int).tolist(); d['id'] = t; d['km'] = round(meta[t]['km'], 2)
                d['date'] = ''
                tr.append(d); inp.append(F.net_input(r, ex['flags'][j]))
        P1 = E.predict(net, inp[:1], np.zeros(1, int)); Pk = E.predict(net, inp, np.zeros(len(inp), int))
        row = vt.loc[v]
        cars.append(dict(vid=v, ved_type=row.ved_type, arch_rd=row.architecture, arch_conf=row.arch_confidence, cls=row.cls,
                         curb_kg=None if not np.isfinite(row.gen_weight_lb) else round(row.gen_weight_lb * LB), weight_lb=None if not np.isfinite(row.gen_weight_lb) else int(row.gen_weight_lb),
                         disp=None if not np.isfinite(row.displacement_L) else row.displacement_L, engine=row.engine_str, trans=row.trans_str,
                         likely=row.likely_models if isinstance(row.likely_models, str) else '', trips=tr,
                         py1=dict(mass=float(from_reg(0, P1['mu'][0, 0])), arch=P1['arch'][0].round(4).tolist()),
                         pyk=dict(mass=float(from_reg(0, Pk['mu'][0, 0])), disp=float(from_reg(1, Pk['mu'][0, 1])), arch=Pk['arch'][0].round(4).tolist())))
        print(v, row.ved_type, len(tr), cars[-1]['pyk'], flush=True)
    data = dict(cars=cars, exam=jload(f'{ROOT}/exam.json'), ratchet=slim_ratchet(), epa_sim=jload(f'{ROOT}/epa_sim.json'),
                twin=jload(f'{ROOT}/twin_ledger_dev.json'), twin_exam=jload(f'{ROOT}/twin_ledger_exam.json'))
    json.dump(data, open(f'{ROOT}/web/data.json', 'w'), separators=(',', ':'), default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print('data.json', os.path.getsize(f'{ROOT}/web/data.json') // 1024, 'kB')


def slim_ratchet():
    log = jload(f'{ROOT}/ratchet.json', [])
    return [dict(ckpt=r['ckpt'], note=r['note'], objective=round(r['objective'], 4), accepted=r['accepted'], why=r['why'], val_tag=r['val_tag'],
                 single=r['val']['single']['skill'], pooled=r['val']['pooled']['skill'],
                 dev={k: round(float(r['dev'][k]), 3) for k in ['mass_mae_car', 'disp_mae_car', 'mass_mae_trip', 'disp_mae_trip', 'vedtype_acc_trip', 'vedtype_acc_car']})
            for r in log]


if __name__ == '__main__':
    build(sys.argv[1])
