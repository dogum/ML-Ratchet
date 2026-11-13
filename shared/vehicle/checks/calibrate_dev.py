"""Fit the few global calibration constants on the EPA *dev* vehicles only (fixed-seed third of each architecture group).

1. cold_fric_gasoline: extra friction torque (Nm per litre) at 20 C coolant, fading to zero when warm, matched to the
   median FTP bag1/bag3 fuel-economy ratio of the dev conventional vehicles.
2. bev_motor_loss_scale: one factor on the ALPHA e-machine loss maps for BEV drive units, matched to the median
   UDDS+HWFET error of the dev BEVs
   (with an assumed 0.88 wall-to-battery charging efficiency).
3. cold_fric_hybrid_factor: multiplier on that cold-start penalty for hybrid engines (which warm faster at high load,
   with exhaust-heat recovery), matched to the dev HEVs' median bag1/bag3 ratio (only dev HEVs with bag1 < bag3 count).

Writes roadsim/roadsim/data/calibration.json and road/checks/calibration.json (with the objective values).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
import roadsim.sim as rsim
from roadsim import calib
from epa_procedures import ftp, hot_cycle, test_spec

CAL = os.path.join(HERE, '..', 'roadsim', 'roadsim', 'data', 'calibration.json')
V = json.load(open(os.path.join(HERE, 'epa_vehicles.json')))
dev = [v for v in V if v['split'] == 'dev']


def set_cal(c):
    json.dump(c, open(CAL, 'w'), indent=1)
    calib.reload(); rsim._VEH_CACHE.clear()


def bag_ratio_err(vs):
    errs = []
    for v in vs:
        e = v['epa']
        if not (e.get('FE_Bag_1', 0) > 5 and e.get('FE_Bag_3', 0) > 5 and e['FE_Bag_1'] / e['FE_Bag_3'] < 1.0):
            continue
        r = ftp(test_spec(v['spec']))
        errs.append(np.log((r['bags'][0] / r['bags'][2]) / (e['FE_Bag_1'] / e['FE_Bag_3'])))
    return float(np.median(errs)), len(errs)


def bev_err(vs):
    errs = []
    for v in vs:
        e = v['epa']
        for cyc, key in (('udds', 'CD_UDDS'), ('hwfet', 'CD_HWY')):
            r = hot_cycle(test_spec(v['spec']), cyc)
            if v['bev_unit'] == 'MPGe':
                errs.append(np.log(r['MPGe'] / e[key]))
            else:
                errs.append(np.log(e[key] / r['kWh_100mi_ac']))
    return float(np.median(errs)), len(errs)


def bisect(f, lo, hi, n=7):
    flo = f(lo)[0]
    for _ in range(n):
        mid = 0.5 * (lo + hi); fm = f(mid)[0]
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == '__main__':
    c = dict(calib.DEFAULTS)
    set_cal(c)
    log = {}
    conv = [v for v in dev if v['spec']['architecture'] == 'conv' and v['spec']['engine']['fuel'] == 'gasoline']
    hev = [v for v in dev if v['spec']['architecture'].startswith('hev')]
    bev = [v for v in dev if v['spec']['architecture'] == 'bev']
    log['before'] = dict(conv_bag_ratio_logerr=bag_ratio_err(conv), hev_bag_ratio_logerr=bag_ratio_err(hev), bev_logerr=bev_err(bev))

    def f_cold(x):
        c['cold_fric_gasoline'] = x; set_cal(c); return bag_ratio_err(conv)
    c['cold_fric_gasoline'] = round(bisect(lambda x: tuple(-np.array(f_cold(x))), 2.0, 60.0), 2)
    set_cal(c)

    def f_mot(x):
        c['bev_motor_loss_scale'] = x; set_cal(c); return bev_err(bev)
    c['bev_motor_loss_scale'] = round(bisect(f_mot, 0.3, 1.2), 3)
    set_cal(c)

    c['cat_warmup_s'] = 10.0        # fixed (physics judgement: catalyst light-off ~10 s of engine running)

    def f_hev(x):
        c['cold_fric_hybrid_factor'] = x; set_cal(c); return bag_ratio_err(hev)
    c['cold_fric_hybrid_factor'] = round(bisect(lambda x: tuple(-np.array(f_hev(x))), 0.0, 1.5), 3)
    set_cal(c)
    c['cold_fric_diesel'] = c['cold_fric_gasoline']   # no diesel dev car: same friction penalty per litre
    set_cal(c)
    log['after'] = dict(conv_bag_ratio_logerr=bag_ratio_err(conv), hev_bag_ratio_logerr=bag_ratio_err(hev), bev_logerr=bev_err(bev))
    log['calibration'] = c
    log['dev_vehicles'] = [v['name'] for v in dev]
    json.dump(log, open(os.path.join(HERE, 'calibration.json'), 'w'), indent=1)
    print(json.dumps(log, indent=1))
