"""Widen-only error-bar factors for the release, fitted on the 16 dev cars only (the same arithmetic as exam.py), so the
weights can be exported and committed before the exam is run. usage: python sigk.py <ckpt>  -> prints the JSON"""
from common import *
import evaluate as E, realbank
from exam import sig_factor
import realdata as rd


def fit(ckpt):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    vt = rd.ved_vehicles()
    _, Rd = E.real_scores(net, 'dev', rb=realbank.load('dev'))
    out = {}
    for mode, P, vids in [('car', Rd['Pc'], Rd['cars']), ('trip', Rd['Pt'], Rd['trip_vid'])]:
        cb = np.array([vt.loc[c, 'gen_weight_lb'] * LB for c in vids]); ok = np.isfinite(cb)
        zm = (P['mu'][ok, 0] - np.log((cb[ok] + CURB_OFFSET) / 1800)) / np.sqrt(P['sig'][ok, 0] ** 2 + (113 / 1700) ** 2 / 3)
        dd = np.array([vt.loc[c, 'displacement_L'] for c in vids]); okd = np.isfinite(dd)
        zd = (P['mu'][okd, 1] - np.log(dd[okd] / 2.5)) / P['sig'][okd, 1]
        out[f'mass_{mode}'] = sig_factor(zm); out[f'disp_{mode}'] = sig_factor(zd)
    return out


if __name__ == '__main__':
    print(json.dumps(fit(sys.argv[1])))
