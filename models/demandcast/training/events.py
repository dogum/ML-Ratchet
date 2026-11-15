"""Named extreme periods (reported, never used for choosing). Actual weather.
  Uri (Feb 2021, dev year), Winter Storm Elliott (Dec 2022, training-input year), heat waves 2023-2024 (exam).
python3 events.py <ckpt> -> data/exam/<ckpt>_events.json"""
import sys, json, numpy as np, pandas as pd
from common import *
import real, evaluate as E

EVENTS = [
    ('Winter Storm Uri', 'dev', '2021-02-10', '2021-02-20', ['ERCO', 'SWPP', 'MISO']),
    ('Winter Storm Elliott', 'training-input year', '2022-12-21', '2022-12-27', ['PJM', 'MISO', 'ERCO', 'SWPP', 'NYIS', 'ISNE']),
    ('Texas heat, Aug 2023', 'exam', '2023-08-01', '2023-08-31', ['ERCO']),
    ('Plains heat, Aug 2023', 'exam', '2023-08-18', '2023-08-26', ['SWPP', 'MISO']),
    ('California heat, Jul 2024', 'exam', '2024-07-02', '2024-07-12', ['CISO']),
    ('Northwest heat, Jul 2024', 'exam', '2024-07-05', '2024-07-12', ['BPAT']),
    ('Eastern heat dome, Jun 2024', 'exam', '2024-06-17', '2024-06-23', ['PJM', 'NYIS', 'ISNE', 'MISO']),
    ('Arctic outbreak, Jan 2024', 'exam', '2024-01-14', '2024-01-17', ['ERCO', 'SWPP', 'MISO', 'PJM', 'BPAT']),
    ('Arctic outbreak, Jan 2025', 'exam', '2025-01-19', '2025-01-23', ['ERCO', 'SWPP', 'MISO', 'PJM', 'NYIS', 'ISNE']),
]


def main(ck):
    nets = E.load(ck)
    sc = json.load(open(f'{DATA}/exam/{ck}.json'))['sigma_scale_dev']
    out = []
    for name, split, a, b, bas in EVENTS:
        for ba in bas:
            calib = 'dev2019' if split == 'dev' else 'dev'      # Uri (2021) uses the 2019 fit, like the dev bank
            real._CACHE.clear(); g = real.real_region(ba, calib)
            rows = [g.sample(day_start_index(ba, d)) for d in pd.date_range(a, b)]
            Bk = {k: np.stack([r[k] for r in rows]) for k in ('xd', 'yd', 'xw', 'yw', 'pw', 'xs', 'ys', 'ps', 'night') if all(k in r for r in rows)}
            Bk['night'] = Bk.get('night', np.zeros((len(rows), 24)))
            pred = E.predict({k: v for k, v in nets.items() if ('x' + k) in Bk}, Bk)
            y = np.stack([r['d_true'] for r in rows]); prior = np.stack([r['pd_total'] for r in rows])
            idx = [day_start_index(ba, d) for d in pd.date_range(a, b)]
            df = np.stack([g.df[i:i + 24] for i in idx])
            p = prior * np.exp(pred['d'][0]); m = np.isfinite(y) & np.isfinite(df) & (y > 0)
            z = np.abs(np.log(np.where(m, y, 1) / prior) - pred['d'][0]) / (pred['d'][1] * sc['d'])
            rec = dict(event=name, split=split, ba=ba, start=a, end=b, hours=int(m.sum()),
                       model=float(np.mean(np.abs(p - y)[m] / y[m]) * 100), ba_day_ahead=float(np.mean(np.abs(df - y)[m] / y[m]) * 100),
                       physics_prior=float(np.mean(np.abs(prior - y)[m] / y[m]) * 100), cov95=float(np.mean(z[m] <= 1.96)),
                       peak_actual=float(np.nanmax(y)), peak_model=float(np.nanmax(np.where(m, p, np.nan))))
            out.append(rec)
            print(f"{name:30s} {ba:5s} model {rec['model']:.2f}% BA DA {rec['ba_day_ahead']:.2f}% prior {rec['physics_prior']:.2f}% cov95 {rec['cov95']:.2f} peak {rec['peak_actual']:.0f} vs {rec['peak_model']:.0f}")
    jdump(out, f'{DATA}/exam/{ck}_events.json')


if __name__ == '__main__':
    main(sys.argv[1])
