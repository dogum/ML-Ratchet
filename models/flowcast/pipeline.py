"""FlowCast pipeline.

1) Any basin from CSV files:
   python3 pipeline.py --forcing my_forcing.csv --attrs my_attrs.json --start 1995-10-01 --end 1997-09-30 --out q.csv
   forcing CSV: a date column first, then P, PET, T_mean, T_min, T_max, SRAD (net shortwave W/m2), vp (Pa), wind (m/s);
   it must start 365 days before --start and run one day past --end. attrs JSON: see flowcast.py.
2) A CAMELS-US basin through the water platform's waterdata package (ML Ratchet: shared/water; the real forcing
   must have been rebuilt with `mlratchet data water`, and $ML_RATCHET_WATER moves the data root):
   python3 pipeline.py --camels camels_01013500 --start 1995-10-01 --end 1997-09-30 [--waterdata path/to/shared/water]
   Only this optional mode imports waterdata; flowcast.py itself needs numpy and pandas only.
"""
import argparse, json, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flowcast as fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forcing'); ap.add_argument('--attrs'); ap.add_argument('--camels'); ap.add_argument('--waterdata')
    ap.add_argument('--start', required=True); ap.add_argument('--end', required=True); ap.add_argument('--out')
    a = ap.parse_args()
    if a.camels:
        sys.path.insert(0, a.waterdata or os.path.join(HERE, '..', '..', 'shared', 'water'))
        import waterdata as wd
        s0 = (pd.Timestamp(a.start) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        e1 = (pd.Timestamp(a.end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        forcing, attrs = wd.forcing(a.camels, s0, e1), wd.spec(a.camels)
    else:
        forcing = pd.read_csv(a.forcing, index_col=0, parse_dates=True)
        with open(a.attrs, encoding="utf-8") as fh:
            attrs = json.load(fh)
    out = fc.FlowCast().predict(forcing, attrs, a.start, a.end)
    if a.out:
        out.to_csv(a.out); print('wrote', a.out)
    else:
        print(out.round(3).to_string(max_rows=20))


if __name__ == '__main__':
    main()
