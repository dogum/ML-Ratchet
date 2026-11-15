"""Command line: python pipeline.py output.csv [--price price.csv] [--context context.json] [--net]

output.csv : columns time_utc (ISO, hour beginning) and mw (hourly average output; blank = missing)
price.csv  : columns time_utc, price ($/MWh), gas ($/MMBtu)     (optional)
context.json: EIA-860/923 facts, see plantdetective.CONTEXT_HELP (optional)
Prints the Detective's report as JSON.
"""
import sys, json, argparse
import numpy as np
from plantdetective import Detective


def read_csv(path):
    import csv
    rows = list(csv.DictReader(open(path)))
    t = np.array([np.datetime64(r['time_utc'].replace('Z', '').split('+')[0], 's').astype('int64') for r in rows])
    return t, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output'); ap.add_argument('--price'); ap.add_argument('--context'); ap.add_argument('--net', action='store_true')
    ap.add_argument('--utc-offset', type=float, default=-6)
    a = ap.parse_args()
    t, rows = read_csv(a.output)
    x = np.array([float(r['mw']) if r['mw'] not in ('', None) else np.nan for r in rows])
    price = gas = None
    if a.price:
        tp, pr = read_csv(a.price)
        d = {int(k): (float(r['price']), float(r['gas'])) for k, r in zip(tp, pr)}
        price = np.array([d.get(int(k), (np.nan, np.nan))[0] for k in t]); gas = np.array([d.get(int(k), (np.nan, np.nan))[1] for k in t])
    ctx = json.load(open(a.context)) if a.context else None
    r = Detective().infer(x, t, price, gas, ctx, utc_offset=a.utc_offset, net=a.net)
    r.pop('windows')
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
