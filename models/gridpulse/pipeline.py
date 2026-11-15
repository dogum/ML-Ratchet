"""Command line: python pipeline.py trace.csv [GB|ERCO] [demand_MW] [N]
trace.csv: two columns t [s], f [Hz] at 1 s, covering >= 60 s before and 60 s after the trip."""
import sys, json, numpy as np
from gridpulse import GridPulse


def main(a):
    d = np.loadtxt(a[1], delimiter=',', skiprows=1)
    r = GridPulse().read(d[:, 0], d[:, 1], system=a[2] if len(a) > 2 else 'GB',
                         demand_MW=float(a[3]) if len(a) > 3 else np.nan, N=int(a[4]) if len(a) > 4 else 60)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main(sys.argv)
