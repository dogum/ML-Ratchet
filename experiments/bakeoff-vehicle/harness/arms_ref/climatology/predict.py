"""Climatology (the dumb baseline): trip energy = the simulated train bank's median energy per km for the vehicle's
(architecture, class) x the trip distance; band = that group's 5-95 % spread of log(energy / (rate x distance)) on
the train bank. Distance: table track 5000 * exp(X[:, 1]) (EnergyNet's route vector R[1] = log(distance / 5000 m));
trace track the sum of the 1 Hz speeds (the same number). Numpy only; fitted by work/fit.py on simulated data only."""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'clim.json')))


def predict(batch):
    if batch['track'] == 'table':
        dist = 5000.0 * np.exp(np.asarray(batch['X'], np.float64)[:, 1])
    else:
        dist = np.array([max(float(np.sum(t[:, 0])), 1.0) for t in batch['traces']])
    k = np.asarray(batch['arch']) * 10 + np.asarray(batch['cls'])
    rate = np.array([P.get(str(int(g)), P['all'])['rate_MJ_per_m'] for g in k])
    lq = np.array([P.get(str(int(g)), P['all'])['logq'] for g in k])
    base = np.maximum(rate * dist, 1e-6)
    Q = base[:, None] * np.exp(lq)
    out = {'pred': base * np.exp(lq[:, 2])}
    out.update({q: Q[:, j] for j, q in enumerate(['q05', 'q25', 'q50', 'q75', 'q95'])})
    return out
