"""Real day-ahead temperature forecast-error library from the training-input year 2022 (Open-Meteo previous runs,
population-weighted): per BA, one 38-h error series (issue 10:00 LST D-1 -> end of D) per day. -> data/noise/tfc_2022.npz"""
import numpy as np
from common import *
import real

out = {}
for ba in BAS:
    F = real.Forecasts(ba)
    rows = []
    for d in days_of_year(ba, 2022):
        i0 = day_start_index(ba, d)
        e = F.T[i0 - 14:i0 + 24]
        if np.isfinite(e).all(): rows.append(e)
    if rows:
        out[ba] = np.array(rows, np.float32)
        a = out[ba]
        print(ba, len(rows), 'rmse D (K)', np.sqrt(np.mean(a[:, 14:] ** 2)).round(2), 'mean bias', a.mean().round(2))
np.savez_compressed(f'{DATA}/noise/tfc_2022.npz', **out)
