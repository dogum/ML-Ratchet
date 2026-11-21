"""FireDetective pipeline: raw fire observations -> model features -> predictions (numpy/scipy only).

Inputs for one fire (all on one 180 m EPSG:5070 grid, row 0 = north):
  masks   [n, ny, nx] bool   cumulative burned area at each FEDS half-day step (S-NPP night ~01:30 and day ~13:30
                             overpasses), starting at the first step with burned area; up to 12 steps are used
  dets    list of n dicts    VIIRS 375 m detections of that step: 'row', 'col' (grid pixel coordinates, float),
                             'frp' (MW)
  day     [n] bool           True for the day (13:30) overpass step
  wx      list of n dicts    gridMET daily values for the local day driving the step's growth (PM step: same day;
                             AM step: the previous day): tmmx (C), rmin, rmax (%), vs (m/s at 10 m), th (deg from,
                             true north), fm100, fm1000 (%), erc, bi, pr (mm), vpd (kPa)
  layers  dict of [ny, nx]   LANDFIRE at 180 m: fuel (FBFM40), cc (%), ch (m), cbh (m), cbd (kg/m3), slope (deg),
                             elev (m)
  doy     day of year of the first step; gamma: grid convergence (deg) at the fire (EPSG:5070 grid north vs true north)
"""
import numpy as np
import feats
import firedetective as fdet


def features(masks, dets, day, wx, layers, doy, gamma=0.0):
    masks = np.asarray(masks, bool)[:feats.T_STEPS]
    seq = feats.seq_features(masks, dets, np.asarray(day), wx, gamma_deg=gamma)
    seq_h = feats.seq_features(masks, dets, np.asarray(day), wx, wind_hidden=True, gamma_deg=gamma)
    st = feats.static_features(layers, feats.area_mask_from(masks), doy, len(masks))
    return seq, seq_h, st


def run(fires, model=None):
    """fires: list of dicts with the keys above. Returns the prediction dict (arrays over fires)."""
    model = model or fdet.load()
    F = [features(f['masks'], f['dets'], f['day'], f['wx'], f['layers'], f['doy'], f.get('gamma', 0.0)) for f in fires]
    S, Sh, St = (np.stack(x) for x in zip(*F))
    return model.predict(S, Sh, St)
