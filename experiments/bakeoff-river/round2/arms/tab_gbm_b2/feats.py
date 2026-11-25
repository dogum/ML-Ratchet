"""Daily lag / antecedent features for the table learners (GBM, GP) of arms_tab. numpy only.

Built ONLY from the batch inputs (forcing, day of year, static vector), identically for sim and real windows.
For target row t (absolute row n_warm + j of the batch forcing):
  precipitation lags   P(t+1), P(t), P(t-1), P(t-2), sum P(t-6..t-3)                  / p_mean
  liquid input lags    W(t), W(t-1), W(t-2), melt(t)                                  / p_mean
                       W = rain + melt of a fixed one-band degree-day snow proxy (TT 0 C, interval 2 C,
                       melt factor 3 mm/C/day); SWE proxy (mm) and log1p(SWE)
  antecedent EMAs      exponential moving averages of W with time constants 3, 10, 30, 90, 365 days / p_mean
                       (initialised at the mean of W over the warm-up rows)
  two buckets          S += W; spill above Smax; ET = PET x S/Smax; drain S/30 per day; Smax 50 and 300 mm:
                       fill fraction, spill EMA(3 d) / p_mean, drain / p_mean (initialised half full)
  slow store           linear reservoir (k = 1/60 per day) fed by the 300 mm bucket's spill + drain: outflow / p_mean
  temperature / PET    T(t), EMA10 T, PET(t), EMA30 PET
  season               sin, cos of 2 pi doy / 365.25
  static               FlowCast's 18-value static vector as given
The target used in training is qo / p_mean (p_mean = exp(static[11]), the forcing's own mean precipitation).
"""
import numpy as np

DYN_NAMES = ['P_tp1', 'P_t', 'P_tm1', 'P_tm2', 'P_tm3_6', 'W_t', 'W_tm1', 'W_tm2', 'melt_t',
             'ema3', 'ema10', 'ema30', 'ema90', 'ema365',
             'b50_frac', 'b50_spill', 'b50_drain', 'b300_frac', 'b300_spill', 'b300_drain', 'slow_out',
             'T_t', 'T_ema10', 'PET_t', 'PET_ema30', 'swe', 'log_swe', 'doy_sin', 'doy_cos']
STATIC_NAMES = ['log_area', 'elev_mean_km', 'elev_range_km', 'log_slope_mkm', 'forest_frac', 'soil_porosity',
                'log_ksat_cmh', 'carbonate_frac', 'reservoir_frac', 'irrigation_frac', 'urban_frac',
                'log_p_mean', 'pet_mean', 'log_aridity', 'frac_snow', 'hp_freq', 'lp_freq', 't_mean_c']
NAMES = DYN_NAMES + ['s_' + s for s in STATIC_NAMES]
# monotone constraints (+1: more water in -> more flow; physics fixes the sign); 0 elsewhere
MONO_POS = {'P_tp1', 'P_t', 'P_tm1', 'P_tm2', 'P_tm3_6', 'W_t', 'W_tm1', 'W_tm2', 'melt_t', 'ema3', 'ema10',
            'ema30', 'ema90', 'ema365', 'b50_frac', 'b50_spill', 'b50_drain', 'b300_frac', 'b300_spill',
            'b300_drain', 'slow_out'}
MONOTONE = [1 if n in MONO_POS else 0 for n in NAMES]
TAUS = (3.0, 10.0, 30.0, 90.0, 365.0)


def features(F, doy, static, n_warm, n_target):
    """F (N, T, 8) forcing (C.DYN order), doy (N, T), static (N, 18). Returns X (N, n_target, D) float32 and
    p_mean (N,). Rows n_warm .. n_warm + n_target - 1 of each window; row t + 1 must exist (T >= n_warm+n_target+1)."""
    F = np.asarray(F, np.float64)
    N, T, _ = F.shape
    static = np.asarray(static, np.float64).reshape(N, -1)
    pm = np.exp(static[:, 11])
    P = np.maximum(F[:, :, 0], 0.0)
    E = np.maximum(F[:, :, 1], 0.0)
    Tm = F[:, :, 2]
    # --- snow proxy (one band, degree-day) and buckets: one loop over days, vectorised over windows
    fs = np.clip((1.0 - Tm) / 2.0, 0.0, 1.0)
    snow = P * fs
    rain = P - snow
    potmelt = 3.0 * np.maximum(Tm, 0.0)
    swe = np.zeros(N)
    SWE = np.empty((N, T)); MELT = np.empty((N, T))
    for t in range(T):
        swe = swe + snow[:, t]
        m = np.minimum(swe, potmelt[:, t])
        swe = swe - m
        SWE[:, t] = swe; MELT[:, t] = m
    W = rain + MELT
    # EMAs of W, temperature and PET
    taus = np.array(TAUS)
    a = 1.0 / taus
    e = np.repeat(W[:, :n_warm].mean(1)[:, None], len(taus), 1)
    eT = Tm[:, :n_warm].mean(1); eE = E[:, :n_warm].mean(1)
    EMA = np.empty((N, T, len(taus))); ET10 = np.empty((N, T)); EE30 = np.empty((N, T))
    smax = np.array([50.0, 300.0])
    S = np.repeat((0.5 * smax)[None], N, 0)
    sp_ema = np.zeros((N, 2))
    G = np.zeros(N)
    BF = np.empty((N, T, 2)); BS = np.empty((N, T, 2)); BD = np.empty((N, T, 2)); GO = np.empty((N, T))
    for t in range(T):
        w = W[:, t]
        e = e + (w[:, None] - e) * a[None]
        EMA[:, t] = e
        eT = eT + (Tm[:, t] - eT) / 10.0
        eE = eE + (E[:, t] - eE) / 30.0
        ET10[:, t] = eT; EE30[:, t] = eE
        S = S + w[:, None]
        spill = np.maximum(S - smax[None], 0.0)
        S = S - spill
        et = np.minimum(E[:, t][:, None] * S / smax[None], S)
        S = S - et
        dr = S / 30.0
        S = S - dr
        sp_ema = sp_ema + (spill - sp_ema) / 3.0
        G = G + spill[:, 1] + dr[:, 1]
        go = G / 60.0
        G = G - go
        BF[:, t] = S / smax[None]; BS[:, t] = sp_ema; BD[:, t] = dr; GO[:, t] = go
    r = np.arange(n_warm, n_warm + n_target)
    ipm = (1.0 / pm)[:, None]
    P36 = np.zeros((N, T))
    c = np.concatenate([np.zeros((N, 1)), np.cumsum(P, 1)], 1)       # c[:, k] = sum P[:, :k]
    lo = np.clip(r - 6, 0, None); hi = np.clip(r - 2, 0, None)          # rows t-6 .. t-3
    P36 = c[:, hi] - c[:, lo]
    w_ = 2 * np.pi * np.asarray(doy, np.float64)[:, r] / 365.25
    cols = [P[:, r + 1] * ipm, P[:, r] * ipm, P[:, r - 1] * ipm, P[:, r - 2] * ipm, P36 * ipm,
            W[:, r] * ipm, W[:, r - 1] * ipm, W[:, r - 2] * ipm, MELT[:, r] * ipm]
    cols += [EMA[:, r, k] * ipm for k in range(len(taus))]
    cols += [BF[:, r, 0], BS[:, r, 0] * ipm, BD[:, r, 0] * ipm, BF[:, r, 1], BS[:, r, 1] * ipm, BD[:, r, 1] * ipm,
             GO[:, r] * ipm]
    cols += [Tm[:, r], ET10[:, r], E[:, r], EE30[:, r], SWE[:, r], np.log1p(SWE[:, r]), np.sin(w_), np.cos(w_)]
    X = np.stack(cols, 2)
    X = np.concatenate([X, np.repeat(static[:, None, :], n_target, 1)], 2)
    return X.astype(np.float32), pm


def batch_features(b):
    """Features of one harness batch -> (n_target, D), p_mean."""
    X, pm = features(b['forcing'][None], np.asarray(b['doy'])[None], np.asarray(b['static'])[None],
                     int(b['n_warm']), int(b['n_target']))
    return X[0], float(pm[0])
