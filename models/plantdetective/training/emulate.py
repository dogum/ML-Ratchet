"""Measurement and reporting emulators: turn a simulated unit schedule (block MW per hour) into
  (1) a CAMPD-like full-information record (op, P gross, HI, meas) with start-up / shut-down hours, warm-up ramps,
      load noise, ambient derate, CEMS noise, combined-cycle reporting basis and cogeneration steam fuel;
  (2) the EIA-860/923 context PlantDetective may see (prime mover, CHP flag, nameplates, EIA-860 minimum load,
      cold-start time, EIA-923 annual net heat rate of the linked generators with its known distortions).
Every random choice is drawn per unit from ranges in EMU (versioned with the bank; dev-calibrated values replace
them rung by rung, never exam data)."""
import numpy as np
from pdcore import runs, HOT_H, WARM_H

EMU = dict(
    load_sigma=(0.004, 0.03),          # AR(1) relative load noise while on (AGC, base-point following)
    load_rho=(0.5, 0.95),
    amb=dict(gas_ct=(0.0, 0.12), gas_cc=(0.0, 0.08), coal_steam=(0.0, 0.02), gas_steam=(0.0, 0.02)),
    # start-up and shut-down trajectories (v2: measured on the dev units' 2019/2021 starts, see out/dev_starts.json)
    # op1: operating fraction of the first hour; a1: first-hour load [fraction of pmax]; r_wu: warm-up ramp [pmax/h]
    start=dict(gas_ct=dict(op1=(0.15, 1.0), a1=(0.10, 0.40), r_wu=(0.5, 1.0)),
               gas_cc=dict(op1=(0.30, 1.0), a1=(0.08, 0.30), r_wu=(0.25, 0.6)),
               coal_steam=dict(op1=(0.50, 1.0), a1=(0.0, 0.05), r_wu=(0.02, 0.08)),
               gas_steam=dict(op1=(0.60, 1.0), a1=(0.0, 0.05), r_wu=(0.06, 0.2))),
    # p: probability the last on-hour is a partial shut-down hour; a_last: its load [pmax]; r_dn: ramp-down before it
    stop=dict(gas_ct=dict(p=(0.6, 0.95), op_last=(0.2, 1.0), a_last=(0.2, 0.6), r_dn=(0.5, 1.0)),
              gas_cc=dict(p=(0.6, 0.95), op_last=(0.3, 1.0), a_last=(0.15, 0.5), r_dn=(0.3, 0.6)),
              coal_steam=dict(p=(0.6, 0.95), op_last=(0.3, 1.0), a_last=(0.02, 0.15), r_dn=(0.1, 0.3)),
              gas_steam=dict(p=(0.6, 0.95), op_last=(0.3, 1.0), a_last=(0.02, 0.15), r_dn=(0.2, 0.5))),
    # short spells real units show: trips inside a run, aborted starts (probability per run)
    q_trip=dict(gas_ct=0.02, gas_cc=0.05, coal_steam=0.08, gas_steam=0.05),
    q_abort=dict(gas_ct=0.02, gas_cc=0.05, coal_steam=0.08, gas_steam=0.05),
    hi_sigma=(0.005, 0.025),
    p_ct_only=0.45, s0=(0.58, 0.72), s1=(0.0, 0.15), n_share=(1, 2, 3),
    chp_p=dict(gas_cc=0.5, gas_ct=0.3, gas_steam=0.1, coal_steam=0.03),
    phi1=(0.0, 0.5), phi2=(0.0, 0.4),  # cogeneration: extra fuel for process steam
    aux=dict(coal_steam=(0.05, 0.10), gas_steam=(0.03, 0.07), gas_cc=(0.02, 0.04), gas_ct=(0.005, 0.02)),
    hr923_sigma=0.04, hr923_gross_err=0.05, hr923_missing=0.15, chp_alloc=(0.45, 0.95),
    minload_sigma=0.3, minload_missing=0.2,
    cold_h=dict(gas_ct=(0.17, 1.0), gas_cc=(1.0, 12.0), coal_steam=(12.0, 24.0), gas_steam=(12.0, 24.0)),
)


def draw_reporting(tech, rng, emu=EMU):
    """Per-unit reporting / plant-type draws."""
    rep = dict(chp=bool(rng.random() < emu['chp_p'][tech]), ct_only=False, n_share=1, s0=1.0, s1=0.0)
    if tech == 'gas_cc':
        rep['ct_only'] = bool(rng.random() < emu['p_ct_only'])
        rep['n_share'] = int(rng.choice(emu['n_share']))
        rep['s0'] = float(rng.uniform(*emu['s0'])); rep['s1'] = float(rng.uniform(*emu['s1']))
    rep['phi1'] = float(rng.uniform(*emu['phi1'])) if rep['chp'] else 0.0
    rep['phi2'] = float(rng.uniform(*emu['phi2'])) if rep['chp'] else 0.0
    rep['aux'] = float(rng.uniform(*emu['aux'][tech]))
    return rep


def fuel_fn(u):
    """Block fuel input F(P) [MMBtu/h] from hr_full, beta (no-load share) and curvature."""
    pmax, hr, b, c = u['pmax'], u['hr_full'], u['beta'], u.get('curv', 0.0)
    def F(P):
        f = np.maximum(np.asarray(P, float) / pmax, 1e-3)
        return P * hr * (1 + b * (1 / f - 1) + c * (f - 1) ** 2)
    return F


def short_spells(on, tech, rng, emu=EMU):
    """Insert trips (1-3 h off inside a run) and aborted starts (1-2 h on, then 1-3 h off, before a start)."""
    on = np.asarray(on, bool).copy()
    s, L, v = runs(on)
    for i in np.flatnonzero(v):
        t0, n = s[i], L[i]
        if n > 6 and rng.random() < emu['q_trip'][tech]:
            k = t0 + int(rng.integers(2, n - 3)); on[k:k + int(rng.integers(1, 4))] = False
        if i > 0 and L[i - 1] > 5 and rng.random() < emu['q_abort'][tech]:
            a = int(rng.integers(1, 3)); g = int(rng.integers(1, 4))
            lo = t0 - a - g
            if lo > s[i - 1]:
                on[lo:lo + a] = True
    return on


def record(u, on, Pd, doy, rng, rep, emu=EMU):
    """CAMPD-like record of one unit.  u: unit dict (tech, pmax, pmin, hr_full, beta, curv, start_heat);
    on/Pd: hourly schedule (block MW); doy: day of year per hour.  Returns dict op, P, HI, meas (reported basis)."""
    tech = u['tech']; pmax = u['pmax']; pmin = u['pmin']
    T = len(on)
    on0 = np.asarray(on, bool)
    on = short_spells(on0, tech, rng, emu)
    Pd = np.where(on, np.where(on0, np.asarray(Pd, float), pmin), 0.0)
    # load noise (AR1) and ambient derate of gas turbines (summer)
    sig = rng.uniform(*emu['load_sigma']); rho = rng.uniform(*emu['load_rho'])
    e = rng.normal(0, 1, T) * sig * np.sqrt(1 - rho ** 2)
    ar = np.empty(T); a = 0.0
    for t in range(T):
        a = rho * a + e[t]; ar[t] = a
    amb = rng.uniform(*emu['amb'][tech])
    pcap = pmax * (1 - amb * 0.5 * (1 + np.cos(2 * np.pi * (np.asarray(doy) - 196) / 365.25)))
    P = np.clip(Pd * (1 + ar), 0.0, None)
    P = np.where(on, np.clip(P, 0.9 * pmin, pcap * 1.01), 0.0)
    op = on.astype(float)
    extra = np.zeros(T)
    s, L, v = runs(on)
    sp = emu['start'][tech]; tp = emu['stop'][tech]
    a1r = rng.uniform(*sp['a1']); rwu = rng.uniform(*sp['r_wu'])
    p_stop = rng.uniform(*tp['p']); rdn = rng.uniform(*tp['r_dn'])
    sh = u.get('start_heat', 5.0) * pmax
    for i in np.flatnonzero(v):
        t0, n = s[i], L[i]
        if t0 == 0:
            continue
        off_before = L[i - 1] if i > 0 else 10 ** 6
        typ = 0 if off_before < HOT_H else (1 if off_before < WARM_H else 2)
        r_ = rwu * (1.0, 0.85, 0.7)[typ]
        op[t0] = rng.uniform(*sp['op1'])
        lvl = pmax * a1r * rng.uniform(0.5, 1.5)
        j = 0
        while t0 + j < T and j < n and lvl < P[t0 + j]:
            P[t0 + j] = max(lvl, 0.0)
            lvl += r_ * pmax; j += 1
        mult = (0.6, 1.0, 1.6)[typ]
        extra[t0:t0 + max(j, 1)] += sh * mult / max(j, 1)
        te = t0 + n - 1
        if n >= 2 and te + 1 < T and rng.random() < p_stop:
            op[te] = rng.uniform(*tp['op_last'])
            P[te] = min(P[te], pmax * rng.uniform(*tp['a_last']))
            lvl = P[te]; k = 1
            while te - k > t0 + j and P[te - k] > lvl + rdn * pmax:
                lvl += rdn * pmax; P[te - k] = lvl; k += 1
    F = fuel_fn(u)
    Fb = F(np.where(op > 0, np.maximum(P, 1e-3), 1e-3)) * op
    HI = Fb + extra
    if rep['chp']:
        HI = HI + op * (rep['phi1'] * Fb / np.maximum(op, 1e-6) + rep['phi2'] * F(pmax))
    HI = HI * np.exp(rng.normal(0, rng.uniform(*emu['hi_sigma']), T))
    # combined-cycle reporting basis: CT-only gross load, split over the CTs that share the block
    if tech == 'gas_cc':
        f = np.clip(P / pmax, 0, 1)
        sct = np.clip(rep['s0'] + rep['s1'] * (1 - f), 0.4, 0.95) if rep['ct_only'] else 1.0
        P = P * sct / rep['n_share']; HI = HI / rep['n_share']
    P = np.round(np.where(op > 0, P, 0.0))
    HI = np.where(op > 0, HI, np.nan)
    meas = rng.random(T) > 0.02
    return dict(op=op.astype(np.float32), P=P.astype(np.float32), HI=HI.astype(np.float32), meas=meas)


def context(u, rec, years, rng, rep, emu=EMU):
    """EIA-860/923-like context of one simulated unit (same keys as realdata.context)."""
    tech = u['tech']; pmax = u['pmax']; pmin = u['pmin']
    ns = rep['n_share']
    if tech == 'gas_cc':
        s0 = rep['s0']
        np_own = pmax * s0 / ns * rng.uniform(1.0, 1.2)
        np_st = pmax * (1 - s0) / ns * rng.uniform(1.0, 1.2) if rng.random() < 0.9 else 0.0
    else:
        np_own = pmax * rng.uniform(1.0, 1.2); np_st = 0.0
    rep_pmin = pmin * ((rep['s0'] + rep['s1'] * (1 - pmin / pmax)) if rep['ct_only'] else 1.0) / ns
    ml = rep_pmin * np.exp(rng.normal(0, emu['minload_sigma'])) if rng.random() > emu['minload_missing'] else None
    c = dict(pm_ct=float(tech in ('gas_ct', 'gas_cc')), pm_st=float(tech in ('coal_steam', 'gas_steam')),
             pm_ca=float(tech == 'gas_cc' and np_st > 0), fuel_coal=float(tech == 'coal_steam'), chp=float(rep['chp']),
             np_own=float(np_own), np_st_share=float(np_st), minload=None if ml is None else float(ml),
             cold_h=float(rng.choice(emu['cold_h'][tech])) if rng.random() > 0.1 else None,
             online_year=float(rng.integers(1960, 2021)))
    # EIA-923 annual net heat rate of the linked generators
    hr = {}
    op, P, HI = rec['op'], rec['P'], np.nan_to_num(rec['HI'])
    for y in np.unique(years):
        m = years == y
        E = float((P[m] * op[m]).sum())
        if E <= 0 or rng.random() < emu['hr923_missing']:
            continue
        fuel = float(HI[m].sum())
        if rep['chp']:
            fuel *= rng.uniform(*emu['chp_alloc'])
        # v1: EIA-923 net generation = gross x (1 - aux).  v2 (dev-calibrated): on the 2019/2021 dev unit-years the
        # EIA-923 net heat rate of non-CHP, non-CC units equals the CAMPD gross one to within ~5 % (median ratio ~1.0):
        # CEMS heat input runs above EIA-923 fuel by about the station-service share, so the two cancel.
        k_aux = (1 - rep['aux']) if emu.get('hr923_basis', 'net') == 'net' else 1.0
        if tech == 'gas_cc':
            s = rep['s0'] if rep['ct_only'] else 1.0
            blk = E * ns / (s if rep['ct_only'] else 1.0)          # block gross MWh
            ct_part = blk * (rep['s0'] if np_st > 0 else 1.0) / ns
            st_all = blk * (1 - rep['s0']) if np_st > 0 else 0.0
            net = (ct_part + st_all) * k_aux
        else:
            net = E * k_aux
        h = fuel / net * np.exp(rng.normal(0, emu['hr923_sigma']))
        if rng.random() < emu['hr923_gross_err']:
            h *= rng.uniform(0.6, 1.6)
        if 5 < h < 25:
            hr[int(y)] = float(h)
    c['hr923'] = hr
    return c
