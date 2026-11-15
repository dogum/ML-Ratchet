"""Self-commitment of one thermal unit against a price series (a price-taker wrapper around GridSim units).

ERCOT is an energy-only, self-committed market: each unit decides its own on/off schedule from expected prices.
This dynamic programme finds the profit-maximising schedule of one unit with
  - a convex fuel curve F(P) on L output levels between pmin and pmax (MMBtu/h), fuel price per hour, VOM,
  - start costs by start type (hot < 8 h off, warm 8-48 h, cold >= 48 h; RTS-GMLC-style multipliers 0.6/1/1.6),
  - minimum up and down times, hourly ramp limit, start-up / shut-down capability max(pmin, ramp),
  - forced and planned outages (unit must be off).
The schedule is optimised against a FORECAST price (actual price x day noise + hour noise), as a real unit commits
before it knows the price.  Exact DP over states (off for d hours) and (on for k hours, at level l).
"""
import numpy as np

HOT_H, WARM_H = 8, 48


def schedule(price, fuel_price, pmax, pmin, F, vom, start_warm, min_up, min_dn, ramp, outage=None, L=9,
             start_mult=(0.6, 1.0, 1.6), kappa=0.0):
    """Return (on bool[T], P float[T] MW).  F: callable fuel MMBtu/h at output P.  ramp: MW/h."""
    T = len(price)
    lev = np.linspace(pmin, pmax, L)
    fuel_l = np.array([F(p) for p in lev])
    Um = int(max(min_up, 1)); Dm = int(max(min_dn, WARM_H) + 1)
    cap = max(pmin, ramp)
    startable = lev <= cap + 1e-9
    stoppable = lev <= cap + 1e-9
    dl = (pmax - pmin) / max(L - 1, 1)
    rb = int(np.floor(ramp / dl + 1e-9)) if dl > 0 else L
    rb = min(rb, L - 1)
    out = np.zeros(T, bool) if outage is None else np.asarray(outage, bool)
    sc = np.empty(Dm)
    for d in range(Dm):          # off for d+1 hours
        hrs = d + 1
        sc[d] = start_warm * (start_mult[0] if hrs < HOT_H else start_mult[1] if hrs < WARM_H else start_mult[2])
    # profit of each level at each hour
    prof = (price[:, None] - vom) * lev[None, :] - fuel_price[:, None] * fuel_l[None, :]   # (T, L)
    # offer slope: real offer curves rise across the operating range (kappa $/MWh at pmax, quadratic in output);
    # it changes dispatch decisions only, not the physical fuel use
    prof = prof - kappa * (lev[None, :] - pmin) ** 2 / max(pmax - pmin, 1e-6)
    Woff = np.zeros(Dm); Won = np.zeros((Um, L))
    dec_off = np.zeros((T, Dm), np.int16)        # -1 stay off, else start at level
    dec_on = np.zeros((T, Um, L), np.int16)      # -1 stop, else next level
    NEG = -1e18
    for t in range(T - 1, -1, -1):
        p = prof[t]
        # next-state values after acting in hour t
        Woff_next = Woff; Won_next = Won
        # --- off states
        stay = Woff_next[np.minimum(np.arange(Dm) + 1, Dm - 1)]
        if out[t]:
            nWoff = stay; dec_off[t] = -1
        else:
            gain = np.where(startable, p + Won_next[0], NEG)
            bl = int(np.argmax(gain)); bg = gain[bl]
            can = (np.arange(Dm) + 1) >= min_dn
            st_val = np.where(can, bg - sc, NEG)
            take = st_val > stay
            nWoff = np.where(take, st_val, stay)
            dec_off[t] = np.where(take, bl, -1)
        # --- on states
        k_next = np.minimum(np.arange(Um) + 1, Um - 1)
        M = p[None, :] + Won_next[k_next]                      # (Um, L): value of moving to level l'
        best = np.full((Um, L), NEG); arg = np.zeros((Um, L), np.int16)
        for s in range(-rb, rb + 1):                            # sliding max over the ramp band
            lo = max(0, -s); hi = min(L, L - s)
            cand = M[:, lo + s:hi + s]
            upd = cand > best[:, lo:hi]
            best[:, lo:hi] = np.where(upd, cand, best[:, lo:hi])
            arg[:, lo:hi] = np.where(upd, np.arange(lo + s, hi + s)[None, :], arg[:, lo:hi])
        stop_val = Woff_next[0]
        can_stop = ((np.arange(Um) + 1) >= min_up)[:, None] & stoppable[None, :]
        if out[t]:
            nWon = np.full((Um, L), stop_val); dec_on[t] = -1
        else:
            take = can_stop & (stop_val > best)
            nWon = np.where(take, stop_val, best)
            dec_on[t] = np.where(take, -1, arg)
        Woff, Won = nWoff, nWon
    # forward pass from 'off for a long time' (cold)
    on = np.zeros(T, bool); P = np.zeros(T)
    st, d, k, l = 'off', Dm - 1, 0, 0
    for t in range(T):
        if st == 'off':
            a = dec_off[t, d]
            if a < 0:
                d = min(d + 1, Dm - 1)
            else:
                st, k, l = 'on', 0, int(a); on[t] = True; P[t] = lev[l]
        else:
            a = dec_on[t, k, l]
            if a < 0:
                st, d = 'off', 0
            else:
                k = min(k + 1, Um - 1); l = int(a); on[t] = True; P[t] = lev[l]
    return on, P


def outages(T, rng, efor, planned_frac, mttr_h=60.0):
    """Two-state Markov forced outages plus one planned outage block (hours)."""
    lam_rep = 1.0 / mttr_h
    lam_fail = lam_rep * efor / max(1 - efor, 1e-6)
    up = rng.random() > efor
    o = np.zeros(T, bool)
    u = rng.random(T)
    for t in range(T):
        up = (u[t] > lam_fail) if up else (u[t] < lam_rep)
        o[t] = not up
    dur = int(round(planned_frac * T))
    if dur > 0:
        s = int(rng.integers(0, max(T - dur, 1)))
        o[s:s + dur] = True
    return o
