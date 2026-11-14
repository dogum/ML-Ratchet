"""Human driver model (RoadSim physics v2, work package B3; option `driver_model: 'human'`).

v1's driver is a near-perfect follower: feed-forward to the next second's target plus PI, so it misses the trace only
at the powertrain's limits. That stays the default. The human driver follows the same speed target the way people do.
Everything it sees is tau old (perception-reaction delay): its own speed v_s = v(t - tau) and the target from t' = t - tau.

    a_track = (v_tgt(t' + T_p) - v_s) / T_p                  smooth tracking with anticipation T_p
    a_up    = (max v_tgt over [t', t' + T_up] - v_s - db) / T_g   speeding up: head for the coming top speed (T_up 0: off)
    a_dn    = min over s in [1 s, T_dn], v_tgt(t' + s) < v_s - db, of (v_tgt(t' + s) - v_s + db) / s
                                                             slowing down: be at the coming lower speeds in time
    a_des   = min(max(a_track, a_up), a_dn)
    a_cmd   = clip(a_des, a_min, a_max), slewed by |da/dt| <= j_max          comfort limits and jerk limit
    a(t)    : first-order pedal / brake actuation lag tau_a on a_cmd

Both the speed and the target are seen tau late, so a steady ramp is tracked without offset; at a change of slope the
delay and the actuation lag make the driver overshoot (more with a short T_p). db = 0.3 m/s is a speed-perception
deadband (about 1 km/h) so a driver does not chase the trace's noise. At launch the driver waits tau for the
"green light"; when the target says stop and the car is slow, it brakes to a halt.

One parameter, aggressiveness g in [0, 1] (option `driver_aggressiveness`), sets every number above, interpolated
between three anchor drivers:
  eco (0)        long anticipation (T_p 3 s, slows for anything in the next 5 s), gentle (1.5 m/s^2, 1.5 m/s^3)
  typical (0.5)  T_p 1.6 s, heads for the top speed of the next 2 s, 2.6 m/s^2, 4 m/s^3
  aggressive (1) T_p 0.8 s, rushes to the top speed of the next 6 s (T_g 1 s), brakes late (looks 1.5 s ahead),
                 4 m/s^2, -6 m/s^2, 10 m/s^3
The anchors follow published naturalistic-driving ranges: comfortable acceleration ~1.5 m/s^2 and jerk < 2 m/s^3 for
calm drivers, 3-4 m/s^2 and jerk up to ~10 m/s^3 for aggressive ones (Bosetti, Da Lio & Saroldi 2014, Eur. Transp.
Res. Rev. 6:235; Murphey, Milton & Kiliaris 2009, IEEE CIVVS, jerk-based driving-style classes). The delay tau (option
`driver_reaction_s`, 0.5 s) is a tracking-task reaction time (Green 2000, Transp. Hum. Factors 2:195: 0.7-0.75 s for
expected braking events, less in continuous tracking). `rng` (simulate's) jitters each parameter by +-15 %.
"""
import math

ANCHORS = {   # T_p s, T_up s, T_g s, T_dn s, a_max m/s2, a_min m/s2, jerk m/s3, tau_a s
    0.0: dict(T_p=3.0, T_up=0.0, T_g=3.0, T_dn=5.0, a_max=1.5, a_min=-2.5, jerk=1.5, tau_a=0.35),     # eco
    0.5: dict(T_p=1.6, T_up=2.0, T_g=2.0, T_dn=2.5, a_max=2.6, a_min=-4.0, jerk=4.0, tau_a=0.25),     # typical
    1.0: dict(T_p=0.8, T_up=6.0, T_g=1.0, T_dn=1.5, a_max=4.0, a_min=-6.0, jerk=10.0, tau_a=0.15),    # aggressive
}
STOP_SPEED = 1.0          # m/s: below this, with the target at zero, brake to a halt
N_DN = 5                  # look-ahead points for a_dn
S_DN_MIN = 1.0            # s: shortest look-ahead for a_dn (nobody sheds speed in a fraction of a second)
DB = 0.3                  # m/s: speed-perception deadband for a_up / a_dn


def params(g):
    g = min(max(float(g), 0.0), 1.0)
    lo, hi = (0.0, 0.5) if g <= 0.5 else (0.5, 1.0)
    w = (g - lo) / (hi - lo)
    return {k: ANCHORS[lo][k] + w * (ANCHORS[hi][k] - ANCHORS[lo][k]) for k in ANCHORS[0.0]}


class HumanDriver:
    def __init__(self, vt_sub, h, aggressiveness=0.5, reaction_s=0.5, rng=None, a_veh=(4.5, -7.5)):
        import numpy as np
        p = params(aggressiveness)
        if rng is not None:
            p = {k: v * float(rng.uniform(0.85, 1.15)) for k, v in p.items()}
        self.p = p
        self.vt = list(vt_sub); self.n = len(self.vt); self.h = h
        self.tau = max(float(reaction_s), 0.0)
        self.a_veh_max, self.a_veh_min = a_veh
        self.v_hist = []
        self.a_cmd = 0.0; self.a_act = 0.0
        self.k_act = 1.0 - math.exp(-h / max(p['tau_a'], 1e-3))
        # running maximum of the target over the next T_up (on the step grid)
        w = int(round(p['T_up'] / h))
        if w > 0:
            x = np.asarray(self.vt, float)
            xp = np.concatenate([x, np.full(w, x[-1])])
            self.vmax = np.lib.stride_tricks.sliding_window_view(xp, w + 1).max(axis=1)[:self.n].tolist()
        else:
            self.vmax = None
        T_dn = max(p['T_dn'], S_DN_MIN)
        self.s_dn = [S_DN_MIN + (T_dn - S_DN_MIN) * j / max(N_DN - 1, 1) for j in range(N_DN)]

    @staticmethod
    def _interp(arr, x, n):
        if x <= 0.0:
            return arr[0]
        if x >= n - 1:
            return arr[-1]
        i = int(x); f = x - i
        return arr[i] + f * (arr[i + 1] - arr[i])

    def _seen(self, i, v0):
        """speed seen tau ago (linear interpolation of the step history)."""
        d = self.tau / self.h
        if d <= 0.0:
            return v0
        j = i - d
        if j <= 0.0:
            return self.v_hist[0]
        lo = int(j); f = j - lo
        a = self.v_hist[lo]; b = self.v_hist[lo + 1] if lo + 1 < len(self.v_hist) else v0
        return a + f * (b - a)

    def accel(self, i, v0):
        """commanded acceleration for step i (from v0 at its start)."""
        self.v_hist.append(v0)
        p = self.p; h = self.h; vt = self.vt; n = self.n
        ts = i - self.tau / h                       # perceived time, in steps
        v_s = self._seen(i, v0)
        v_now = self._interp(vt, ts, n)
        if v_now <= 0.0 and v0 < 0.05:
            # stopped and the (perceived) target is still zero: wait (launch reacts tau after the target moves)
            self.a_cmd = 0.0; self.a_act = 0.0
            return 0.0
        v_ref = self._interp(vt, ts + p['T_p'] / h, n)
        if vt[i] <= 0.0 and v_ref <= 0.0 and v0 < STOP_SPEED:
            self.a_cmd = 0.0; self.a_act = -v0 / h
            return max(-v0 / h, self.a_veh_min)
        a_des = (v_ref - v_s) / p['T_p']
        if self.vmax is not None:
            a_des = max(a_des, (self._interp(self.vmax, ts, n) - v_s - DB) / p['T_g'])
        for s in self.s_dn:            # only coming speeds below the current one constrain (slowing down)
            vs = self._interp(vt, ts + s / h, n)
            if vs < v_s - DB:
                a_des = min(a_des, (vs - v_s + DB) / s)
        lo = p['a_min']
        if v_s - v_now > 1.5:        # well above the target (overshot a slow-down): brake as hard as needed
            lo = self.a_veh_min
        a_des = min(max(a_des, lo), p['a_max'])
        dj = p['jerk'] * h
        self.a_cmd = min(max(a_des, self.a_cmd - dj), self.a_cmd + dj)
        self.a_act += (self.a_cmd - self.a_act) * self.k_act
        return min(max(self.a_act, self.a_veh_min), self.a_veh_max)
