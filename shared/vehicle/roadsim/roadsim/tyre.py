"""Tyre model (RoadSim physics v2, work package B3; option `tyre_model`).

What it adds to the 1 Hz v1 road-load model:

* Axle loads with static split, grade and dynamic (acceleration) load transfer:
      N_f = m g (b cos(th) - h sin(th))/L - m a h/L,    N_r = m g (a_cg cos(th) + h sin(th))/L + m a h/L
  (Gillespie 1992 ch. 2; a_cg / b = CG distance behind the front / ahead of the rear axle, h = CG height).
* A friction ellipse per axle: the longitudinal force an axle can carry is N sqrt(mu^2 - (a_y/g)^2), with a_y from the
  trip's heading rate (v * dpsi/dt) and the peak friction mu by surface (MU_PEAK).
  - traction is limited on the driven axle(s);
  - braking (all wheels, ABS) is limited by the whole vehicle's friction;
  - regenerative braking is limited to `tyre_regen_mu_frac` of the driven axle's friction (the brake controller keeps
    a stability margin), which replaces v1's fixed `regen.max_decel_g`.
* Longitudinal slip from the brush model with a parabolic pressure distribution (Pacejka, Tire and Vehicle Dynamics,
  2012, sec. 3.2): with theta = C_x / (3 mu N) and f = |F_x| / (mu N),
      f = 1 - (1 - theta |sigma|)^3   ->   sigma = (1 - (1 - f)^(1/3)) / theta,     kappa = sigma / (1 - sigma)
  where kappa = (omega r - v) / v is the practical slip. The wheels then turn at v (1 + kappa): in traction the
  powertrain supplies F v (1 + kappa), in braking it absorbs |F| v (1 + kappa). The difference F v kappa is the slip
  loss, booked in the ledger term `tyre_slip`.
* Cornering drag (lateral slip loss) F_c = m a_y^2 / (g c_alpha) (each axle at slip angle a_y / (g c_alpha), neutral
  steer), also booked in `tyre_slip`.
* EU tyre-label rolling-resistance classes A-E (Regulation (EU) 2020/740, Annex I): if the spec gives
  `tire.eu_class`, Crr is rescaled by class_mid / ref_class_mid (the spec's Crr is taken to be the option
  `tyre_ref_eu_class`, default C). Surface 5 = ice (v1 knows 0-4).

Vehicle fields read here with defaults (not in vehicle.py): spec['chassis'] = {cg_height_m, wheelbase_m,
front_axle_load_frac, drive: 'FWD'|'RWD'|'AWD', height_m}, spec['tire'] = {eu_class, size ('205/55R16'),
slip_stiffness_per_N, cornering_stiffness_per_N_rad}. Every tunable number is in the tables below.
"""
import math
import re
import numpy as np

G = 9.80665

# peak tyre-road friction by surface code: 0 dry asphalt, 1 dry concrete, 2 wet, 3 gravel, 4 packed snow, 5 ice
# (Wong, Theory of Ground Vehicles 2008, Table 1.3: asphalt/concrete dry 0.8-0.9, wet asphalt 0.5-0.7, gravel 0.6,
#  snow packed 0.2, ice 0.1; modern passenger tyres sit at the top of the dry range)
MU_PEAK = [0.90, 0.90, 0.60, 0.60, 0.25, 0.10]
# rolling-resistance factor by surface (v1's SURFACE_RR plus ice: like a hard dry surface)
SURFACE_RR6 = [1.0, 0.95, 1.12, 1.6, 2.2, 1.0]

# EU label class -> mid-point rolling-resistance coefficient (kg/t = 1e-3), Regulation (EU) 2020/740 Annex I table A
# C1 (cars): A <= 6.5, B 6.6-7.7, C 7.8-9.0, D 9.1-10.5, E >= 10.6;  C3 (trucks, buses): A <= 4.0, B 4.1-5.0,
# C 5.1-6.0, D 6.1-7.0, E >= 7.1 (open-ended classes take a half-class step past the bound)
EU_CRR = {'C1': dict(A=6.0, B=7.15, C=8.4, D=9.8, E=11.3),
          'C3': dict(A=3.7, B=4.55, C=5.55, D=6.55, E=7.6)}

# chassis defaults by class: CG height m, wheelbase m, front axle static load share, drive axle, overall height m
# (Gillespie 1992 table 2.x and typical published specs; the Class 8 tractor-trailer is folded into an equivalent
#  two-axle vehicle whose 'rear' is the tandem drive axle carrying ~42 % of 80,000 lb)
CHASSIS = {
    'car':      dict(cg_height_m=0.55, wheelbase_m=2.70, front_axle_load_frac=0.60, drive='FWD', height_m=1.45),
    'suv':      dict(cg_height_m=0.65, wheelbase_m=2.80, front_axle_load_frac=0.57, drive='FWD', height_m=1.70),
    'pickup':   dict(cg_height_m=0.75, wheelbase_m=3.60, front_axle_load_frac=0.57, drive='RWD', height_m=1.90),
    'van':      dict(cg_height_m=0.70, wheelbase_m=3.10, front_axle_load_frac=0.55, drive='FWD', height_m=2.00),
    'bus':      dict(cg_height_m=1.10, wheelbase_m=6.10, front_axle_load_frac=0.35, drive='RWD', height_m=3.20),
    'truck_md': dict(cg_height_m=1.30, wheelbase_m=5.00, front_axle_load_frac=0.35, drive='RWD', height_m=3.50),
    'truck_hd': dict(cg_height_m=1.90, wheelbase_m=5.80, front_axle_load_frac=0.58, drive='RWD', height_m=4.10),
}
BEV_CHASSIS = dict(cg_height_m=0.50, front_axle_load_frac=0.50, drive='RWD')   # skateboard pack, rear motor (car/suv)
# brush-model longitudinal slip stiffness C_x / F_z (1 / unit slip) and cornering stiffness C_alpha / F_z (1/rad)
# (passenger radials C_x ~ 15-20 F_z, C_alpha ~ 0.2 F_z per degree; truck tyres about half: Pacejka 2012 ch. 1, Wong 2008)
SLIP_STIFF = dict(light=15.0, heavy=8.0)
CORNER_STIFF = dict(light=11.5, heavy=6.0)


def tyre_size(size):
    """'205/55R16' -> (width m, unloaded radius m), or None."""
    m = re.match(r'^\s*(?:[A-Z]+)?(\d{3})/(\d{2})\s*[A-Z]*R?\s*(\d{2}(?:\.\d)?)', str(size or '').upper())
    if not m:
        return None
    w = int(m.group(1)) / 1000.0; ar = int(m.group(2)) / 100.0; rim = float(m.group(3)) * 0.0254
    return w, 0.5 * rim + w * ar


def surface_codes(trip_inputs, N):
    """surface codes 0..5 (v1's normalize_trip clips to 0..4; ice is 5)."""
    s = trip_inputs.get('surface')
    if s is None:
        return np.zeros(N, int)
    s = np.asarray(s, float)
    if s.ndim == 0:
        s = np.full(N, float(s))
    return np.clip(np.nan_to_num(s), 0, 5).astype(int)


def lateral_accel(trip, dt=1.0):
    """|a_y| (m/s^2) per trip second from the heading rate: v * dpsi/dt (3 s smoothing, yaw rate <= 30 deg/s)."""
    h = np.unwrap(np.radians(np.asarray(trip['heading'], float)))
    v = np.asarray(trip['v'], float)
    if len(h) < 3 or np.ptp(h) == 0.0:
        return np.zeros(len(v))
    r = np.gradient(h, dt)
    r = np.convolve(r, np.ones(3) / 3.0, mode='same')
    r = np.clip(r, -math.radians(30.0), math.radians(30.0))
    return np.minimum(np.abs(v * r), 0.9 * G)


class Tyre:
    """compiled per vehicle (veh._b3_tyre)."""

    def __init__(self, veh):
        s = veh.spec
        cls = veh.cls
        ch = dict(CHASSIS[cls])
        if veh.arch == 'bev' and cls in ('car', 'suv'):
            ch.update(BEV_CHASSIS)
        ch.update({k: v for k, v in (s.get('chassis') or {}).items() if v is not None})
        self.h = float(ch['cg_height_m']); self.L = float(ch['wheelbase_m'])
        self.ff = float(ch['front_axle_load_frac']); self.drive = str(ch['drive']).upper()
        self.height = float(ch['height_m'])
        t = s.get('tire') or {}
        heavy = veh.hd
        self.cx = float(t.get('slip_stiffness_per_N', SLIP_STIFF['heavy' if heavy else 'light']))
        self.ca = float(t.get('cornering_stiffness_per_N_rad', CORNER_STIFF['heavy' if heavy else 'light']))
        sz = tyre_size(t.get('size'))
        if sz is not None and 'slip_stiffness_per_N' not in t:
            self.cx *= (sz[0] / 0.205) ** 0.5          # wider tread, longer stiffer brush (mild)
        self.size = sz
        self.eu_class = t.get('eu_class')
        self.group = 'C3' if heavy else 'C1'
        self.m = veh.mass

    # ------------------------------------------------------------------ loads
    def axle_loads(self, a, sin_t, cos_t):
        """(N_front, N_rear) in N for longitudinal acceleration a (m/s^2) on grade angle th."""
        mg = self.m * G
        b = self.ff * self.L            # CG to rear axle
        ac = self.L - b                 # CG to front axle
        tr = (self.m * a + mg * sin_t) * self.h / self.L
        Nf = mg * cos_t * b / self.L - tr
        Nr = mg * cos_t * ac / self.L + tr
        lo = 0.05 * mg * cos_t          # an axle never fully unloads (wheelie / stoppie are out of scope)
        return max(Nf, lo), max(Nr, lo)

    def drive_load(self, a, sin_t, cos_t):
        Nf, Nr = self.axle_loads(a, sin_t, cos_t)
        if self.drive == 'FWD':
            return Nf
        if self.drive == 'RWD':
            return Nr
        return Nf + Nr

    # ------------------------------------------------------------------ friction limits
    @staticmethod
    def mu_long(mu, ayg):
        """longitudinal friction left on the ellipse with lateral use a_y/g."""
        return math.sqrt(max(mu * mu - ayg * ayg, 0.04 * mu * mu))

    def limits(self, a, sin_t, cos_t, mu, ayg, regen_frac):
        """-> (traction cap N, braking cap N (all wheels, ABS), regen cap N (driven axle with the stability margin),
        driven-axle load N, total load N, longitudinal friction left mu_x)."""
        Nd = self.drive_load(a, sin_t, cos_t)
        ml = self.mu_long(mu, ayg)
        Ntot = self.m * G * cos_t
        return ml * Nd, ml * Ntot, regen_frac * ml * Nd, Nd, Ntot, ml

    def step(self, F, u, a, sin_t, cos_t, mu, ayg, regen_frac, env, regen=False):
        """friction-limit the requested tyre force F (N) and return (F_ask, wheel-rim speed u (1 + kappa)).
        Sets env['tyre_regen_cap'], env['tyre_kappa'], env['tyre_lim']. In braking the slip is taken on the driven
        axle when the vehicle regenerates (regen first: the driven axle carries the braking), else on all wheels."""
        cap_t, cap_b, cap_r, Nd, Nt, ml = self.limits(a, sin_t, cos_t, mu, ayg, regen_frac)
        env['tyre_regen_cap'] = cap_r
        lim = False
        if F > cap_t:
            F = cap_t; lim = True
        elif F < -cap_b:
            F = -cap_b; lim = True
        if F >= 0.0:
            kap = self.slip(F, Nd, ml)
        elif regen:
            kap = self.slip(max(F, -cap_r), Nd, ml)
        else:
            kap = self.slip(F, Nt, ml)
        env['tyre_kappa'] = kap; env['tyre_lim'] = lim
        return F, u * (1.0 + kap)

    # ------------------------------------------------------------------ slip
    def slip(self, F, N, mu):
        """practical slip kappa (signed like F) carrying force F on tyres with total load N (brush model)."""
        if F == 0.0 or N <= 0.0:
            return 0.0
        muN = mu * N
        f = min(abs(F) / muN, 0.999)
        theta = self.cx / (3.0 * mu)
        sig = (1.0 - (1.0 - f) ** (1.0 / 3.0)) / theta
        sig = min(sig, 0.9)
        if F < 0.0:
            sig = -sig
        return sig / (1.0 - sig)          # theoretical slip sigma = kappa / (1 + kappa), both signs

    def corner_drag(self, ay):
        """lateral-slip (cornering) drag force N at lateral acceleration ay."""
        return self.m * ay * ay / (G * self.ca)

    # ------------------------------------------------------------------ rolling resistance class
    def crr_ratio(self, ref_class):
        if not self.eu_class:
            return 1.0
        tab = EU_CRR[self.group]
        c = str(self.eu_class).upper()[:1]; r = str(ref_class or 'C').upper()[:1]
        if c not in tab or r not in tab:
            raise ValueError(f'EU tyre class must be A-E (got {self.eu_class!r} / ref {ref_class!r})')
        return tab[c] / tab[r]


def get(veh):
    t = getattr(veh, '_b3_tyre', None)
    if t is None:
        t = Tyre(veh)
        veh._b3_tyre = t
    return t
