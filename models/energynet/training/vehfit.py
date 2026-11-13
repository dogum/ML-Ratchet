"""Label-matched twins: tune a generic completed spec so that it reproduces the vehicle's UNADJUSTED (lab) city and
highway mpg from fueleconomy.gov. Two knobs: engine efficiency scale (level) and road-load scale k (CdA and Crr x k;
mostly sets the highway/city ratio). City = FTP-75 approximation 0.43 x cold UDDS + 0.57 x hot UDDS (fuel per mile)."""
import copy
from common import *
from roadsim import simulate, make_trip

_TR = {}


def _trip(c):
    if c not in _TR: _TR[c] = make_trip(c, T_amb=24.0)
    return _TR[c]


def cycle_fuel_per_mi(spec, cyc, cold=False, soc=None):
    """gal/mi on a cycle (fuel corrected to zero net battery change roughly: HEV SOC is mid-window and A-ECMS holds it)."""
    it = None if cold else warm_temps(24)
    o = simulate(spec, _trip(cyc), init_temps=it, init_soc=soc, record=False)
    s = o['summary']
    return 1.0 / max(s['mpg'], 1e-3)


def city_hwy(spec):
    c = 0.43 * cycle_fuel_per_mi(spec, 'udds', cold=True) + 0.57 * cycle_fuel_per_mi(spec, 'udds')
    h = cycle_fuel_per_mi(spec, 'hwfet')
    return 1 / c, 1 / h


def scaled(spec, eff, k):
    s = copy.deepcopy(spec)
    base = fill_like(s)
    s['CdA_m2'] = base['CdA_m2'] * k; s['Crr'] = base['Crr'] * k
    s.setdefault('engine', {})['eff_scale'] = eff
    return s


def fill_like(s):
    from roadsim.vehicle import fill_spec
    f = fill_spec(s)
    return {'CdA_m2': f['CdA_m2'], 'Crr': f['Crr']}


def label_match(spec, ucity, uhwy, ks=(0.75, 1.0, 1.35, 1.8)):
    """returns (spec, info). Fits k on the hwy/city ratio (log-interp over ks), then eff on the geometric-mean level."""
    tgt_ratio = np.log(uhwy / ucity)
    rs = []
    for k in ks:
        c, h = city_hwy(scaled(spec, 1.0, k)); rs.append((np.log(h / c), c, h))
    r = np.array([x[0] for x in rs])
    o = np.argsort(r)
    k = float(np.exp(np.interp(tgt_ratio, r[o], np.log(np.array(ks))[o])))
    c, h = city_hwy(scaled(spec, 1.0, k))
    eff = float(np.sqrt(ucity * uhwy / (c * h)))
    s = scaled(spec, eff, k)
    c2, h2 = city_hwy(s)
    return s, dict(k=k, eff=eff, city=c2, hwy=h2, ucity=ucity, uhwy=uhwy)
