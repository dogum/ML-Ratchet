"""SunDoctor on real data: a California inverter with and without its irradiance sensor, and a Colorado tracker module in snow."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sundoctor
doc = sundoctor.SunDoctor()

def show(title, r):
    s = r['summary']; print(f"\n{title}\n  {r['mode']}: {s['days']} days, {s['kwh_made']:,.0f} kWh made")
    for k, v in s['loss_pct_of_possible'].items():
        if v >= 0.05: print(f"    lost to {k:15s} {v:5.2f}%  ({s['kwh_lost'][k]:,.0f} kWh)")
    return s

ca = dict(lat=38.996, lon=-122.134, tilt=25, azimuth=180, dc_kw=37.2, tz='US/Pacific', n_strings=6)
f = os.path.join(HERE, 'examples', 'pvdaq2107_inverter11_2019H2.csv')
for home in (False, True):
    s = show('PVDAQ 2107, inverter 11, Jun-Nov 2019', doc.diagnose_csv(f, home=home, **ca))
    print('    days a string failed or was repaired:', ', '.join(s['failure_or_repair_days']) or 'none')
print('  (the 23 sibling inverters show one string out from 2019-09-27, restored by 2019-10-16)')

r = doc.diagnose_csv(os.path.join(HERE, 'examples', 'pvdaq9068_module2_2018winter.csv'), lat=40.3864, lon=-104.5512, mount='tracker',
                     max_angle=60, gcr=0.35, backtrack=False, dc_kw=592.25, tz='US/Mountain')
show('PVDAQ 9068, module 2 (single-axis trackers), Dec 2017-Mar 2018', r)
snowy = [(str(d), x) for d, x in zip(r['date'], r['snow']) if x > 0.2]
print('    snow days (over 20% lost):', ', '.join(f"{d} ({100 * x:.0f}%)" for d, x in snowy) or 'none')
