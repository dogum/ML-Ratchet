"""Build drive cycles (1 Hz, m/s) into roadsim/data/cycles.npz.

Sources:
- EPA dynamometer drive schedules (epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules):
  UDDS, HWFET, US06, SC03, NYCC, LA92.
- 40 CFR 1037 Appendix A (eCFR): heavy-duty transient chassis cycle (CARB HHDDT transient mode).
- 40 CFR 1037 Appendix D (eCFR): grade profile for the 55 / 65 mph highway cruise cycles (GEM).
- CBD14 transit-bus cycle constructed from the SAE J1376 description (14 x [7 s idle, 0-20 mph, cruise, 2.5 mph/s decel],
  ~2 mi, ~600 s); the tabulated ADVISOR file was not reachable.
"""
import os, re, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, '..', 'refdata')
OUT = os.path.join(HERE, '..', 'roadsim', 'data')
MPH = 0.44704


def read_epa_txt(fn):
    t, v = [], []
    raw = open(os.path.join(REF, 'cycles', fn), 'rb').read()
    txt = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('latin-1')
    for line in txt.splitlines():
        p = line.replace(',', ' ').split()
        if len(p) >= 2:
            try:
                a, b = float(p[0]), float(p[1])
            except ValueError:
                continue
            t.append(a); v.append(b)
    t = np.array(t); v = np.array(v) * MPH
    tt = np.arange(0, int(t.max()) + 1)
    return np.interp(tt, t, v)


def ecfr_text(fn):
    t = open(os.path.join(REF, 'ecfr', fn)).read()
    t = re.sub('<[^>]+>', ' ', t)
    return ' '.join(t.split()).replace('−', '-').replace('−', '-')


cyc = {}
meta = {}
for name, fn in [('udds', 'uddscol.txt'), ('hwfet', 'hwycol.txt'), ('us06', 'us06col.txt'), ('sc03', 'sc03col.txt'),
                 ('nycc', 'epa-new-york-city-cycle.txt'), ('la92', 'la92col.txt')]:
    cyc[name] = read_epa_txt(fn)
    meta[name] = 'EPA dynamometer drive schedule ' + fn

# HD transient (40 CFR 1037 App A)
s = ecfr_text('app_A_1037.xml')
nums = [float(x) for x in re.findall(r'(?<![\w.])-?\d+\.?\d*', s.split('Speed (mi/hr)', 1)[1])]
pairs = np.array(nums[: (len(nums) // 2) * 2]).reshape(-1, 2)
# keep only consecutive time stamps
ok = [0]
for i in range(1, len(pairs)):
    if pairs[i, 0] == pairs[ok[-1], 0] + 1:
        ok.append(i)
pairs = pairs[ok]
cyc['hd_transient'] = np.concatenate([[0.0], pairs[:, 1] * MPH])
meta['hd_transient'] = '40 CFR 1037 Appendix A heavy-duty transient test cycle (CARB HHDDT transient mode), %d s' % len(pairs)

# GEM highway cruise with grade (40 CFR 1037 App D)
s = ecfr_text('app_D_1037.xml')
body = s.split('Grade (%)', 1)[1]
nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', body)]
gp = np.array(nums[: (len(nums) // 2) * 2]).reshape(-1, 2)
keep = [0]
for i in range(1, len(gp)):
    if gp[i, 0] > gp[keep[-1], 0]:
        keep.append(i)
gp = gp[keep]
grade_dist, grade_pct = gp[:, 0], gp[:, 1]
for mph in (55, 65):
    v = mph * MPH
    L = grade_dist[-1]
    T = int(np.ceil(L / v))
    # 1037.510: start at speed (we include a short ramp at start for forward sim)
    tt = np.arange(T + 1)
    vv = np.full(T + 1, v)
    d = np.minimum(tt * v, L)
    g = np.interp(d, grade_dist, grade_pct) / 100.0
    cyc['gem_%d' % mph] = vv
    cyc['gem_%d_grade' % mph] = g
    meta['gem_%d' % mph] = '%d mph cruise over the 40 CFR 1037 Appendix D grade profile (%.1f km), starts at speed' % (mph, L / 1000)

# CBD14 (SAE J1376 description)
seg = []
seg += [0.0] * 7
a = 20 * MPH / 14.0
seg += list(np.arange(1, 15) * a)
dist_acc = 0.5 * 20 * MPH * 14
dist_dec = 0.5 * 20 * MPH * 8
cruise = int(round((0.143 * 1609.344 - dist_acc - dist_dec) / (20 * MPH)))
seg += [20 * MPH] * cruise
seg += list(20 * MPH - np.arange(1, 9) * 2.5 * MPH)
seg = np.maximum(np.array(seg), 0)
cbd = np.concatenate([[0.0]] + [seg] * 14)
cyc['cbd14'] = cbd
meta['cbd14'] = 'CBD14 bus cycle constructed from SAE J1376 description: 14 x (7 s idle, 0-20 mph in 14 s, %d s cruise, 2.5 mph/s decel)' % cruise

for k, v in cyc.items():
    if k.endswith('_grade'):
        continue
    d = np.sum(0.5 * (v[1:] + v[:-1]))
    print(f'{k:14s} {len(v):5d} s  {d/1609.344:6.2f} mi  vmax {v.max()/MPH:5.1f} mph  avg {d/len(v)/MPH:5.1f} mph')
np.savez_compressed(os.path.join(OUT, 'cycles.npz'), **{k: v.astype(np.float32) for k, v in cyc.items()})
json.dump(meta, open(os.path.join(OUT, 'cycles_meta.json'), 'w'), indent=1)
