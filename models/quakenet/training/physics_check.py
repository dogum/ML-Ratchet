"""Check the Green's-function synthesis against Syngine's own full synthetics (same model, same source, same receiver) for
random moment tensors, two azimuths. Agreement shows the convention, signs and scale are right."""
import _paths as _P  # portable paths (see _paths.py)
import io, json, urllib.request, numpy as np, sys
from obspy import read
sys.path.insert(0, _P.SRC); import quakesim as QS
U = ('https://service.earthscope.org/irisws/syngine/1/query?model=ak135f_2s&sourcelatitude=0&sourcelongitude=0&sourcedepthinmeters=33000'
     '&sourcemomenttensor={mt}&receiverlatitude={rla}&receiverlongitude={rlo}&components=ZRT&units=velocity&dt=0.5&starttime=P-60&endtime=P%2B2400&format=miniseed&sourcewidth=10')
rng = np.random.default_rng(3); rows = []
for (rla, rlo, az, baz) in [(0, 60, np.pi / 2, None), (0, -60, 3 * np.pi / 2, None), (60, 0, 0.0, None), (-60, 0, np.pi, None), (37.761244, 50.768480, np.radians(45), None)]:
    for k in range(1):
        M = rng.normal(0, 1, 6) * 1e18; st = read(io.BytesIO(urllib.request.urlopen(U.format(mt=','.join(f'{x:.6e}'.replace('+', '') for x in M), rla=rla, rlo=rlo), timeout=120).read()))
        ref = {t.stats.channel[-1]: t.data.astype(float) for t in st}; G = QS.gf(33, 60); X = np.stack(QS.zrt(G, M, az))
        for j, c in enumerate('ZRT'):
            a, b = ref[c], X[j]; L = min(len(a), len(b)); a, b = a[:L], b[:L]
            # source time functions differ (service: 10 s width): compare both below 0.05 Hz, where they do not matter
            from scipy.signal import butter, sosfiltfilt
            sos = butter(4, [0.008, 0.03], 'band', fs=2.0, output='sos'); a, b = sosfiltfilt(sos, a), sosfiltfilt(sos, b)
            cc = float(np.corrcoef(a, b)[0, 1]); ratio = float(np.std(b) / np.std(a))
            rows.append(dict(az=round(float(np.degrees(az))), comp=c, corr=cc, amp_ratio=ratio)); print(rows[-1], flush=True)
json.dump(rows, open(_P.WORK + '/physics_check.json', 'w'), indent=1)
