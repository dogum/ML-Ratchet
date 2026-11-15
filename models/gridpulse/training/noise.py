"""Quiet 1 s GB frequency segments from the train-input years (splits.json frequency_noise: 2018, 2020, 2022,
>= 1 h from any detected event). Only event onset TIMES are used (for exclusion), never their labels.
Output data/noise.npz: seg (n, L) float32 Hz deviation from 50 Hz, day id, month."""
from common import *
import griddata as gd

L = 140          # seconds per segment (60 s pre + 68 s post + slack)


def build():
    sp = gd.splits()['frequency_noise']
    ev = gd.events('GB')
    t_ev = pd.to_datetime(ev.t0_utc, utc=True).values.astype('datetime64[s]').astype(np.int64)
    segs, days, months = [], [], []
    for m in gd.frequency_months():
        if int(m[:4]) not in sp['years']:
            continue
        start = pd.Timestamp(m + '-01', tz='UTC'); end = start + pd.offsets.MonthBegin(1)
        meta, t, f = gd.frequency('GB', str(start), str(end))
        t0 = int(pd.Timestamp(meta['start_utc']).timestamp())
        n = len(f) // L
        F = f[:n * L].reshape(n, L)
        ts = t0 + np.arange(n) * L + t[0]
        ok = ~np.isnan(F).any(1)
        # >= 1 h from any event (segment span [ts, ts+L])
        near = np.zeros(n, bool)
        for te in t_ev:
            near |= (ts + L > te - 3600) & (ts < te + 3600)
        ok &= ~near
        # also drop sub-threshold disturbances: largest 10 s fall or rise beyond 0.10 Hz
        d10 = F[:, 10:] - F[:, :-10]
        ok &= (np.nanmin(d10, 1) > -0.10) & (np.nanmax(d10, 1) < 0.10)
        segs.append((F[ok] - 50.0).astype(np.float32))
        days.append(((ts[ok] - ts[0]) // 86400).astype(np.int16))
        months.append(np.full(ok.sum(), int(m.replace('-', '')), np.int32))
        print(m, n, int(ok.sum()), 'std', float(np.nanstd(F[ok])), flush=True)
    np.savez_compressed(f'{DATA}/noise.npz', seg=np.concatenate(segs), day=np.concatenate(days),
                        month=np.concatenate(months))


if __name__ == '__main__':
    build()
