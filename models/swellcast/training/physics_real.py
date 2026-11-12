"""Physics check on one well-documented real event: Hurricane Marie (East Pacific, Cat 5, peak 140 kt on 2014-08-24 18 UTC
near 15.9N 111.9W per IBTrACS). Its long-period swell hit Southern California on 26-27 Aug 2014. These buoy pairs are
'crowded' (Karina and Lowell were active at the same time), so they are in neither the dev set nor the exam.

Check: (1) fit Munk's straight ridge (classical.read, grid search + least squares) to the real 3-hourly buoy spectrogram;
compare the fitted distance and origin time with the IBTrACS position and peak time. (2) Compare the observed arrival time of
the 0.06 Hz component (peak energy excess at 0.06 Hz) with the analytic arrival t0 + D / c_g(0.06 Hz).
Writes physics_real.json and figs/marie2014.png."""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import realdata as RD, swellsim as SS, classical as CL, geo

ROOT = RD.ROOT
LAT, LON, T0 = 15.9, -111.9, 391362.0            # IBTrACS peak wind, hours since 1970 (2014-08-24 18 UTC)
BUOYS = {'46025': (33.749, -119.053), '46086': (32.491, -118.034), '46069': (33.670, -120.200), '46047': (32.433, -119.533)}

def h2s(h): return str(np.datetime64(int(h), 'h'))

if __name__ == '__main__':
    out = dict(event='Hurricane Marie 2014 (IBTrACS 2014234N12261)', storm=dict(lat=LAT, lon=LON, t0=h2s(T0)), buoys=[])
    fig, axs = plt.subplots(1, len(BUOYS), figsize=(4 * len(BUOYS), 3.6), sharey=True)
    TC = CL.TC
    for ax, (b, (bl, bo)) in zip(axs, BUOYS.items()):
        D = float(geo.dist(LAT, LON, bl, bo)); tarr = T0 + D / SS.cg(0.06) / 3600
        (E, C, S), tc = RD.window(b, tarr)
        if not np.isfinite(E).any(): continue
        r = CL.read(np.stack([E, C, S])); t0_rel = T0 - tc
        L = np.log10(np.maximum(np.where(np.isfinite(E), E, 0.005), 0.005)); base = np.nanpercentile(L, 25, axis=0); A = L - base
        j = int(np.argmin(np.abs(SS.FGRID - 0.0625))); i_obs = int(np.nanargmax(A[:, j])); t_obs = TC[i_obs]
        t_pred = t0_rel + D / SS.cg(SS.FGRID[j]) / 3600
        rec = dict(buoy=b, D_true_km=D / 1e3, D_fit_km=r['D_km'], D_err_pct=100 * (r['D_km'] / (D / 1e3) - 1), t0_true_rel_h=t0_rel, t0_fit_rel_h=r['t0'],
                   t0_err_h=r['t0'] - t0_rel, bearing_true=float(geo.bearing(bl, bo, LAT, LON)), dir_fit=r['dir'],
                   arrival_0625Hz_pred=h2s(tc + t_pred), arrival_0625Hz_obs=h2s(tc + t_obs), arrival_err_h=float(t_obs - t_pred))
        out['buoys'].append(rec); print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()}))
        ax.pcolormesh(TC, SS.FGRID, A.T, cmap='magma', vmin=0, vmax=2.0, shading='nearest')
        f = np.linspace(0.04, 0.10, 20)
        ax.plot(t0_rel + D / SS.cg(f) / 3600, f, 'c--', lw=1.5, label='IBTrACS ridge')
        k = 4 * np.pi * r['D_km'] * 1e3 / 9.81 / 3600; ax.plot(r['t0'] + k * f, f, 'w-', lw=1, label='fitted ridge')
        ax.set_xlim(TC[0], TC[-1]); ax.set_ylim(0.035, 0.12); ax.set_title(f'{b}  D {D / 1e3:.0f} km, fit {r["D_km"]:.0f} km', fontsize=9)
        ax.set_xlabel(f'hours from {h2s(tc)}')
    axs[0].set_ylabel('frequency (Hz)'); axs[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout(); os.makedirs(os.path.join(ROOT, 'figs'), exist_ok=True); fig.savefig(os.path.join(ROOT, 'figs/marie2014.png'), dpi=110)
    R = out['buoys']
    out['summary'] = dict(median_abs_D_err_pct=float(np.median([abs(x['D_err_pct']) for x in R])), median_abs_t0_err_h=float(np.median([abs(x['t0_err_h']) for x in R])),
                          median_abs_arrival_err_h=float(np.median([abs(x['arrival_err_h']) for x in R])))
    print(out['summary']); json.dump(out, open(os.path.join(ROOT, 'physics_real.json'), 'w'), indent=1)
