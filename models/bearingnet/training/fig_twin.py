"""Diagnostic figure: real dev reading vs its v2 twin: wide-band envelope spectrum with defect markers, and the 21-strip feature map."""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, _P.SRC); import bearsim as BS, features as FE, realdata as RD
ids = [int(a) for a in sys.argv[1].split(',')]; chan = sys.argv[2] if len(sys.argv) > 2 else 'DE'
_, ref, f98 = RD.cwru_rec(98); rng = np.random.default_rng(3)
fig, ax = plt.subplots(len(ids), 3, figsize=(18, 3.2 * len(ids)), squeeze=False)
for k, i in enumerate(ids):
    r = [q for q in RD.cwru_readings([i]) if q['kind'] == 'main' and q['chan'] == chan][0]; x = FE.pad_short(r['x'])[:FE.NWIN]
    ex = 10 * np.log10(BS.band_power(x) / BS.band_power(ref[chan][:FE.NWIN])); snr = 10 * np.log10(max(10 ** (ex / 10) - 1, 1e-2))
    P = BS.draw_params(rng, r['cls'], 'v2'); P['snr_db'] = snr
    xt, _ = BS.simulate(rng, ref[chan], f98, cls=r['cls'], geom=r['geom'], size_mm=r['size_mm'], rho=r['f_true'] / f98, P=P, distract=False, aperiodic=False, version='v2')
    o = FE.orders(**r['geom']); F = np.fft.rfftfreq(FE.NWIN, 1 / FE.FS)
    for j, (sig, nm) in enumerate(((x, 'real'), (xt.astype(float), 'twin v2'))):
        ses, _ = FE.envelope_spectrum(np.fft.rfft(sig - sig.mean()), *FE.sk_band(sig)); a = ax[k, j]; m = F < 600
        a.semilogy(F[m], ses[m], lw=0.6); a.set_title('%s %s %s %s ex %.1f dB' % (nm, r['label_txt'], chan, r['loc'], ex))
        for key, c in (('BPFO', 'r'), ('BPFI', 'g'), ('BSF', 'm')):
            for h in (1, 2, 3): a.axvline(h * o[key] * (2 if key == 'BSF' else 1) * r['f_true'], color=c, lw=0.6, alpha=0.6)
        for h in (1, 2, 3): a.axvline(h * r['f_true'], color='k', lw=0.5, ls=':')
    Fr, _, _ = FE.window_features(x, r['geom'], r['f_nom']); Ft, _, _ = FE.window_features(xt.astype(float), r['geom'], r['f_nom'])
    ax[k, 2].imshow(np.concatenate([Fr, np.full((1, 295), 5), Ft]), aspect='auto', vmin=-1, vmax=3); ax[k, 2].set_title('features: real (top) / twin (bottom)')
plt.tight_layout(); plt.savefig(_P.WORK + '/figs/twin_%s_%s.png' % (sys.argv[1].replace(',', '-'), chan), dpi=60)
