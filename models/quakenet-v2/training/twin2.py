"""Twin test on the dev events: re-simulate each dev event (USGS epicentre, depth, Mw and moment tensor) at the same
stations through the full simulator, in real quiet-time noise from the same station, read it exactly like the real
records, and score QuakeNet v2 (and the classical combination) on the twins next to the real records.
If the twin reads right and the real record doesn't, the gap is in the simulator. Writes data/twin_dev_samples.npz
(dev_samples.npz with the dev nodes replaced by their twins) and prints a comparison."""
import sys, json, numpy as np, torch
from _paths import H; sys.path.insert(0, H + 'src'); sys.path.insert(0, H + 'src/v12')
import sim2 as S2, quakesim as QS, tt as TT, train2 as T2, prep as PR

def build(seed=0):
    torch.set_num_threads(1); rows = json.load(open(H + 'data/real_nodes.json')); MT = json.load(open(H + 'data/usgs_mt_dev.json')); SP = json.load(open(H + 'splits.json'))
    D = T2.load(H + 'data/dev_samples.npz'); enc = S2.Encoder(); NP, by = S2.noise_pool(); rng = np.random.default_rng(seed); oc = {}
    W, LS, PK, PZ, TPK, TSP, which = [], [], [], [], [], [], []
    for i, r in enumerate(rows):
        if r['split'] != 'dev': continue
        e = SP['events'][r['id']]; z = int(min(QS.DEPTHS, key=lambda v: abs(v - e['depth']))); M = np.array(MT[r['id']]['M']); m0 = np.sqrt(0.5 * np.sum(M[:3] ** 2) + np.sum(M[3:] ** 2))
        if r['id'] not in oc: oc[r['id']] = S2.ocean_params(rng, z)
        d = float(TT.dist(e['lat'], e['lon'], r['slat'], r['slon'])); az = float(TT.azimuth(e['lat'], e['lon'], r['slat'], r['slon'])); baz = float(TT.azimuth(r['slat'], r['slon'], e['lat'], e['lon']))
        pool = by.get(r['station']); nz = NP[rng.choice(pool)] if pool else NP[rng.integers(len(NP))]
        w, ls, p, peak, pf, spk, _ = S2.station_record(rng, (e['lat'], e['lon'], z, e['mw']), M, QS.half_duration(m0), oc[r['id']], d, az, baz, 0.0, nz)
        tP = float(TT.tt('P', d, z)); W.append(w); LS.append(ls); PK.append(peak); PZ.append(w[0, :240].astype(np.float16)); TPK.append(tP - 300 + p); TSP.append((tP - 300 + spk) if spk >= 0 else np.nan); which.append(i)
    E, O = enc(np.stack(W), LS, PK); mu, sd = PR.enc_norm(); R = S2.decode(O, mu, sd); which = np.array(which)
    D['n_emb'][which] = E.astype(np.float16); D['n_pz'][which] = np.stack(PZ); D['n_ls'][which] = LS; D['n_peak'][which] = PK; D['n_tpick'][which] = TPK; D['n_tspick'][which] = TSP
    for k, v in R.items(): D['n_r_' + k][which] = v
    import locate2 as LC                                                 # anchors recomputed from the twins' v1.2 readings
    for s in range(len(D['K'])):
        K = int(D['K'][s]); idx = D['idx'][s, :K]; A = LC.combine_v12(D['n_slat'][idx], D['n_slon'][idx], PR.Rof({k[2:]: v for k, v in D.items() if k.startswith('n_')}, idx))
        D['anchor'][s] = [A[k] for k in ('lat', 'lon', 'logz', 'mw', 'logz_sig', 'mw_sig')]; D['anchor_cov'][s] = A['cov']
    np.savez(H + 'data/twin_dev_samples.npz', **D); print('twins', len(which))

if __name__ == '__main__':
    build()
