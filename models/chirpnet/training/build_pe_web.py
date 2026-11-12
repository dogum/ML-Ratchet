"""Posteriors for the page: each event's window is placed using ChirpNet's own scan time (not LIGO's), then the
portable numpy flow is sampled. Also P-P calibration curves from held-out injections."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release')
import chirpnet, gwphys as G

# GWTC-1 Table III (source frame, median and 90% interval) where we have them
REF = {'GW150914': dict(m1=35.6, m2=30.6, mc=[28.6, 1.7, 1.5], chi=[-0.01, 0.12, 0.13], dl=440),
       'LVT151012': dict(m1=23.2, m2=13.6, mc=[15.2, 2.1, 1.2], chi=[0.05, 0.31, 0.20], dl=1080),
       'GW151226': dict(m1=13.7, m2=7.7, mc=[8.9, 0.3, 0.3], chi=[0.18, 0.20, 0.12], dl=450),
       'GW170104': dict(m1=30.8, m2=20.0, mc=[21.4, 2.2, 1.8], chi=[-0.04, 0.17, 0.21], dl=990)}
KEYS = ['m1_source', 'm2_source', 'chirp_mass_source', 'chi_eff', 'luminosity_distance_mpc', 'cos_inclination', 'mass_ratio']

def f16(a): return base64.b64encode(np.asarray(a, '<f2').tobytes()).decode()

def main(out=_P.LAB + '/pe.json', keep=1500):
    pe = chirpnet.ChirpPE(_P.WORK + '/release/chirpnet_pe_weights.npz')
    hits = {e['id']: e['hit'] for e in json.load(open(_P.LAB + '/data.json'))['events']}
    ev = {}
    for name, info in G.events().items():
        h1, t0 = chirpnet.load_gwosc(f"{_P.WORK}/data_raw/{info['fn_H1']}"); l1, _ = chirpnet.load_gwosc(f"{_P.WORK}/data_raw/{info['fn_L1']}")
        r = pe.weigh(h1, l1, t0, hits[name]['gps'], n=20000, seed=1)
        S = r['samples']; idx = np.random.default_rng(0).choice(20000, keep, replace=False)
        ev[name] = {'summary': r['summary_90'], 'samples': {k: f16(S[k][idx]) for k in KEYS}, 'n': keep, 'ref': REF[name],
                    'placed_on': 'chirpnet', 't_hit': hits[name]['gps'], 't_ligo': info['tevent']}
        s = r['summary_90']; print(name, {k: [round(v, 2) for v in s[k]] for k in ('m1_source', 'm2_source', 'chirp_mass_source', 'chi_eff', 'luminosity_distance_mpc')}, flush=True)
    from eval_pe import load, pp
    p = pp(load(_P.WORK + '/ckpt/pe1.pt'), n=400, ns=1000)
    print('pp', p['maxdev'], flush=True)
    json.dump({'events': ev, 'pp': p, 'params': ['log Mc', 'q', 'chi_eff', 'log10 dL', 'cos iota', 't_c']}, open(out, 'w'))

if __name__ == '__main__':
    main()
