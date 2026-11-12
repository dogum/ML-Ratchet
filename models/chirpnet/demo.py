"""Run ChirpNet v2 on the LIGO events from the ChirpNet page.

    python demo.py                 # the four black-hole events (their 32 s files are in examples/)
    python demo.py --bns           # also GW170817, if you add its two 2048 s files to examples/:
        https://dcc.ligo.org/public/0146/P1700349/001/H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5
        https://dcc.ligo.org/public/0146/P1700349/001/L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5

Any other event: GWOSC 4 kHz HDF5 files for H1 and L1 (32 s is enough for black holes; neutron stars need minutes).
"""
import sys, glob, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import chirpnet

EVENTS = {'GW150914': '1126259446', 'GW151012': '1128678884', 'GW151226': '1135136334', 'GW170104': '1167559920'}
folder = next((a for a in sys.argv[1:] if not a.startswith('--')), os.path.join(HERE, 'examples'))
net = chirpnet.ChirpNet(os.path.join(HERE, 'chirpnet_weights.npz'))            # loads the coincidence checker too
pe = chirpnet.ChirpPE(os.path.join(HERE, 'chirpnet_pe_weights.npz'))
for name, gps in EVENTS.items():
    fh = glob.glob(os.path.join(folder, f'H-H1_*-{gps}-32.hdf5')); fl = glob.glob(os.path.join(folder, f'L-L1_*-{gps}-32.hdf5'))
    if not fh or not fl:
        print(f'{name}: files not found in {folder}'); continue
    h1, t0 = chirpnet.load_gwosc(fh[0]); l1, _ = chirpnet.load_gwosc(fl[0])
    hits = net.scan(h1, l1, gps_start=t0)
    if not hits:
        print(f'{name}: nothing above threshold'); continue
    h = hits[0]
    print(f"{name}: GPS {h['gps']:.3f}  statistic {h['score']:.1f}  coincidence {h['coincidence']:.1f}  FAR < {h['far_per_hour']:.2f}/h  "
          f"chirp mass {h['chirp_mass']:.1f} [{h['chirp_mass_68'][0]:.1f}, {h['chirp_mass_68'][1]:.1f}] (detector frame)")
    s = pe.weigh(h1, l1, t0, h['gps'])['summary_90']
    f = lambda k, d=1: f"{s[k][1]:.{d}f} [{s[k][0]:.{d}f}, {s[k][2]:.{d}f}]"
    print(f"   posterior (90%): m1 {f('m1_source')}  m2 {f('m2_source')}  chirp mass {f('chirp_mass_source')}  "
          f"chi_eff {f('chi_eff', 2)}  distance {f('luminosity_distance_mpc', 0)} Mpc  (source frame)")

if '--bns' in sys.argv:
    fh = glob.glob(os.path.join(folder, 'H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5')); fl = glob.glob(os.path.join(folder, 'L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5'))
    if not fh or not fl:
        print('GW170817: add the two 2048 s files listed at the top of this script'); sys.exit()
    bns = chirpnet.ChirpNetBNS(os.path.join(HERE, 'chirpnet_bns_weights.npz'))
    h1, t0 = chirpnet.load_gwosc(fh[0]); l1, _ = chirpnet.load_gwosc(fl[0])
    a = int((1187008882.43 - 60 - t0) * 4096)       # scan the two minutes around the event (the whole file takes longer)
    W = chirpnet.whiten_bns(h1), chirpnet.whiten_bns(l1)
    k0 = int((1187008882.43 - 60 - t0) * chirpnet.FSB); k1 = k0 + 120 * chirpnet.FSB
    hits = bns.scan(W[0][k0:k1], W[1][k0:k1], gps_start=t0 + k0 / chirpnet.FSB, whitened=True)
    for h in hits[:3]:
        print(f"GW170817 candidate: GPS {h['gps']:.2f}  statistic {h['score']:.1f}  FAR < {h['far_per_hour']:.2f}/h  "
              f"chirp mass {h['chirp_mass']:.4f} [{h['chirp_mass_68'][0]:.4f}, {h['chirp_mass_68'][1]:.4f}] (LIGO: 1.1975)")
