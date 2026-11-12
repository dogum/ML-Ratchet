"""Independent checks of every piece of the BeatNet simulator. Writes physics_check.json.
 1. Eikonal solver: 26-neighbour Dijkstra vs exact straight-line travel time in a homogeneous block.
 2. Heart dipole: the binned 'sweep' implementation vs a direct evaluation of the volume integral -sum(grad Vm) dV, and
    vs the surface (equivalent double layer) form -sum(Vm n dA) over the tissue boundary (divergence theorem).
 3. Lead field: Dower lead vectors vs an infinite homogeneous conductor with electrodes at standard torso positions.
 4. ECGSYN (McSharry et al. 2003): our RR generator reproduces the requested LF/HF ratio; our physical normal beat vs
    the ECGSYN default beat in lead II (correlation, intervals).
 5. Physiology vs textbook ranges: QRS width (normal, LBBB, RBBB, PVC), frontal QRS axis, T/QRS and P/QRS ratios,
    compensatory pause after PVCs vs non-compensatory pause after atrial premature beats."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, scipy.signal as ss
sys.path.insert(0, _P.SRC); import heart as Hm, sim, strip as S
OUT = {}

# ---- 1. eikonal
n = 41; g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing='ij'), -1).reshape(-1, 3); idx = np.arange(n ** 3).reshape(n, n, n)
I, J, L = [], [], []
for o in Hm.OFFS:
    nb = g + o; ok = np.all((nb >= 0) & (nb < n), 1); I.append(np.nonzero(ok)[0]); J.append(idx[tuple(nb[ok].T)]); L.append(np.full(ok.sum(), Hm.HV * np.linalg.norm(o)))
import scipy.sparse as sp; from scipy.sparse.csgraph import dijkstra
I, J, L = map(np.concatenate, (I, J, L)); v = 0.5; A = sp.csr_matrix((L / v, (I, J)), shape=(n ** 3, n ** 3)); c = idx[20, 20, 20]
t = dijkstra(A, indices=c); ex = np.linalg.norm((g - 20) * Hm.HV, axis=1) / v; m = ex > 20
rel = (t[m] - ex[m]) / ex[m]; OUT['eikonal'] = dict(median_rel_err=float(np.median(rel)), max_rel_err=float(rel.max()), min_rel_err=float(rel.min()),
                                                   note='26-neighbour shortest paths overestimate oblique travel times; max error is a geometry bound (~8% at directions like (2,1,0))')

# ---- 2. dipole: sweep vs direct volume integral vs surface integral
p = sim.Patient(3, bbb='none'); h = p.h; tau = p.vmap('normal'); mk = np.isfinite(tau); wd = 1.5
f = lambda tt: 1 / (1 + np.exp(-(tt) / wd))           # Vm upstroke (unit amplitude), the QRS part only
Hs = Hm.HV ** 3 * Hm.sweep(tau[mk], h.grad(np.where(mk, tau, np.nan))[mk], 140, wd)
ts = np.arange(5, 130, 5); Hv, Hf = [], []
# boundary faces of the ventricular tissue: (voxel, outward normal) where the face neighbour is not ventricular tissue
faces = []
for a in range(3):
    for s_, j in ((1, h.nbr[2 * a]), (-1, h.nbr[2 * a + 1])):
        k = np.nonzero(mk & ((j < 0) | ~mk[np.maximum(j, 0)]))[0]; nrm = np.zeros(3); nrm[a] = s_; faces.append((k, nrm))
for tt in ts:
    phi = np.where(mk, f(tt - tau), np.nan); gv = h.grad(phi); Hv.append(-Hm.HV ** 3 * gv[mk].sum(0))
    Hf.append(-sum(Hm.HV ** 2 * phi[k].sum() * nrm for k, nrm in faces))
Hv, Hf = np.array(Hv), np.array(Hf); Hs_t = -Hs[ts]       # sweep returns +sum(f' grad tau) = -sum(grad Vm)
cc = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1]); rr_ = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
OUT['dipole'] = dict(sweep_vs_volume_corr=cc(-Hs_t, Hv), sweep_vs_volume_relrms=rr_(-Hs_t, Hv), volume_vs_surface_corr=cc(Hv, Hf), volume_vs_surface_relrms=rr_(Hv, Hf),
                     note='volume form -sum(grad Vm) dV and surface form -sum(Vm n dA) agree up to the finite-difference treatment of the wall voxels')

# ---- 3. lead field: infinite homogeneous medium, electrodes on a torso (Frank axes: x left, y down, z back; mm, heart centre at origin)
E = {'RA': (-170, -230, 0), 'LA': (170, -230, 0), 'LL': (60, 420, 0), 'V1': (-35, -10, -105), 'V2': (25, -10, -110), 'V3': (65, 15, -105),
     'V4': (95, 40, -90), 'V5': (135, 40, -55), 'V6': (155, 40, 0)}
r0 = np.array([25.0, 0, -45]); lv = lambda e: (np.array(E[e]) - r0) / np.linalg.norm(np.array(E[e]) - r0) ** 3     # potential per unit dipole
wct = (lv('RA') + lv('LA') + lv('LL')) / 3; LF = {'I': lv('LA') - lv('RA'), 'II': lv('LL') - lv('RA'), **{k: lv(k) - wct for k in ('V1', 'V2', 'V3', 'V4', 'V5', 'V6')}}
ang = {k: float(np.degrees(np.arccos(np.dot(Hm.unit(LF[k]), Hm.unit(Hm.DOWER[k]))))) for k in LF}
OUT['lead_field'] = dict(angle_deg_dower_vs_infinite_medium=ang, median_angle=float(np.median(list(ang.values()))),
                         note='Dower vectors come from measurements on a torso model; the infinite medium ignores the torso boundary and lungs, so agreement is expected only to tens of degrees')

# ---- 4. ECGSYN
def ecgsyn_beat(fs=1000, hr=60.0):
    th = np.radians([-70, -15, 0, 15, 100]); a = np.array([1.2, -5, 30, -7.5, 0.75]); b = np.array([0.25, 0.1, 0.1, 0.1, 0.4]); w = 2 * np.pi * hr / 60
    def f(s):
        x, y, z = s; al = 1 - np.hypot(x, y); t = np.arctan2(y, x); d = (t - th + np.pi) % (2 * np.pi) - np.pi
        return np.array([al * x - w * y, al * y + w * x, -np.sum(a * d * np.exp(-d ** 2 / (2 * b ** 2))) - z])
    s = np.array([-1.0, 0, 0.04]); dt = 1 / fs; Z = []
    for _ in range(int(5 * fs * 60 / hr)):
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3); s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); Z.append(s[2])
    Z = np.array(Z)[-int(fs * 60 / hr):]; Z = (Z - Z.min()) / (Z.max() - Z.min()) * 1.6 - 0.4; return np.roll(Z, len(Z) // 2 - np.argmax(Z))
ref = ecgsyn_beat(); r_peak = np.argmax(ref)
rng = np.random.default_rng(0); lfhf = []
for target in (0.5, 1.0, 2.0):
    est = []
    for k in range(40):
        x = sim.rr_ecgsyn(512, 60, 0.05, target, rng); fq, P = ss.periodogram(x - x.mean(), fs=1.0)
        est.append(P[(fq > 0.04) & (fq < 0.15)].sum() / P[(fq > 0.15) & (fq < 0.4)].sum())
    lfhf.append(dict(target=target, measured_median=float(np.median(est))))
beats, qw, axes_, tq, pq = [], [], [], [], []
LII = np.array(Hm.DOWER['II'])
for sd in range(30):
    pp = sim.Patient(sd, bbb='none'); tn = pp.vmap('normal'); Hd = pp.vdipole(tn, 1.0, 'n'); a_ = pp.adip(pp.sinus); T = np.zeros((1000, 3)); on = 350
    T[on:on + len(Hd)] += Hd[:1000 - on]; T[on - int(pp.PR):on - int(pp.PR) + len(a_)] += a_[:1000 - on + int(pp.PR)]; v2 = T @ LII
    rp = on + int(np.argmax(np.abs(v2[on:on + 120]))); beats.append(np.roll(v2, 500 - rp) / np.ptp(v2[on:on + 150]))
    w = pp.qrs_width(tn); qw.append(w); qi = (T[on:on + int(w)] @ np.array(Hm.DOWER['I'])).sum(); qf = (T[on:on + int(w)] @ np.array(Hm.DOWER['aVF'])).sum(); axes_.append(np.degrees(np.arctan2(qf, qi)))
    tq.append(np.abs(v2[on + int(w) + 60:]).max() / np.ptp(v2[on:on + int(w) + 10])); pq.append(np.abs(a_ @ LII)[:150].max() / np.ptp(v2[on:on + int(w) + 10]))
mb = np.median(beats, 0); seg = slice(500 - 300, 500 + 500); rs = ref[r_peak - 300:r_peak + 500] if r_peak >= 300 else np.roll(ref, 300 - r_peak)[:800]
OUT['ecgsyn'] = dict(rr_lfhf=lfhf, beat_corr_leadII=cc(mb[seg], rs), ecgsyn_qt_ms=None,
                     sim_P_peak_to_R_ms=float(500 - (200 + np.argmax(mb[200:420]))), ecgsyn_P_peak_to_R_ms=float(np.radians(70) / (2 * np.pi) * 1000),
                     sim_T_peak_after_R_ms=float(np.argmax(mb[620:900]) + 120), ecgsyn_T_peak_after_R_ms=float(np.radians(100) / (2 * np.pi) * 1000),
                     note='ECGSYN is a phenomenological single-lead model (5 Gaussians on a limit cycle); BeatNet beats come from activation physics, so only the gross timing and shape should match')
# ---- 5. physiology
wl, wr_, wv = [], [], []
for sd in range(30):
    wl.append(sim.Patient(sd, bbb='LBBB').qrs_width(sim.Patient(sd, bbb='LBBB').vmap('normal')))
    pr = sim.Patient(sd, bbb='RBBB'); wr_.append(pr.qrs_width(pr.vmap('normal')))
    pv = sim.Patient(sd, bbb='none'); wv.append(pv.qrs_width(pv.vmap('pvc', pv.random_vfocus(), pv.rng.uniform(1.8, 4.0))))
pc = lambda x: [float(np.percentile(x, q)) for q in (10, 50, 90)]
# pauses: (RR before + RR after) / (2 x local sinus RR) around isolated ectopics in sinus rhythm
comp = {1: [], 2: []}; rng = np.random.default_rng(1)
for sd in range(40):
    pp = sim.Patient(1000 + sd, bbb='none'); P = S.program(rng); P.update(base='sinus', mode=['S', 'V'][sd % 2], pattern='iso', std=0.01, ps=0.15, pv=0.15, junc=False, interp=False)
    V, A = S.events(pp, P, rng, 40); t = np.array([e['t'] for e in V]); y = np.array([e['label'] for e in V])
    for i in range(2, len(V) - 2):
        if y[i] in (1, 2) and (y[[i - 2, i - 1, i + 1, i + 2]] == 0).all(): comp[int(y[i])].append((t[i + 1] - t[i - 1]) / (t[i - 1] - t[i - 2] + t[i + 2] - t[i + 1]))
OUT['physiology'] = dict(qrs_ms_normal_p10_50_90=pc(qw), qrs_ms_LBBB=pc(wl), qrs_ms_RBBB=pc(wr_), qrs_ms_PVC=pc(wv), frontal_axis_deg=pc(axes_), T_over_QRS=pc(tq), P_over_QRS=pc(pq),
                         pause_ratio_PVC_median=float(np.median(comp[2])), pause_ratio_APB_median=float(np.median(comp[1])),
                         textbook='normal QRS 70-110 ms (vector-onset/offset definition reads shorter than manual calipers); BBB >= 120 ms; PVC 120-200 ms; axis -30..+90 deg; T/R ~0.1-0.4; P/R ~0.05-0.2; full compensatory pause ratio 1.0 after PVC, < 1 after APB')
json.dump(OUT, open(_P.WORK + '/physics_check.json', 'w'), indent=1); print(json.dumps(OUT, indent=1))
