"""Data for the BearingNet page and the release demo examples.
 - weights.json: float32 weights of release/bearingnet_weights.npz (BatchNorm folded), meta incl. dev-fitted calibration
 - examples: real EXAM readings drawn at random within (source, class) strata with a fixed seed (never picked by result)
 - labBg: healthy NOISE recordings (CWRU 98, Paderborn K001) for the in-browser lab (these are training backgrounds, not exam)
 - exam / ratchet tables from exam_<ckpt>.json and ratchet.json; prose from web/copy.json (hand-written after the exam)
Usage: webdata.py r3   -> web/data_auto.json, web/weights.json, release/examples/*.npz"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, base64, glob, numpy as np
sys.path.insert(0, _P.SRC); import realdata as RD, features as FE
B = _P.WORK + '/'; OUT = B + 'web/'; CK = sys.argv[1]
NAME = dict(N='Healthy', OR='Outer race', IR='Inner race', B='Ball')

def b64i16(x):
    q = float(np.abs(x).max()) or 1.0; return base64.b64encode(np.round(x / q * 32767).astype(np.int16).tobytes()).decode(), q

# ---- weights
W = np.load(B + 'release/bearingnet_weights.npz'); meta = json.loads(str(W['meta'])); man, buf, off = [], [], 0
for k in W.files:
    if k == 'meta': continue
    a = W[k].astype(np.float32).ravel(); man.append(dict(k=k, shape=list(W[k].shape), off=off)); buf.append(a); off += a.size
json.dump(dict(meta=meta, man=man, b64=base64.b64encode(np.concatenate(buf).tobytes()).decode()), open(OUT + 'weights.json', 'w'))

# ---- examples: fixed random draw within strata of the exam readings
E = json.load(open(B + 'exam_%s.json' % CK)); R = E['readings']; rng = np.random.default_rng(2026)
strata = [('CWRU', 'N', 2), ('CWRU', 'OR', 2), ('CWRU', 'IR', 2), ('CWRU', 'B', 2), ('PB', 'N', 2), ('PB', 'OR', 2), ('PB', 'IR', 2)]
pick = []
for src, c, n in strata:
    idx = [i for i, r in enumerate(R) if r['src'] == src and r['cls'] == c and r['kind'] == 'main' and r['loc'] == 'near']
    pick += sorted(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
ex, rel = [], []
for os_ in glob.glob(B + 'release/examples/*.npz'): os.remove(os_)
for j in pick:
    r = R[j]
    if r.get('f_nom') is None: r['f_nom'] = RD.CWRU_FNOM if r['src'] == 'CWRU' else float(r['cond'][1:3]) * 100 / 60
    if r['src'] == 'CWRU':
        c, ch, fr = RD.cwru_rec(int(r['rec'])); x = FE.pad_short(ch[r['chan']]); machine = 'CWRU test rig, %s bearing, %d hp load' % ({'DE': 'drive-end SKF 6205', 'FE': 'fan-end SKF 6203'}[r['bearing']], r['load'])
        geom = RD.G[r['bearing']]; name = 'CWRU %s %s' % (r['rec'], r['label_txt']); short = 'CWRU %s · %s' % (r['rec'], r['cls'])
        size_txt = ('%.3f in (%.2f mm) %s' % (r['size_mm'] / 25.4, r['size_mm'], 'spot') if r['cls'] != 'N' else '')
    else:
        b, cond, rep = r['rec'].split('/'); z = np.load(RD.D + 'pb/%s.npz' % b); mm = json.loads(str(z['meta']))
        k = next(i for i, m in enumerate(mm) if m['cond'] == cond and str(m['rep']) == rep); x = z['x'][k].astype(float)
        geom = dict(n=8, d=6.75, D=29.05, phi=0.0); machine = 'Paderborn test rig, 6203 bearing, %s' % cond; name = 'Paderborn %s %s' % (b, cond); short = 'PB %s · %s' % (b, r['cls'])
        size_txt = ('%s damage, up to %.1f mm long' % (r['mode'] or '', r['size_mm']) if r['cls'] != 'N' else '')
    w = x[:FE.NWIN]; s, q = b64i16(w)
    ex.append(dict(short=short, title=name, machine=machine, cls=r['cls'], chan=r['chan'] if r['src'] == 'CWRU' else 'housing', f_true=r['f_true'], f_nom=r['f_nom'], geom=geom,
                   size_txt=size_txt, py=dict(pred=r['pred'], p=r['p']), cl_pred=r['pred_cl'], q=q, x=s))
    seg = x[:int(5.5 * FE.FS)].astype(np.float32)
    np.savez_compressed(B + 'release/examples/%02d_%s.npz' % (len(ex), short.replace(' · ', '_').replace(' ', '_')), x=seg, fs=np.float32(FE.FS),
                        meta=json.dumps(dict(name=name, truth=r['cls'], bearing=geom, rpm_nominal=r['f_nom'] * 60, rpm_true=r['f_true'] * 60, size_mm=r['size_mm'] if r['cls'] != 'N' else 0)))

# ---- lab backgrounds (training noise files, not exam)
labBg = []
c, ch, fr = RD.cwru_rec(98); x = ch['DE'][:int(FE.NWIN * 1.32) + 8]; s, q = b64i16(x)
labBg.append(dict(name='CWRU rig, drive end, healthy (1772 rpm)', f=fr, geom=RD.G['DE'], x=s, q=q))
z = np.load(RD.D + 'pb/K001.npz'); mm = json.loads(str(z['meta'])); k = next(i for i, m in enumerate(mm) if m['cond'] == 'N15_M07_F10')
x = z['x'][k][:int(FE.NWIN * 1.32) + 8].astype(float); s, q = b64i16(x)
labBg.append(dict(name='Paderborn rig, healthy K001 (1500 rpm)', f=mm[k]['rpm'] / 60, geom=dict(n=8, d=6.75, D=29.05, phi=0.0), x=s, q=q))

# ---- tables
K = json.load(open(B + 'checks/kinematics_check.json'))['SKF6205']; RT = json.load(open(B + 'ratchet.json')); CP = json.load(open(OUT + 'copy.json'))
f3 = lambda v: '%.2f' % v
rows = []
for key, lab in (('cwru_near', 'CWRU · sensor at the faulty bearing'), ('cwru_far', 'CWRU · sensor at the other end'), ('cwru_base', 'CWRU · base plate'), ('pb', 'Paderborn · new machine')):
    b = E[key]; rows.append(dict(c=[lab, b['n'], f3(b['net']['bal_acc']), f3(b['classical']['bal_acc']), f3(b['majority']['bal_acc']), f3(b['net']['acc']), f3(b['classical']['acc'])]))
table = dict(head=['Exam set', 'Readings', 'Network', 'Envelope', 'Majority', 'Net acc.', 'Env. acc.'], rows=rows)
confs = [dict(title='Network · CWRU near sensor', m=E['cwru_near']['net']['conf']), dict(title='Envelope method · CWRU near sensor', m=E['cwru_near']['classical']['conf']),
         dict(title='Network · Paderborn', m=E['pb']['net']['conf']), dict(title='Envelope method · Paderborn', m=E['pb']['classical']['conf'])]
byLoad = dict(head=['CWRU near sensor', 'Readings', 'Net accuracy'], rows=[dict(c=['%s hp load' % k, v['n'], f3(v['acc'])]) for k, v in E['by_load'].items()] +
              [dict(c=['sensor: %s' % k, v['n'], f3(v['acc'])]) for k, v in E['by_loc_net'].items()])
bySize = dict(head=['Defect', 'Readings', 'Net accuracy'], rows=[dict(c=[k, v['n'], f3(v['acc'])]) for k, v in E['by_size'].items()] +
              [dict(c=['PB %s' % k, v['n'], f3(v['acc'])]) for k, v in E['pb_by_kind'].items()])
SHORT = {'r1.pt': 'v1 physics', 'r2.pt': 'v2 physics, calibrated on dev', 'r3.pt': 'Shared trunk per fault hypothesis', 'r4.pt': 'r2 + band dropout'}
rrows = []
for r in RT:
    dv = r['dev']['all']['bal_acc']; why = 'first rung' if 'vs' not in r else ('kept' if r['accepted'] else ('sim gain under 0.01' + ('' if dv >= 0 else '')))
    rrows.append(dict(c=[r['ckpt'].replace('.pt', ''), SHORT.get(r['ckpt'], r['note']), f3(r['objective']), f3(r['val']['bal_acc']), f3(dv), ('kept' if r['accepted'] else 'rejected') + ('' if 'vs' not in r else ' (' + r['why'].replace('sim objective ', 'sim ').replace('dev bal. acc ', 'dev ') + ')')], best=r['accepted']))
ratchet = dict(head=['Rung', 'Change', 'Sim objective', 'Sim bal. acc', 'Dev bal. acc', 'Decision'], rows=rrows)
D = dict(headline=CP['headline'], kin={k: dict(ours=v['ours'], published=v['published']) for k, v in K.items()}, kinNote=CP['kinNote'], geomDE=RD.G['DE'],
         examples=ex, labBg=labBg, exam=dict(text=CP['examText'], table=table, confs=confs, byLoad=byLoad, bySize=bySize, more=CP['examMore']),
         ratchet=ratchet, steps=CP['steps'], limits=CP['limits'], credits=CP['credits'])
json.dump(D, open(OUT + 'data_auto.json', 'w')); print('examples', len(ex), 'data_auto.json', os.path.getsize(OUT + 'data_auto.json'), 'weights.json', os.path.getsize(OUT + 'weights.json'))
