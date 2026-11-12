"""Real recordings: catalogue, file-level dev/exam split (fixed seed), and 'readings' (one channel of one recording read
with one bearing's geometry). Nothing in the exam is used for any choice."""
import _paths as _P  # portable paths (see _paths.py)
import json, os, re, subprocess, numpy as np
import features as FE
D = _P.WORK + '/data/'; SEED = 20261106
INCH = 25.4
G = {'DE': dict(n=9, d=0.3126 * INCH, D=1.537 * INCH, phi=0.0), 'FE': dict(n=8, d=0.2656 * INCH, D=1.122 * INCH, phi=0.0)}
CWRU_NOM_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}; CWRU_FNOM = 1750 / 60     # 'about 1750 rpm' is all the network is told
CWRU_FS = {'normal': 48000, 'de12': 12000, 'de48': 48000, 'fe12': 12000}        # the normal baseline files are 48 kHz (checked: 360 Hz line)

def pb_docs():
    out = {}
    for f in sorted(os.listdir(D + 'pb/docs')):
        if not f.endswith('.pdf') or f.startswith('measuring'): continue
        t = subprocess.run(['pdftotext', '-layout', D + 'pb/docs/' + f, '-'], capture_output=True, text=True).stdout
        b = f[:-4]; g = lambda k: re.search(k + r'\s+(?:mm|-|pc\.)?\s+(\S.*?)\s*$', t, re.M)
        rec = dict(pitch=float(g('Pitch circle diameter').group(1)), maker=g('Manufacturer').group(1).strip())
        comp = re.search(r'Component\s+(.*)$', t, re.M); ln = re.search(r'Length\s+mm\s+(.*)$', t, re.M); mode = re.search(r'Mode\s+(.*)$', t, re.M)
        meth = re.search(r'Damage method\s+(.*)$', t, re.M); ext = re.search(r'Extent of damage\s+(.*)$', t, re.M)
        rec.update(component=comp.group(1).split() if comp else [], length=[float(v) for v in re.findall(r'[\d.]+', ln.group(1))] if ln else [],
                   mode=mode.group(1).split() if mode else [], method=meth.group(1).strip() if meth else None, extent=ext.group(1).split() if ext else [])
        out[b] = rec
    return out

def split():
    """file-level split. CWRU normal: 1 noise (training background), 1 dev, 2 exam. CWRU faults: 1 dev file per
    (set, class), the rest exam. Paderborn: K001-K003 background; dev = 1 healthy, 1 outer, 1 inner (artificial); rest exam."""
    rng = np.random.default_rng(SEED); cat = json.load(open(D + 'cwru_catalogue.json'))
    norm = sorted(c['id'] for c in cat if c['set'] == 'normal'); p = list(rng.permutation(norm))
    S = dict(cwru_noise=[int(p[0])], cwru_dev=[int(p[1])], cwru_exam=[int(v) for v in p[2:]])
    for s in ('de12', 'de48', 'fe12'):
        for c in ('OR', 'IR', 'B'):
            ids = sorted(x['id'] for x in cat if x['set'] == s and x['cls'] == c); k = int(rng.choice(ids)); S['cwru_dev'].append(k)
            S['cwru_exam'] += [i for i in ids if i != k]
    pbh = ['K004', 'K005', 'K006']; h = str(rng.choice(pbh)); ka = ['KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09']; ki = ['KI01', 'KI03', 'KI05', 'KI07', 'KI08']
    a = str(rng.choice(ka)); i = str(rng.choice(ki))
    allpb = sorted(f[:-4] for f in os.listdir(D + 'pb') if f.endswith('.npz'))
    S.update(pb_noise=['K001', 'K002', 'K003'], pb_dev=[h, a, i], pb_exam=[b for b in allpb if b not in ('K001', 'K002', 'K003', h, a, i)])
    json.dump(S, open(D + 'splits.json', 'w'), indent=1); return S

def splits(): return json.load(open(D + 'splits.json'))

def cwru_rec(i):
    cat = {c['id']: c for c in json.load(open(D + 'cwru_catalogue.json'))}; c = cat[i]; z = np.load(D + 'cwru/%d.npz' % i)
    fs = CWRU_FS[c['set']]; ch = {k: FE.to_fs(z[k], fs) for k in ('DE', 'FE', 'BA') if k in z.files}
    rpm = c['rpm_logged'] if c['rpm_logged'] == c['rpm_logged'] else CWRU_NOM_RPM[c['load']]
    if i == 99: rpm = CWRU_NOM_RPM[2]
    return c, ch, rpm / 60.0

def cwru_readings(ids):
    """near sensor (the faulty bearing's own end), far sensor (the other end), base plate; plus 'other bearing' reads
    (the healthy bearing at the other end of a faulty machine) labelled N with kind='other'"""
    R = []
    for i in ids:
        c, ch, fr = cwru_rec(i)
        base = dict(src='CWRU', rec=str(i), label_txt=c['label'], set=c['set'], load=c['load'], size_mm=c['size'] * INCH, pos=c.get('pos'), f_true=fr, f_nom=CWRU_FNOM)
        if c['cls'] == 'N':
            for s in ('DE', 'FE'):
                for b in ('DE', 'FE'): R.append(dict(base, cls='N', chan=s, bearing=b, loc='near' if s == b else 'far', kind='main', x=ch[s]))
            continue
        fe = c['fault_end']; other = 'FE' if fe == 'DE' else 'DE'
        for s in ch: R.append(dict(base, cls=c['cls'], chan=s, bearing=fe, loc='near' if s == fe else ('base' if s == 'BA' else 'far'), kind='main', x=ch[s]))
        if other in ch: R.append(dict(base, cls='N', chan=other, bearing=other, loc='near', kind='other', size_mm=0.0, x=ch[other]))
    for r in R: r['geom'] = G[r['bearing']]
    return R

def pb_readings(names, per_cond=None):
    docs = json.load(open(D + 'pb_docs.json')); R = []
    for b in names:
        z = np.load(D + 'pb/%s.npz' % b); meta = json.loads(str(z['meta'])); d = docs[b]
        cls = {'K0': 'N', 'KA': 'OR', 'KI': 'IR', 'KB': 'OR+IR'}[b[:2]]      # Paderborn naming; KI04/KI14 carry a minor OR mark (extent n/a)
        geom = dict(n=8, d=6.75, D=d['pitch'], phi=0.0); cnt = {}
        for x, m in zip(z['x'], meta):
            cnt[m['cond']] = cnt.get(m['cond'], 0) + 1
            if per_cond and cnt[m['cond']] > per_cond: continue
            R.append(dict(src='PB', rec=b + '/' + m['cond'] + '/%d' % m['rep'], bearing_id=b, cls=cls, chan='housing', bearing='PB6203', loc='near', kind='main',
                          cond=m['cond'], size_mm=float(max(d['length'])) if cls != 'N' and d['length'] else 0.0, mode=' '.join(d['mode']), method=d['method'],
                          f_true=m['rpm'] / 60, f_nom=float(m['cond'][1:3]) * 100 / 60, geom=geom, x=x.astype(float)))
    return R

if __name__ == '__main__':
    json.dump(pb_docs(), open(D + 'pb_docs.json', 'w'), indent=1); print(json.dumps(split()))
