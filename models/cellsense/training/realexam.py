"""CellSense final exam, health now: the 118 Severson cells not used for calibration, read every 25 cycles through life.
Inputs per reading: the 4C discharge curve Q(V), T(V) (constant-current part, as recorded), and, optionally, the same
cell's curve from cycles 5-10. Answer keys, all measured independently by the cycler:
  total capacity  discharge capacity including the 2.0 V hold (what the 80% end-of-life rule uses)
  pulse resistance 30 ms, +-3.6C pulses at 80% SOC during charge
  fade and resistance growth, relative to the same cell at cycles 5-10
Baseline for capacity: the end of the constant-current curve itself (what you would read off without any model)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M, realdata as RD, calibrate as CA
STEP = 4

def load_net(path):
    ck = torch.load(path, weights_only=False); lab = ck.get('labels', M.LABELS); ex = ck.get('extra', 0)
    net = M.CellSense(tuple(int(w) for w in ck['widths'].split(',')), n_out=len(lab), extra=ex); net.load_state_dict(ck['state']); net.eval()
    net.mu_y, net.sd_y, net.labels, net.extra = np.array(ck['mu_y']), np.array(ck['sd_y']), list(lab), ex; net.smooth = ck.get('smooth', 0); return net

def predict(net, Q1, T1, Q0=None, T0=None, S=None, bs=512):
    """S: (B, 2) cycle number and charging time 0-80% (min), for networks with a lifetime head"""
    import train as TR; out = []
    with torch.no_grad():
        for i in range(0, len(Q1), bs):
            X = torch.from_numpy(M.features(Q1[i:i + bs], T1[i:i + bs], None if Q0 is None else Q0[i:i + bs], None if T0 is None else T0[i:i + bs], smooth=getattr(net, 'smooth', 0)))
            Si = np.full((len(X), 2), np.nan) if S is None else S[i:i + bs]
            out.append(net(X, torch.from_numpy(TR.scalars(Si)) if getattr(net, 'extra', 0) else None).numpy())
    o = np.concatenate(out); lab = getattr(net, 'labels', M.LABELS); n = len(lab); mu = o[:, :n] * net.sd_y + net.mu_y; sg = np.exp(np.clip(o[:, n:], -7, 3)) * net.sd_y
    return {k: mu[:, j] for j, k in enumerate(lab)} | {k + '_sig': sg[:, j] for j, k in enumerate(lab)}

def readings(cells, every=25, only=None):
    """only=None: the exam cells (calibration cells left out); only=list: just those cells (the dev set)"""
    rows = []
    for k, c in cells.items():
        if (only is None and k in CA.CAL) or (only is not None and k not in only): continue
        S = c['summary']; qd = S['QDischarge']; ir = S['IR']; idx = c['cyc_idx']
        ref_rows = [i for i, j in enumerate(idx) if 4 <= j <= 9]
        q0 = np.nanmean(c['Qdlin'][ref_rows], 0)[::STEP]; t0 = np.nanmean(c['Tdlin'][ref_rows], 0)[::STEP]
        ok = lambda j: j < len(qd) and 0.5 < qd[j] < 1.2 and 0.005 < ir[j] < 0.05
        qd_ref = np.nanmean([qd[j] for j in idx[ref_rows] if ok(j)]); ir_ref = np.nanmean([ir[j] for j in idx[ref_rows] if ok(j)])
        for r, j in enumerate(idx):
            if j < 9 or (j % every and j != idx[-1]) or not ok(j): continue
            q = c['Qdlin'][r][::STEP]; t = c['Tdlin'][r][::STEP]
            if not (np.isfinite(q).all() and np.isfinite(t).all() and 0.5 < np.nanmax(q) < 1.25 and np.nanmin(q) > -0.01 and 15 < t.min() and t.max() < 60): continue   # corrupted records
            rows.append(dict(cell=k, batch=c['batch'], cycle=int(j + 1), life=c['cycle_life_pub'], Q1=q, T1=t, Q0=q0, T0=t0, q_total=float(qd[j]), q_cc=float(np.nanmax(c['Qdlin'][r])), ir=float(ir[j] * 1000),
                             fade=float(1 - qd[j] / qd_ref), r_growth=float(ir[j] / ir_ref - 1), policy=c['policy']))
    return rows

def score(rows, P, tag):
    g = lambda k: np.array([r[k] for r in rows]); qt, qc, ir, fd, rg = g('q_total'), g('q_cc'), g('ir'), g('fade'), g('r_growth')
    cor = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    S = {'n': len(rows), 'cells': len({r['cell'] for r in rows}),
         'q_total_mae_mAh': float(1000 * np.mean(np.abs(P['q_total'] - qt))), 'q_total_baseline_mae_mAh': float(1000 * np.mean(np.abs(qc - qt))),
         'q_total_bias_mAh': float(1000 * np.mean(P['q_total'] - qt)), 'q_total_corr': cor(P['q_total'], qt),
         'q_total_in_2sigma': float(np.mean(np.abs(P['q_total'] - qt) < 2 * P['q_total_sig'])),
         'ir_mae_mohm': float(np.mean(np.abs(P['ir'] - ir))), 'ir_bias_mohm': float(np.mean(P['ir'] - ir)), 'ir_corr': cor(P['ir'], ir),
         'fade_mae_pct': float(100 * np.mean(np.abs(P['fade'] - fd))), 'fade_corr': cor(P['fade'], fd),
         'r_growth_corr': cor(P['r_growth'], rg), 'r_growth_mae_pct': float(100 * np.mean(np.abs(P['r_growth'] - rg))),
         'eol_called_right': float(np.mean((P['q_total'] < 0.88) == (qt < 0.88)))}
    print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.items()}, flush=True); return S

def exam(net, rows=None):
    if rows is None: cells, V = RD.load(); rows = readings(cells)
    st = lambda k: np.stack([r[k] for r in rows])
    Pr = predict(net, st('Q1'), st('T1'), st('Q0'), st('T0')); Pn = predict(net, st('Q1'), st('T1'))
    return {'with_ref': score(rows, Pr, 'with reference'), 'no_ref': score(rows, Pn, 'curve only')}, rows, Pr, Pn

DEPTHS = (0.9, 0.75, 0.6, 0.45, 0.3)

def exam_partial(net, rows=None, depths=DEPTHS):
    """the same exam readings as partial discharges: each 4C curve cut once it has delivered D Ah from full (the rest of the
    curve missing), read with the cell's full early curve as reference, and with the reference cut at the same depth.
    Baseline (no model): the reference's total capacity plus the shift between the two curves at the deepest common point."""
    import train as TR
    if rows is None: cells, V = RD.load(); rows = readings(cells)
    st = lambda k: np.stack([r[k] for r in rows]); Q1, T1, Q0, T0 = st('Q1'), st('T1'), st('Q0'), st('T0'); qt = st('q_total'); out = {}
    q0_tot = np.array([r['q_total'] / (1 - r['fade']) for r in rows])          # reference total capacity (cycles 5-10)
    for D in depths:
        Dv = np.full(len(rows), D); q1, t1 = TR.truncate(Q1, T1, Dv); q0c, t0c = TR.truncate(Q0, T0, Dv); r = {}
        for tag, a, b in (('full_ref', Q0, T0), ('cut_ref', q0c, t0c)):
            P = predict(net, q1, t1, a, b); r[tag] = {'q_total_mae_mAh': float(1000 * np.mean(np.abs(P['q_total'] - qt))), 'fade_corr': float(np.corrcoef(P['fade'], st('fade'))[0, 1]),
                                                     'ir_mae_mohm': float(np.mean(np.abs(P['ir'] - st('ir')))), 'q_total_in_2sigma': float(np.mean(np.abs(P['q_total'] - qt) < 2 * P['q_total_sig']))}
        last = np.array([np.where(np.isfinite(x))[0].max() for x in q1]); dq = q1[np.arange(len(q1)), last] - Q0[np.arange(len(q1)), last]
        r['baseline_q_total_mae_mAh'] = float(1000 * np.mean(np.abs(q0_tot + dq - qt))); r['frac_cut'] = float(np.mean(np.isnan(q1).any(1)))
        out[str(D)] = r; print('depth', D, json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in r.items()}), flush=True)
    return out

if __name__ == '__main__':
    net = load_net(sys.argv[1]); res, rows, Pr, Pn = exam(net)
    json.dump(res, open(sys.argv[1].replace('.pt', '_real.json'), 'w'), indent=1)
