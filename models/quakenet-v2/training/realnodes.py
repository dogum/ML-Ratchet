"""Real records -> node features, exactly as sim2.py builds them for simulations: QuakeNet v1.2's pipeline (STA/LTA P
pick within 30 s of sample 300, i.e. of the ak135 P time predicted from the catalogue origin, as in v1.2), the classical
S pick, and the frozen v1.2 network's 256-number summary and 8 outputs.
Stations per event, in the order fixed by splits.json + data/station_order.json: QuakeNet's original records first,
then the newly fetched ones; at most 10. Absolute pick times are stored relative to the catalogue origin time only for
grading; the models see pick times relative to the event's earliest pick."""
import os, sys, json, numpy as np
from _paths import H; sys.path.insert(0, H + 'src'); sys.path.insert(0, H + 'src/v12')
import pipeline as PL, sim2 as S2, tt as TT, torch
from _paths import Q

def records(split_names=('dev', 'exam')):
    SP = json.load(open(H + 'splits.json')); order = json.load(open(H + 'data/station_order.json')); R = []
    for sp in split_names:
        for k in SP[sp]:
            e = SP['events'][k]; sts = order.get(k, e['stations']); n = 0
            for s in sts:
                f = (Q if s in e['stations'] else H + 'data/') + f'wf/ev_{k}_{s}.npz'
                if not os.path.exists(f): continue
                z = np.load(f); x = z['x'].astype(np.float64)
                if x.shape[1] < 3000 or not np.isfinite(x).all() or (x.std(1) == 0).any(): continue
                R.append((sp, k, s, n, f)); n += 1
                if n >= 10: break
    return R

if __name__ == '__main__':
    torch.set_num_threads(1); R = records(); enc = S2.Encoder(); W, LS, PK, PZ, rows = [], [], [], [], []
    ST = json.load(open(Q + 'stations.json'))
    for sp, k, s, n, f in R:
        z = np.load(f); x = z['x'].astype(np.float64); m = json.loads(str(z['meta'])); w, ls, p, peak = PL.read(x, 300); spk = S2.s_pick(x, p)
        t_start = m['p_time'] - 300 - m['event']['time']                 # record start, s after the catalogue origin
        W.append(w); LS.append(ls); PK.append(peak); PZ.append(w[0, :240].astype(np.float16))
        rows.append(dict(split=sp, id=k, station=s, order=n, slat=ST[s]['lat'], slon=ST[s]['lon'], dist=m['dist'], baz=m['baz'], az=m['az'],
                         tpick=t_start + p, tspick=(t_start + spk) if spk >= 0 else None, tP_cat=m['p_time'] - m['event']['time'], ls=float(ls), peak=float(peak)))
    E, O = enc(np.stack(W), LS, PK)
    np.savez(H + 'data/real_nodes.npz', emb=E.astype(np.float16), pz=np.stack(PZ), out8=O.astype(np.float32), ls=np.array(LS, np.float32), peak=np.array(PK, np.float32))
    json.dump(rows, open(H + 'data/real_nodes.json', 'w'))
    print(len(rows), 'records', len({r['id'] for r in rows}), 'events')
