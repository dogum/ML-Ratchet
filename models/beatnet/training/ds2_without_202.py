"""Dated DS2 score of the released v1.0 without record 202 (27 Nov 2026; audit queue). No v2 rung was promoted, so the
released weights are scored once on DS2 with 202 left out (split v2), with v1.0's own dev prior (lam 0.25, the logprior
stored in the weights), next to all 22 records (which must reproduce the published 0.547). Chooses nothing.
Writes work/ds2_without_202.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import bnet as BN, baselines as BL, common as C, beatnet as R

if __name__ == '__main__':
    rel = R.BeatNet(); lp = np.asarray(rel.meta['logprior']); Z = BN.real_bank('mitdb'); out = dict(run_utc=time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), logprior=lp.tolist())
    for split in ('ds2', 'ds2all'):
        _, idx = BN.split_idx(split); y = Z['y'][idx]; import train as TR
        x, r = TR.inputs(Z, idx, 1); o = np.concatenate([rel.forward(x[k:k + 2048], r[k:k + 2048]) for k in range(0, len(idx), 2048)])
        p = BN.probs(o, lp); out[split] = dict(records=sorted(set(int(v) for v in Z['rec'][idx])), beatnet=BN.scores(y, p.argmax(1)), beatnet_raw=BN.scores(y, o[:, :4].argmax(1)),
                                                rules=BN.scores(y, BL.rules(Z, idx)), majority=BN.scores(y, np.zeros(len(y), int)), counts=np.bincount(y, minlength=5).tolist())
        s = out[split]['beatnet']; print(split, len(out[split]['records']), 'records; macro-F1', round(s['macro_f1'], 4), 'SV-F1', round(s['sv_f1'], 4), 'Se', np.round(s['se'], 3), '+P', np.round(s['ppv'], 3), '| rules', round(out[split]['rules']['macro_f1'], 4))
    json.dump(out, open(_P.WORK + '/ds2_without_202.json', 'w'), indent=1)
