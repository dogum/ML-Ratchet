"""Round 2: write ratchet.json and card.md of a released round-2 arm from round2/decisions.json.

python finalize.py <arm_id>
decisions.json[arm_id] = {label, family, family_label, tracks, folder, rungs: [{id, change, decision, why, simval_skill,
simval_MdAPE, dev_D16, ...}], released, compute: {...}, feature_engineering, extra_files, card: {what, training,
limits}}
"""
import json
import os
import sys

R2 = os.path.dirname(os.path.abspath(__file__))
RULE = ('keep a rung if its full frozen sim val skill beats the incumbent by > 0.01 AND its dev D16 is not worse than the '
        "incumbent's by more than 0.01 (the bake-off's rule; PREREG.md section 0). Physics rungs: PREREG.md section 2.")


def main(arm_id):
    D = json.load(open(os.path.join(R2, 'decisions.json')))[arm_id]
    d = os.path.join(R2, 'arms', D['folder'])
    meta = {'arm_id': arm_id, 'label': D['label'], 'family': D['family'], 'family_label': D['family_label'],
            'group': 'round2', 'tracks': D['tracks'], 'numpy_only': True, 'numpy_only_reason': '',
            'real_labels_in_training': False, 'compute': D['compute'], 'feature_engineering': D['feature_engineering'],
            'extra_files': D.get('extra_files', []), 'margin': 0.01, 'rule': RULE, 'rungs': D['rungs'],
            'released': D['released'], 'prereg': 'round2/PREREG.md'}
    json.dump(meta, open(os.path.join(d, 'ratchet.json'), 'w'), indent=1)
    lines = [f"# {arm_id}: {D['label']}", '', f"*Vehicle Bake-off round 2 (24-25 Nov 2026). Pre-registration: "
             f"`round2/PREREG.md`. Track: {', '.join(D['tracks'])}.*", '', '**What.** ' + D['card']['what'], '',
             '**Training.** ' + D['card']['training'], '', '**Rungs** (frozen sim val: 3,985 trips, frozen tiers; dev D16 = '
             'mean of the 16 dev MdAPEs):', '', '| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | Decision | Why |',
             '|---|---|---|---|---|---|---|']
    for r in D['rungs']:
        f = lambda k: (f"{r[k]:.4f}" if isinstance(r.get(k), (int, float)) else '')  # noqa: E731
        lines.append(f"| {r['id']} | {r['change']} | {f('simval_skill')} | {f('simval_MdAPE')} | {f('dev_D16')} | "
                     f"{r['decision']} | {r['why']} |")
    lines += ['', f"**Compute.** {D['compute'].get('notes', '')}", '', '**Inputs and features.** ' + D['feature_engineering'],
              '', '**Limits.** ' + D['card']['limits'], '']
    open(os.path.join(d, 'card.md'), 'w').write('\n'.join(lines))
    print('wrote', d)


if __name__ == '__main__':
    main(sys.argv[1])
