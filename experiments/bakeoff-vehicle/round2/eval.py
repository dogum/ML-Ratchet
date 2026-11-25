"""Round 2: score an arm folder on the frozen sim val and the dev gate with the harness's own evaluate.py (unchanged).

python eval.py <arm_dir> [--track trace] [--tag r0] [--val val|val_v4] [--sets simval,dev] [--out DIR]
python eval.py --refs            dev reference rows (E_phys, MOVES-like, ...) -> work/harness_results/dev/
python eval.py --build-dev       real dev records (harness/build_real.py) -> work/real/dev/

Outputs go to <out> (default round2/work/eval/<arm folder name>/). `--val val_v4` scores on the P0 physics's own version
of the frozen val (PREREG section 2). EnergyNet v2.1's dev predictions (the paired-bootstrap anchor) are expected in
work/harness_results/dev/enet_v21_table_dev_preds.parquet (made by: eval.py ../harness/arms_ref/enet_v21 --anchor).
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths as P  # noqa: E402
P.setup()
import config as C  # noqa: E402
import evaluate as EV  # noqa: E402

if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == '--build-dev':
        import build_real
        for w in (a[1:] or ['ved', 'epa', 'hd']):
            {'ved': build_real.ved, 'epa': build_real.epa, 'hd': build_real.hd}[w]('dev')
        sys.exit(0)
    if a and a[0] == '--refs':
        EV.build_dev_refs(); sys.exit(0)

    def opt(k, d=None):
        return a[a.index(k) + 1] if k in a else d

    arm = os.path.abspath(a[0])
    val = opt('--val', 'val')
    P.use_bank(val, 'val')
    if val != 'val':
        # skill denominators stay train101's dumb rates (same as v1.0); the val rows come from the named bank
        pass
    out = opt('--out', os.path.join(P.WORK, 'eval', os.path.basename(arm.rstrip('/\\')) + ('' if val == 'val' else '__' + val)))
    s = EV.evaluate(arm, opt('--track'), tuple(opt('--sets', 'simval,dev').split(',')), None, opt('--tag'), out,
                    '--no-compare' not in a)
    if '--anchor' in a:
        tr = s['track']
        shutil.copy(os.path.join(out, f"{s['tag']}_{tr}_dev_preds.parquet"),
                    os.path.join(C.RESULTS, 'dev', f"enet_v21_{tr}_dev_preds.parquet"))
