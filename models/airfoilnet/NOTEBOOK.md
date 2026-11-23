# AirfoilNet lab notebook

Built 22 Nov 2026 on the shared 2-core B-batch machine, alongside three other builds (load average 7-9 most of
the day). Plain record of what was tried, what failed and the numbers.

## Solver

- XFOIL itself could not be built (no gfortran). websites.umich.edu (mfoil's home) answered 403 and github.com
  answered 403 from this machine. mfoil.py v2023-06-28 (K. Fidkowski, MIT) came from PyPI as the `pymfoil` 0.1.0
  package, which ships it verbatim. The source was read before running: numpy/scipy/matplotlib only, no file or
  network access.
- One bug fixed in the vendored copy: `build_glob_sys` compared a sparse matrix to a shape tuple
  (`M.glob.R_x == (3*Nsys, Nsys)`), which raises on scipy 1.18; now `R_x.shape == ...`.
- `mfoil_fast.py` (ours) replaces the hot loops with numba kernels doing the same arithmetic in the same order:
  panel influence coefficients, the boundary-layer closures and station residual (Mach 0 only), and COO
  assembly of the Newton system. Whole solves match the unmodified mfoil to 1e-13 in Cl, Cd and Cm
  (`training/verify_fast.py`). Speed: about 10 s -> 0.5 s per point unloaded.
- Further fixes, each a robustness fix, not a physics change:
  - Sharp (closed) trailing edges made `TE_info` divide 0/0 and the wake start backwards. The zero-length TE panel
    now carries nothing (as in XFOIL) and the wake leaves along the TE bisector.
  - `set_coords` had an `X.shape(1)` typo; fixed.
  - Newton limit cycles: mfoil often ends in a 2- or 4-cycle when transition sits in the last panel or flips
    between two nodes; forces then move by ~1e-4 per step. A point is accepted as "loosely converged" if the
    last 10 Newton steps keep Cl within 0.01, Cd within 5% and Cm within 0.005 (XFOIL's own tolerance is of
    this order); the median of the window is reported and the point is flagged.
  - rtol 1e-10 -> 1e-7, and a vectorised force calculation inside the Newton loop (the full one still runs at
    the end): sweeps went from ~29 s to ~17 s.
- Warm starts from a non-converged state blow up (residual 1.6e4 at the first step, in the original too); the
  sweep only warm-starts from the last converged state, then retries cold, then a half step.
- Robustness is the main limit. Strict convergence on 16 random UIUC airfoils at alpha 2, Re 5e5 was 3/16 before
  the fixes, 8/16 after. SD7034 at Re 1e5 and 2e5 gave no converged points from raw coordinates. On the dev
  twins the solver answered 55% (Re <= 1e5) and 62% (Re > 1e5) of the measured angles.
- A CST bug cost an hour of bank time: least-squares CST fits of sharp trailing edges often returned a slightly
  negative TE thickness, i.e. crossed surfaces; 543 of 1,457 pool shapes, all of whose sweeps failed. Fits with
  te < 0 are now redone with te = 0, and the bank restarted.
- Considered and not used: `pyfoil` (XFLR5's C++ XFOIL port, GPL, a compiled wheel for Python 3.12). Running
  its compiled extension was refused by the machine's permission system (an unreviewable binary from a download),
  so it was dropped. It also has no Reynolds-number setter.

## Physics checks (work/verify_physics.json)

| Check | Result |
|---|---|
| Panel method vs exact Joukowski airfoil (conformal map), alpha 0/4/8 | Cl within 0.10-0.22%; Cp mean abs error 0.0012-0.0021 |
| Thin-airfoil theory, NACA 2404 | zero-lift angle -2.096 vs -2.077 deg; slope 1.031 x 2 pi (thickness rule 1 + 0.77 t/c = 1.031) |
| Published XFOIL 6.99 polars (airfoiltools.com, Ncrit 9): NACA 0012, 2412, 4412; 42 converged points | Cl mean abs diff 0.0037, Cd median 0.5%, Cm 0.0007 |
| NeuralFoil numpy port vs the same XFOIL polars | Cl 0.0025, Cd 0.6%, Cm 0.0004 |
| NACA 2412, Re 3e6 vs Abbott & von Doenhoff (approximate book values) | aL0 -2.21 (book ~-2.0), slope 0.111/deg (~0.105), Clmax 1.75 (~1.6), Cdmin 0.0051 (~0.006), Cm -0.052 (~-0.045) |
| JS port of the network vs Python | max relative difference 3.6e-14 |

## Data

- UIUC LSAT volumes 1-5 (Selig et al.), tabulated ASCII from m-selig.ae.illinois.edu. GPL v2 for the data.
  Volume 1 says its moment data are meaningless; those Cm are dropped. Only runs whose comment says "clean"
  and does not mention trips, flaps, Gurney flaps, tape or serrations are used. 82 airfoils, 38 families.
- Coordinates: the as-tested `.dap` files in volumes 1-3 where they exist, else the design `.dat` from the same
  volume, else the UIUC coordinate database. Every shape is refitted with CST (8 weights a side, LE term, TE
  thickness); fit rms median 2.5e-4 c, worst tested airfoil 1.4e-3 c (E387 model C, a measured file).
- Split frozen by family before training (`splits.json`, seed 20261122): dev = 7 airfoils from 4 families
  (A18; FX 63-137, FX 74-CL5-140 MOD; M06-13-128, M6 65%, M6 85%; Trainer 60), exam = the other 75.
- Training shapes: the UIUC coordinate database minus every tested airfoil by name and minus any shape within
  0.002 c rms of a tested one; 1,457 shapes, 174 held out for the frozen simulated set; plus blends.

## Twin test (dev, before any network)

The solver's own polar for each dev airfoil at its tunnel Reynolds numbers, against the tunnel:
- It gives no answer at 40-45% of the measured angles; most gaps are near and past stall, and FX 63-137
  gave none at all.
- Re <= 1e5: stall 4.2 deg early and Clmax 0.11 low on average (the sweep stops where Newton fails);
  drag 13% high.
- Re > 1e5: stall 0.5 deg early; drag 15% low.
- Pre-stall Cl error 0.10-0.11.
The drag trend with Re was built into physics version p1c (see the ratchet). The stall gap was not.

## Machine notes

- Load average 7-9 on 2 cores for most of the build; one sweep took 30-50 s wall instead of ~10 s.
- Never `pkill -f`: other builds run `simbank.py`-like names. Kill by PID.
- numba recompiles everything when mfoil_fast.py changes (about 40 s); the cache lives in numba_cache/.

## 2026-11-22 — imported into ML Ratchet

- Copied from `for_repo/models/airfoilnet/` of `airfoilnet-v1.0.zip` (22 Nov 2026) with `tests/golden/airfoilnet.txt`. Every `.npz` loads with
  `allow_pickle=False` and holds no object arrays; inference imports only numpy and scipy.
- `demo.py` run from this folder reproduces the shipped golden output line for line (timing aside), so
  `tests/golden/airfoilnet.txt` is the zip's own file, unchanged.
- No training script was re-run, and nothing was fetched.
- `order: 34`.
- The shipped weights are the MLP (`meta.arch = mlp`, 3 members, no thin-airfoil anchor). `airfoilnet.py`'s docstring
  still describes a DeepONet with thin-airfoil anchors (rejected rungs r2-r4); the code handles both, so the text is
  stale, not the output. Left as shipped.
- The UIUC excerpts in `examples/` and the polars in `lab/index.html` are GPL v2 data; `examples/LICENSE-GPL-DATA.txt`
  stays next to them, and `DATA_LICENSES.md` says so.
- Training scripts read and write a flat build folder relative to this one (`data/`, `release/`, `work/`), not
  `training/_paths.py`. A folder `.gitignore` keeps `data/` and `release/` out of git; `work/` is ignored repo-wide.
  Some scripts expect `splits.json` at the folder root (it ships as `training/splits.json`). The two shell wrappers
  (`run_bank.sh`, `chain_A.sh`) now `cd` to `training/` instead of the sandbox folder.
- Not done: no site instrument (no `export_instrument.py`; the lab page has the JS port, == numpy to 1.3e-14).

## 2026-11-22 — PR #13 review: training imports and the lab explorer

- The training entry points (`train.py`, `calibrate.py`, `exam.py`, `baselines.py`, `twin_dev.py`, `web_webdata.py` and
  its copy `lab/src/webdata.py`) put the flat build's `release/` on the path but not this folder, so `import airfoilnet`
  failed in the repository layout. They now add the model folder after `release/` (a rebuilt release still wins).
- The lab explorer predicted every run from the default build's geometry (`a.p`). 31 runs of E387, SA7036, SD7032 and
  SD7037 come from other physical builds and carry their own geometry, which the NeuralFoil and solver curves already
  used. `drawEx()` now uses `run.p || a.p` (template, its training copy and the shipped `lab/index.html`; one expression
  changed, page not rebuilt). The exam (`training/exam.py`) is not affected.
- The frozen training snapshots `work/snap_A.json` (151 sweeps, rungs r1-r5) and `snap_B.json` (256 sweeps, r6-r7) were
  not shipped, and no script writes them. `train.py` now uses them if present, else the first 151 / 256 sweeps by id
  with a warning. A rebuild therefore cannot reproduce the exact training sets of the released rungs.
- The training scripts built `models/airfoilnet/work` by hand (32 places in 14 files), ignoring `$ML_RATCHET_WORK`. They
  now take `WORK` from `training/_paths.py`. `data/` and `release/` stay next to the model folder (ignored by its
  `.gitignore`).
