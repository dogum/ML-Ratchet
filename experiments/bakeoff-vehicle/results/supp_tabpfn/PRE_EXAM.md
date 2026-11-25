# TabPFN supplementary exam: state before the exam (24 Nov 2026, written before `exam_supp.py tabpfn --run`)

Why: `tab_tabpfn` and `tab_tabpfn_conformal` were excluded from the main exam (20261123T204251) before it ran, for
compute only (registry/excluded.json). Handoff item 17 asks for a dated supplementary exam, reported beside the main
board and never folded into it. Owner's yes to the weights download: 24 Nov 2026.

**The artefact here.** Rebuilt in this repository from its registry record:
- weights/ from Hugging Face (Prior-Labs/TabPFN-v2-reg): tabpfn-v2-regressor.ckpt, config.json, LICENSE.txt, all three
  equal to the registered sha256;
- pylib/ by `pip install --no-deps --target` of tabpfn 10.2.0, pydantic-settings 2.16.3, pydot 4.0.1, safetensors 0.8.0,
  skrub 0.10.1. Every TabPFN, skrub, pydot and pydantic-settings code file equals registration; 27 files differ: the
  Windows build of safetensors (\_safetensors_rust.pyd for the Linux .so, and its .py stubs) and pip's install metadata;
- card.md, ctx.npz, featsel.py, predict.py, ratchet.json, tp_config.json are the shipped files (equal to registration);
- TabPFN's other dependencies, which the cloud machine had system-wide (pydantic 2.14.4, einops 0.8.2,
  huggingface-hub 2.4.0, tqdm 4.70.1, python-dotenv 1.2.4, typing-inspection 0.4.5, annotated-types 0.8.0), are in
  harness/work/tabpfn_deps (git-ignored), appended to sys.path.
- Environment: Windows 11, torch 2.11.0+cu128, scikit-learn 1.9.1, numpy 2.5.4, one thread, CPU.

**It is not bit-identical to the cloud run.** Dev rehearsal (exam_supp.py tabpfn --rehearsal, dev records only):
dev D16 0.1240 here against 0.1249 at registration (−0.0009). The reference arm enet_v21 gives 0.11330 here, its
recorded value (0.1133), so the dev records are the same; the TabPFN difference is in the arm's run, most likely the
torch / scikit-learn versions (the cloud's are not recorded). It is under the ~0.005 seed-to-seed spread of dev D16
(STATUS.md, Limits). The registered pre-exam numbers stand (sim val skill 0.563, dev D16 0.1249); this file records
the run that will be graded.

**What the supplement scores.** The two TabPFN arms and, beside them, every other table-track arm and the reference
rows, on the main exam's records rebuilt by build_real; the others' per-unit scores must equal the main exam's. One
run; results in this folder.
