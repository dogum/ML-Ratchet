# Prior Wars: experiment scripts

Method experiment C1. Write-up: [`docs/experiments/prior-wars.md`](../../docs/experiments/prior-wars.md). Page: the
site's `experiments/prior-wars/` (source: `site/public/experiments/prior-wars/index.html`, built by `make_page.py`).
Results: `results.json` and `RESULTS.md` in this folder. The pre-registration with its freeze log is in `PREREG.md` and
`prereg_log.json`. Per-model tuning, exam scores and logs are in `scores/<model>/`.

These scripts ran outside the repository, on a shared 2-core build machine, on 18-19 Nov 2026. They read each model's
own feature code, banks and real splits from the sibling project folders. In the repository they reach the same things
through `pw_paths.py`: the released models in `models/<id>/`, their training code in `models/<id>/training/`, and
their banks, real features and checkpoints in each model's own training workspace (`$ML_RATCHET_WORK/<id>`, else
`models/<id>/work/`). They are a record of what was run, not a `mlratchet` command. Inference stays out of the model
folders: nothing here is imported by any model.

| File | What it does |
|---|---|
| `harness.py` | One protocol for every model. Stages: `tune_dev` (G-dev settings by k-fold inside dev), `tune_sim` (G-sim settings on 400 rows of the frozen simulated set), `exam_predict` (every TabPFN arm on the exam inputs; refuses to run unless the model's PREREG section is frozen) and `score` (the exam scored once, with a paired bootstrap over exam units; refuses a second run). |
| `adapters/<model>.py` | Per model: the dev rows, the exam inputs, the exam truth (read only by `score`), the simulated pool, the frozen simulated set, P's predictions from the released weights, the metrics, the headline and the card's baselines. |
| `tp_common.py` | TabPFN-v2 regressor and classifier on CPU (one thread, memory-saving mode). |
| `pwmetrics.py` | Metrics. Errors are negated in the headline, so higher is better. |
| `freeze.py`, `prereg_section.py` | Freeze PREREG.md (a SHA-256 of the file as it stands) and write a model's section from its tuning results. |
| `runner.sh`, `jobenv.sh` | A sequential queue: one TabPFN process at a time, under a 2 GiB address-space cap and `nice`. |
| `make_results.py`, `make_page.py` | Write `results.json` and the tables, and build the Prior Wars page. |
| `pw_paths.py` | Where the scripts read and write in the repository (added on import, 21 Nov 2026). |

## To rerun

Everything a rerun writes goes to `WORK` = `$ML_RATCHET_WORK/priorwars/`, else `experiments/priorwars/work/`
(git-ignored). The shipped `PREREG.md`, `prereg_log.json`, `results.json` and `scores/` are the 18-19 Nov record and
are never written to.

1. Install `tabpfn==10.2.0` in a separate venv. The harness forks one child per fit (`os.fork`), so it needs Linux or
   macOS.
2. Fetch the TabPFN-v2 checkpoints (`Prior-Labs/TabPFN-v2-reg`, `Prior-Labs/TabPFN-v2-clf`) into `WORK/weights/`, or
   point `TABPFN_WEIGHTS` at them. They are open, under the Prior Labs License 1.1. Do not commit them.
3. Rebuild each model's banks, real features and checkpoints with its own training scripts (see each model's
   `training/_paths.py`). P for EnergyNet and PlantDetective is their v1.0 release, not the current one (see below).
4. For each model, run `harness.py <model> tune_dev`, then `tune_sim`.
5. Start a new pre-registration in `WORK/PREREG.md`. Append the model's section (`prereg_section.py`) and freeze it with
   `freeze.py <model>`, which writes `WORK/prereg_log.json`. `runner.sh` does this for a queued `prereg` stage.
6. Run `exam_predict`, then `score`. Then `make_results.py` and `make_page.py` write `WORK/results.json` and
   `WORK/page/prior-wars.html`.

On one shared CPU core, the six models took about 6 hours.

## Import notes (21 Nov 2026)

- **Copied from the release's `for_repo/`.** `package.sh` was left out: it only zipped the sandbox folder.
  The per-model bootstrap replicates (`exam_boot.npz`, 0.1-0.4 MB each) and exam predictions (`pred/`) were left out
  too. `make_results.py` needs them, with each model's `prereg_section.md`, to rebuild `results.json`.
- **`PREREG.md` is byte for byte as frozen.** It still quotes sandbox paths (`/home/claude/...`) in section 3 and in
  the adapter docstrings it copied. Editing them would break the hashes. All six hashes in `prereg_log.json` match
  the file (checked offline on import; `tests/test_priorwars.py` checks them again).
- **Path changes only.** Each sandbox path in the scripts now goes through `pw_paths.py`. No protocol, seed, setting or
  metric changed. From `results.json`, `make_page.py` rebuilt the release's page byte for byte. For the site,
  `page_template.html` then gained a doctype, a viewport and description tag and the title "Prior Wars · ML Ratchet",
  and lost a host-only meta tag. The rest of the page is unchanged.
- **Which P.** P is the release that was on the build machine on 18-19 Nov:
  - GridPulse v1.0 (rung r5), BasinDetective v1.0, BuildingDetective v1.0 (R9): the current releases.
  - PlantDetective R4_ens: v1.0. The repository released v1.1 (R6) on 15 Nov.
  - EnergyNet r6_ens: v1.0. The repository released v2.0 and v2.1 on 14-15 Nov. P's ICE MdAPE on the subsampled
    exam (9.0 %) matches v1.0 re-scored on all 14,469 VED petrol trips (9.1 %), not the 8.6 % of v2.1's card.
  - The results are unchanged. They answer the question for those two models at v1.0.

Built with PriorLabs-TabPFN. The TabPFN-v2 weights are under the Prior Labs License 1.1 (Apache 2.0 plus an attribution
clause); `tabpfn` 10.2.0 is Apache-2.0. Neither is redistributed here.
