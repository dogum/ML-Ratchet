# Training scripts (as-run snapshot)

These are the exact scripts that produced the released weights. They need PyTorch, LALSuite (`lalsimulation`), scipy,
h5py and astropy. They were first run in a cloud sandbox; paths now go through `_paths.py`: code lives here, and
everything the scripts download or write (`data_raw/`, `ckpt/`, banks, release builds) goes in `models/chirpnet/work/`
(or `$ML_RATCHET_WORK/chirpnet/`).

| Stage | Scripts |
|---|---|
| Physics and data | `gwphys.py` (whitening, PSDs, bank), `build_bank.py`, `sampler.py`, `bank_phys.py`, `pe_sampler.py`, `bns.py` |
| Models | `model.py` (ChirpNet, ChirpNet2S, Cascade), `flow.py` (posterior flow), `bns_train.py` (BNS models) |
| Training | `train.py`, `train_pe.py`, `bns_train.py` |
| Judging | `evaluate.py`, `realeval.py`, `ratchet.py`, `cascade_tune.py`, `eval_pe.py`, `bns_real.py`, `bns_ratchet.py` |
| Release | `export.py`, `export_pe.py`, `export_bns.py`, `release_build.py`, `model_card.py`, `make_zip.py` |
| Page | `export_js.py`, `build_web_data.py`, `build_pe_web.py`, `build_bns_web.py`, `build_page.py`, `viz.py` |

`logs/` holds the ratchet decisions for every rung (`ratchet_log.json`, `bns_ratchet_log.json`), the veto tuning, and
per-rung evaluation and real-data exam outputs.
