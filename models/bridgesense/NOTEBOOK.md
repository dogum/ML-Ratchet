# BridgeSense — Lab Notebook

## 2026-11-24 — Build from first principles to three-bridge exam

### 1. Data scouting & licensing hygiene
- **KW51 railway bridge (Leuven, Belgium; 115 m steel bowstring arch):** Zenodo `3745914` (`trackedmodes.zip`, 12.9 MB, 11,328 hourly rows Oct 2018 – Jan 2020, 14 tracked modes, air `tVL` and steel `tBD31A` temperatures, documented retrofit 2019-05-15 to 2019-09-27) + FEM `17366772` (`model_kw51.m`). License is `CC BY-NC-SA 4.0`, so KW51 stays strictly in git-ignored `work/` (never redistributed in `examples/` or the site instrument).
- **Concrete Box-Girder Bridge (Hamburg, Germany; Koehncke et al. 2026, arXiv:2412.15713):** Zenodo `20522191` (`box_girder_bridge_eigenfrequencies.h5`, 0.89 MB, `CC BY 4.0`), 5,264 half-hourly tracked F1/F2 rows from Jan 1 to Apr 21 2024 (includes a `-7.9 °C` January freeze and documented 680 / 1,420 / 2,160 kg sand-bag static load tests in Table 3). Paired with NASA POWER hourly `T2M` at Hamburg (53.55°N, 10.0°E).
- **Hell Bridge Test Arena (NTNU, Norway; 35 m 1905 riveted steel truss):** Zenodo `14632942` (`data_100Hz.h5`, 2.52 GB, `CC BY 4.0`), 50 recordings (10 `UDS` undamaged + 40 imposed local/global damage `DS1–DS8`) across 59 accelerometers with per-recording temperature in HDF5 attributes. Classical multi-channel Welch SVD / FDD (`extract_hbta_modes.py`) anchored strictly on `UDS` extracted 7 global truss modes (`3.125, 7.300, 9.399, 10.718, 12.817, 15.015, 18.262 Hz`) across 8 × 120-s windows per recording.
- **KU Leuven Z24:** Registration-gated (`bwk.kuleuven.be/bwm/z24/registration`). Not bypassed; logged in `help_wanted`.

### 2. Physics checks & numerical catches (`training/physics_check.py`)
Initial run caught three real numerical issues before any bank was generated:
1. Simply-supported 2-D beam mode 6 (`19.768 Hz` vs analytical bending `21.643 Hz`) was actually the 1st **axial** bar mode $\frac{1}{4L}\sqrt{EA/\rho A} = 19.764\text{ Hz}$. Increasing `EA` in the pure Euler–Bernoulli check verified bending modes 1–6 to **`2.60e-4`** (0.026%) and cantilever modes 1–4 to **`3.03e-5`**.
2. 1-D Crank–Nicolson deck heat diffusion initially averaged nodal values with `np.mean(T)` (biasing endpoint `T[0]` by 6.3%) and drove both slab faces. Switching to trapezoidal quadrature and spectral `scipy.signal.lfilter` cut runtime from 114 ms to `< 2 ms` per 8,760-hour trace and matched the exact periodic slab solution to **`5.87e-4`** in amplitude (`0.6059` vs `0.6056 K`) and `0.0060 rad` in phase lag.
3. Widening the modal projection subspace matched full `scipy.linalg.eigh` under 20% local damage + freeze to **`2.04e-13`**.

### 3. Frozen split & unlabelled noise pool (`splits.json`, `noise_pool.npz`)
- Frozen before any training (seed `20261124`).
- KW51 early autumn baseline (`2018-10-01 .. 2018-11-11`, 1,008 h) supplied both the warm reference window and **436 real healthy OMA noise & missing-mode windows** (residual std `0.326%`, mean mode availability `56.0%`).
- KW51 Dev: 10 blocks (`4 healthy_mild`, `2 healthy_freezing`, `4 post_retrofit`). KW51 Exam A: 49 blocks (`10 healthy_mild`, `10 healthy_freezing`, `18 during_retrofit`, `11 post_retrofit`). Concrete Box-Girder and HBTA: 0 blocks in dev (100% zero-shot exams).

### 4. Twin test & ratchet surprises (`ratchet.yaml`)
- **`r1` (`p1` textbook linear thermoelasticity):** Scored `0.726` on `val1`, collapsed to **`0.350`** on `val2` (`cold_spec = 0.112`) and `0.302` on `val3`. On real KW51 Dev it fired 3 false alarms (`Dev Brier = 0.2386`), including `P = 0.98` on healthy April/May blocks whose linear residual was only `-0.25%`.
  - **Twin-test finding:** Real healthy KW51 weeks carry `~0.15–0.22%` per-mode slow seasonal/moisture/solar-heating drift (`maxZ ~ 1.2–2.2`), whereas `p1` white noise averaged away over 32 steps. Meanwhile KW51 post-retrofit shifts modes in *both* signs (`-1.2%` on mass-loaded modes to `+2.8%` on stiffened diagonals, `std = 1.4%`, `maxZ ~ 19`).
- **`r2` (`p2`: 1-D Fourier lag + freezing + axial restraint + warm→cold extrapolation):** Jumped from `0.302` to `0.806` on `val3` (`0.908` on `val2`), and crushed KW51 Dev (`Brier 0.2386 → 0.0000`, `healthy_dk_mae 0.162% → 0.031%`, `freeze_P 0.343 → 0.002`).
- **`r2_no_xattn` (ablation removing baseline-to-query thermal cross-attention):** Barely moved `val2`/`val3` (`-0.005`), **yet exploded real KW51 Dev winter freezing false-alarm probability 240× (`0.002 → 0.456`, `Brier → 0.0603`)**. Rejected by the ratchet.
- **`r3`, `r4_mix`, `r5_no_anchor`:** Pure `p3` (`r3`) and un-anchored `p3` (`r5_no_anchor`) gained `+0.058` and `+0.073` on `val3`, **but failed the KW51 Dev gate**! Strikingly, `r5_no_anchor` predicted `-0.36%` and `-0.25%` (wrong sign!) on KW51's `+1.15%` stiffening retrofit — regressing toward the training prior mean on mixed-sign modes.
- **`r6_mix_no_anchor` (mixed `p2 + p3` 24k bank, direct output heads):** Classic SwellCast/StarQuake/BearingNet pattern — mixing clean `p2` and real-noise `p3` banks both achieved the top `val3` skill (`0.8808`, `+0.074` over `r2`) **and** passed the KW51 Dev gate (`AUC 1.000`, `Brier 0.0000`, `dev_dk_mae 0.071%`, `healthy_dk_mae 0.02%`). Released.
- **`r7_ens3` (3-seed anchored ensemble):** Scored `0.8764` (`-0.004` vs incumbent) — rejected.

### 5. One-shot three-bridge exam takeaways
- **Export parity:** `bridgesense_weights.npz` (`670.3 KB`, `179,144` params), numpy vs PyTorch max abs diff **`2.67e-6`**.
- **Biggest win (zero-shot concrete freeze-thaw):** On the unseen 2-mode Hamburg concrete box-girder bridge during `-7.9 °C` and `-5.0 °C` January freezes, Linear OLS and KW51-dev-tuned Bilinear hallucinated `+9.51%` and `+6.31%` structural shifts; BridgeSense read **`-0.01% ± 1.16%`** and **`+0.01% ± 0.36%`**, cutting healthy freeze MAE 7× (`0.881%` vs `6.372% / 6.246%`) and lifting cold-vs-load AUC from `0.146` to **`0.729`**.
- **Biggest honest losses:** (1) HBTA local secondary-member cracks (`DS1–DS2`) barely move global frequencies (`all-damage AUC 0.595` vs OLS `0.645`, though global primary damage `DS5–DS8` wins `0.800` vs `0.690`); (2) short constant-temperature shaker campaigns (`HBTA UDS`) lack a baseline temperature gradient, causing `UDS` false alarms (`mean P = 0.825`); (3) zero-shot 68% coverage is under-covered (`50.0%` concrete, `33.3%` HBTA vs `96.8%` on KW51).
