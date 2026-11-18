# WaterSim

A daily (and hourly) hydrology simulator built from standard, published physics, for training networks on. It
turns basin-average weather into streamflow through snow, soil, groundwater, river channels and reservoirs. Every
run closes its water balance to round-off and returns the hidden states and the true parameters as labels.

Part of ML Ratchet ("Trained on physics. Graded on reality."), Phase 1 of the Watershed project. Phase 2 trains
FlowCast, SnowSense, BasinDetective and ReservoirPilot on it. B's real-data package (`water/waterdata`, frozen
splits in `water/data/proc/splits.json`) supplies the real forcing, attributes and the dev/exam rules this package
obeys. Every check below is a JSON file in `water/checks/` with its numbers.

What is in it, in one line each:

- **Snow:** degree-day with refreeze, liquid water and elevation bands (HBV-light), or a daily energy-balance model.
- **Runoff models (four structures):** HBV, GR4J, SAC-SMA and VIC-lite.
- **PET:** FAO-56 Penman-Monteith, Priestley-Taylor, Hargreaves, Oudin, or the forcing's own PET.
- **Routing:** unit hydrographs, Muskingum, Muskingum-Cunge, kinematic wave and multi-reach networks.
- **Human influence:** reservoirs (storage-area-elevation curves, operating policies or your own policy function,
  spill, evaporation, hydropower) and irrigation withdrawals with return flows.
- **Training data:** a random-basin generator (priors corrected on the dev basins), a stochastic weather generator,
  and reuse of real non-exam forcing.
- **Batch mode and calibration:** numba batch runs; a basin-decade takes about 2 ms in batch.

## Quick start

```python
import sys; sys.path.insert(0, '/home/claude/water/watersim')
import watersim as ws

# a real dev basin (B's data; SRAD there is NET shortwave and is handled)
sys.path.insert(0, '/home/claude/water'); import waterdata as wd
bid = wd.basins('camels', split='dev')[0]
out = ws.simulate(wd.spec(bid), (wd.forcing(bid), wd.forcing_meta(bid)), '1999-10-01', '2008-09-30',
                  model='hbv', snow='degree_day', routing='uh')
out['ts'][['P', 'SWE', 'soil', 'groundwater', 'ET', 'Q', 'Q_m3s']]     # daily, UTC index
out['ledger']['ok'], out['ledger']['closure_rel']                       # True, ~1e-16
out['labels']['bfi_lyne_hollick'], out['labels']['recession_k_daily']

# random training basins with parameters for every model, and weather for them
specs = ws.generate_basins(1000, rng=0)                    # documented parametric prior (+ dev corrections)
specs = ws.generate_basins(1000, rng=0, prior='camels')    # jittered DEV-basin attributes (B's splits)
f = ws.generate_weather(specs[0], '2000-01-01', '2010-12-31', rng=1)
df, meta, info = ws.realdata.random_forcing(rng=2, years=10)          # real forcing, non-exam years only
out = ws.simulate(specs[0], f, '2001-01-01', model='sacsma', snow='energy_balance', routing='muskingum')

# many parameter sets at once (calibration, ensembles)
r = ws.simulate_batch(spec, forcing, [{'FC': 200, 'K1': 0.1}, {'FC': 300}], model='hbv', start=..., end=...)
r['Q']                                                     # (N, T) mm/day
ws.list_models()
```

`python3 demo.py` takes a few seconds and writes `demo_watershed.png`. It runs a snowmelt dev basin
(camels_09035800) with all four dev-calibrated models, plotted against observations for WY2005-06 (NSE 0.84-0.87).
It also shows the hidden states next to ERA5-Land SWE, the ledger, and a generated regulated basin: natural vs
regulated flow, reservoir storage, hydropower and labels.

The package needs numpy, scipy and pandas. numba (0.68 here) is used when installed: without it, the same
kernels run as plain Python, about 100x slower. The reference R packages live only in the isolated
`watersim/.refenv`.

## API (the shared contract)

```python
out = simulate(spec, forcing, start=None, end=None, model='hbv', params=None, snow='degree_day', routing='uh',
               reservoirs=None, init=None, spinup='1y', rng=None, pet=None, irrigation=None,
               pet_snow_reduction=True, return_bands=False, labels=True, infiltration_excess=None)
```

| argument | meaning |
|---|---|
| `spec` | Basin spec dict (contract fields). Missing fields get documented defaults; `out['spec']['src_fill']` says where each came from. Caravan units are converted: `slope_unit='degrees'` to m/km, `soil_conductivity_unit='mm/h'` to cm/h. `spec['reservoirs']` (list) and `spec['params']` (nested by model) are used when present. |
| `forcing` | DataFrame with a UTC DatetimeIndex and the contract columns `P`, `T_mean`, `T_min`, `T_max`, `SRAD`, `vp` (or `RH`), `wind`, optionally `PET`. It can also be `(DataFrame, meta)` or a parquet path. The step comes from the index: daily, or sub-daily for hbv/sacsma/vic_lite. P and PET are amounts per step. If meta says SRAD is net shortwave (B's `forcing_meta`), the engine converts it (see Limitations). |
| `start`, `end` | Inclusive dates of the reported window. The default is the whole forcing. |
| `model` | `'hbv'`, `'gr4j'`, `'sacsma'`, `'vic_lite'`, or `'none'` (snow and routing only). |
| `params` | Flat (`{'FC': 300, 'CFMAX': 4}`) or nested dict. It overrides `spec['params']`, which overrides the attribute transfer functions, which override the defaults. `out['param_source']` says which layer each value came from. Unknown names raise. |
| `snow` | `'degree_day'`, `'energy_balance'` or `'none'`. |
| `routing` | Channel routing after the model's own hillslope unit hydrograph (UH). `'uh'`: gamma, shape 3, mean travel L/CH_VEL. `'muskingum'`: variable-parameter Muskingum-Cunge over the main channel, centroid length L/2. `'kinematic'`: kinematic wave with lateral inflow along the whole channel. `'none'`: no channel routing. |
| `reservoirs` | List of reservoir dicts (arguments of `reservoir.Reservoir`, plus `upstream_frac`, `inflow_from`, `capacity_years`) or `Reservoir` objects. `policy` can be a name or a function `obs -> release m3/s`. Daily step only. |
| `init` | A previous `out['final_state']` (to chain runs; no spin-up), or `{'model': {state: value}, 'snow_swe_mm': x}`. |
| `spinup` | `'1y'`, `'2y'`, `'180d'` or `0`. Taken from the forcing before `start` when it exists; otherwise the first part of the window is run once as a warm-up (`out['spinup']` says which). |
| `pet` | `'forcing'`, `'penman_monteith'`, `'priestley_taylor'`, `'hargreaves'` or `'oudin'`. The default is the forcing's PET when present, else Penman-Monteith. |
| `irrigation` | `None` uses `spec['irrigation_frac']`; `False` turns it off; a dict changes the settings. |
| `infiltration_excess` | `None`: on for sub-daily steps only. Can also be `True`, `False`, or a capacity in mm/h. |
| `obs_noise`, `rng` | `obs_noise=True` (or a dict of `noise.streamflow_obs` settings) adds a gauge-like `Q_obs` column drawn with `rng`. The physics itself is deterministic. |

`out` is a dict:

- `ts`: DataFrame on the forcing grid, all water in mm per step over the basin area.
  - Precipitation: `P_forcing`, `P` (after the snowfall correction), `rain`, `snowfall`.
  - Snow: `melt`, `refreeze`, `liquid_input` (rain + snowpack outflow), `SWE`, `snow_liquid`, `sublimation`
    (+ = loss), `SCA`, `snow_energy` (EB net energy, MJ/m2).
  - Evaporation: `PET`, `PET_soil` (reduced by snow cover), `ET_soil`, `ET` (soil + sublimation + reservoir).
  - Model runoff: `Q_model` (before channel routing), components `Q_quick`, `Q_inter`, `Q_base` (and `*_routed`
    after channel routing), `exchange` (GR4J groundwater exchange, SAC-SMA deep loss; + = gain), `recharge`,
    `surface_runoff`, `infiltration_excess`.
  - Storages: `soil`, `groundwater`, `routing_store` (model UH buffers and the GR4J routing store share),
    `channel_storage`.
  - Flows: `Q_natural` (after channel routing), `withdrawal`, `return_flow`, `irrigation_store`.
  - Reservoirs: `res_storage_hm3`, `res_release_m3s`, `res_spill_m3s`, `res_evap_mm`, `res_power_MW`,
    `res_withdrawal_mm`.
  - Outlet: **`Q`** (mm per step) and **`Q_m3s`**.
- `ledger`: the water balance over the window (mm).

  P_in + exchange gains = ET_soil + sublimation + reservoir evaporation + Q_out + exchange losses
  + consumptive withdrawals + dS (snow, soil, groundwater, model routing, channel, irrigation return, reservoirs)

  It also gives `closure_mm`, `closure_rel`, `max_daily_residual_mm` (the same balance checked every day) and
  `ok`, which requires closure < 1e-8 relative and every daily residual < 1e-8 mm.
- `labels` (see below), plus `params` (model / snow / routing dicts), `param_source`, `pet_method`, `spinup`,
  `channel` (geometry), `bands`, `reservoirs` (each with its daily series and object), `final_state`,
  `swe_bands` (if `return_bands`), `dt_days` and `wall_s`.

Other entry points:

| function | what |
|---|---|
| `simulate_batch(spec, forcing, params_list, model, snow, start, end, spinup, outputs=('Q','SWE','ET','Q_base'), routing='none'\|'uh')` | N parameter sets for one basin, one numba call; returns (N, T) arrays |
| `simulate_many(specs, forcings, model, ...)` | N basins, each with its own spec, parameters and forcing (same window) |
| `simulate_network(subbasins, forcings, topology, start, end, model, ...)` | sub-basins joined by Muskingum-Cunge reaches (tree topology); per-node flows and a network ledger in hm3 |
| `noise.streamflow_obs(q, rng, ...)`, `noise.forcing_error(df, rng, ...)` | gauge error (AR(1) lognormal, rating revisions, larger low-flow error, rounding, gaps) and forcing error, for training inputs/targets |
| `generate_basins(n, rng, prior='default'\|'global'\|'camels', models=..., regulation=True, attributes=None)` | random specs with `params[model]`, `params['snow_dd'/'snow_eb'/'routing']`, `latent`, `climate`, optional `reservoirs`, and `donor_id` (camels prior) |
| `params_from_attributes(spec, model, snow, latent=None, rng=None, adjust=True)` | the attribute -> parameter transfer functions (an ungauged prior) |
| `generate_weather(spec, start, end, rng, climate=None)`, `fit_climate(real_df)` | stochastic daily forcing in the contract format; `fit_climate` fits the generator to a real basin's forcing (use non-exam years) |
| `realdata.random_forcing(rng, years, region, basin_id)`, `realdata.nonexam_forcing(id, start, end)`, `realdata.donor_table()`, `realdata.free_windows()` | B's data with the split rules enforced (raises `PermissionError` on exam years) |
| `calibrate.calibrate(spec, forcing, obs, start, end, model, snow, objective='kge'\|'nse'\|'kge_lognse')` | per-basin calibration (JADE differential evolution, 40 x 80, about 5 s) |
| `routing.*` | `triangle`, `gr4j_ordinates`, `gamma_uh`, `nash_uh`, `convolve_uh`, `muskingum`, `muskingum_cunge`, `kinematic`, `route_network`, `channel_geometry` |
| `reservoir.Reservoir(...)` | `.level(S)`, `.area(S)`, `.storage_from_level(z)`, `.table()`, `.reset()`, `.step(inflow, release, evap_mm)` for RL/MPC, `.run(inflow, dates, evap, demand, policy)` |
| `pet.*`, `metrics.*` | PET: `compute_pet` and each method. Metrics: `nse`, `kge`, `log_nse`, `pbias`, `fhv`, `flv`, `peak_error`, `all_metrics`. |
| `signatures.*` | `lyne_hollick`, `eckhardt`, `bfi`, `recession`, `camels_signatures`, `richards_baker` |
| `list_models()`, `param_table(model, snow)`, `fill_spec`, `build_bands`, `clear_cache()` | introspection |

### Labels (`out['labels']`)

- **True parameters:** `params_model`, `params_snow`, `params_routing`; the latent factors are in the generated
  spec.
- **Water balance and climate:** `soil_capacity_mm` (FC, X1, UZTWM + LZTWM or W1M), `p_mean`, `pet_mean`,
  `aridity`, `frac_snow` (snowfall share), `frac_snow_t0` (CAMELS definition: P on days below 0 C),
  `evaporative_index`, `runoff_ratio`.
- **Hidden states:** `soil_moisture_mean_mm`, `groundwater_mean_mm`, `recharge_mean`.
- **Baseflow:** `baseflow_frac_model` (the true share of outlet flow from the slow store) and `bfi_lyne_hollick`
  and `bfi_eckhardt` (filters on the outlet flow, computed exactly as on a gauge record).
- **Recessions:** `recession_bn_a`, `recession_bn_b` (Brutsaert-Nieber), `recession_k_daily`, and the model-truth
  `k_slow_true` and `k_fast_true`.
- **Snow:** `swe_peak_mean_mm`, `swe_peak_wy_day_mean`, `melt_out_wy_day_mean`, `snow_days_frac`.
- **CAMELS signatures** (`sig_*`): `q_mean`, `runoff_ratio`, `stream_elas`, `slope_fdc`, `baseflow_index`,
  `q5`, `q95`, `high/low_q_freq/dur`, `zero_q_freq`, `hfd_mean`.
- **Other:** `flashiness_rb`, `withdrawal_frac`, `regulation_degree` (reservoir capacity / mean annual flow) and
  `n_reservoirs`.

For daily SWE, soil moisture and groundwater series, use `out['ts']`.

## Physics

All storages are in mm over the basin area. One daily step runs in this order:

1. **Bands:** split the forcing into elevation bands.
2. **Snow:** run the snow routine for each band.
3. **Area-weight:** combine the bands' liquid water into one basin input.
4. **Infiltration excess:** sub-daily steps only.
5. **Runoff model:** the soil and groundwater model, with its own hillslope UH.
6. **Channel routing.**
7. **Irrigation.**
8. **Reservoirs.**

**Elevation bands.**
- Bands come from `spec['elev_bands']`. Otherwise there are 5 equal-area bands between `elev_min_m` and
  `elev_max_m`, centred on `elev_mean_m`, or one band.
- Band temperature: T_b = T + TLAPSE dz, with dz in km and TLAPSE = -6.5 C/km by default.
- Band precipitation: P_b = P m_b, with m_b = max(0, 1 + PGRAD dz) normalised so that the area-weighted mean
  is P. The forcing is already a basin mean, so mass is exact.

**Degree-day snow** (HBV-light; Bergström 1976; Seibert & Vis 2012):
- The snow fraction falls linearly from 1 to 0 across TT +- TTI/2.
- Snowfall is multiplied by SFCF.
- Melt = CFMAX (T - TT) when T > TT.
- Refreeze = CFR CFMAX (TT - T) when T <= TT.
- Rain on snow joins the liquid store, which holds CWH x ice; the excess drains.

**Energy-balance snow** (daily, single layer, in the spirit of Walter et al. 2005 and UEB, Tarboton & Luce 1996):

  Q = (1 - a) Rs tau_c + eps sigma Ta^4 - 0.98 sigma Ts^4 + H + LE + G + c_w P_rain T

- **Albedo:** a = a_min + (a_max - a_min) exp(-age / ALB_DECAY). Age resets with snowfall and runs 3x faster
  when melting. A shallow pack (< ~15 mm) blends toward the ground albedo 0.2.
- **Longwave:** clear-sky emissivity 1.24 (e/T)^(1/7) (Brutsaert 1975), with the cloud correction
  (1 - 0.84 c) eps + 0.84 c (Unsworth & Monteith 1975), where c = 1 - Rs/Rso.
- **Turbulent fluxes:** H and LE are neutral bulk transfer at 2 m, r_a = ln(2/z0)^2 / (k^2 u). They are reduced
  by 1/(1 + 10 Ri) when stable and scaled by TURB.
- **Surface temperature:** Ts = min(Ta, 0).
- **Ground heat:** G = 0.173 MJ/m2/day.
- **Melt and refreeze:** positive Q first removes cold content, then melts at the latent heat of fusion. Negative Q
  refreezes liquid water, then builds cold content (capped at c_i x ice x (-Ts)).
- **Sublimation and deposition:** mass is LE/lambda. Liquid water is held as in the degree-day model.

**PET** (FAO-56, Allen et al. 1998):
- **Penman-Monteith reference ET0** (eq. 6), with G = 0 at the daily step.
- **Net radiation:** Rn from eqs. 38-40. When SRAD is net shortwave and `LRAD_net` is given (Caravan),
  Rn = SRAD + LRAD_net.
- **Priestley-Taylor:** alpha = 1.26.
- **Hargreaves & Samani (1985)** and **Oudin et al. (2005)**.
- **Open-water Penman (1948)** for reservoirs, with the Shuttleworth (1993) wind function.
- **Sub-daily steps:** daily PET is spread over the day in proportion to the sine of the solar elevation.
- **Snow cover:** soil PET is multiplied by (1 - SCA), with SCA = min(1, SWE/10 mm) per band.

**HBV** (HBV-light; Seibert & Vis 2012), with a lumped soil:
- **Soil:** recharge = I (SM/FC)^BETA, with the excess above FC added.
- **ET:** Ea = PET min(1, SM/(LP FC)).
- **Upper box:** SUZ loses PERC to the lower box, Q0 = K0 (SUZ - UZL)+ and Q1 = K1 SUZ.
- **Lower box:** Q2 = K2 SLZ.
- **Routing:** triangular weights over MAXBAS days.

**GR4J** (Perrin et al. 2003), a line-by-line port of airGR's `frun_GR4J.f`:
- **Interception:** net rainfall and net evaporation.
- **Production store X1:** the tanh formulas for Ps and Es.
- **Percolation:** S [1 - (1 + (4/9 S/X1)^4)^(-1/4)].
- **Unit hydrographs:** a 90/10 split between UH1 (X4) and UH2 (2 X4).
- **Exchange:** F = X2 (R/X3)^3.5.
- **Routing store X3:** Qr = R [1 - (1 + (R/X3)^4)^(-1/4)].

**SAC-SMA** (Burnash et al. 1973; Burnash 1995), following the NWS FLAND1 subroutine (as in hydromad's
`sma_sac.c`):
- **Stores:** upper and lower zone tension and free water, an ADIMP area, and PCTIM.
- **Percolation demand:** PERC = (LZFPM LZPK + LZFSM LZSK)(UZFWC/UZFWM)(1 + ZPERC DEFR^REXP).
- **Outflows:** interflow, primary and supplemental baseflow, and a PFREE split.
- **Losses and ET:** SIDE (deep loss), RSERV and riparian ET.
- **Increments:** 1 + 0.2 (UZFWC + excess) inner increments per day.
- **Routing:** a 2-parameter gamma UH, as in the CAMELS benchmark (Newman et al. 2015).
- **Differences from FLAND1:** two guards. ADIMP ET is never negative, and water removed by FLAND1's
  "zero tiny stores" rule is booked as ET, so mass is exact.

**VIC-lite** (in the spirit of Liang et al. 1994, ARNO/Xinanjiang; Todini 1996). It is structurally different
from the other three:
- **Runoff:** saturation-excess runoff from the variable infiltration capacity curve,
  i = i_m [1 - (1 - A)^(1/b)] (Liang 1994, eq. 13).
- **ET:** from two layers with a linear stress below 70 % of capacity.
- **Drainage:** Brooks-Corey, KSAT (W1/W1M)^EXPT.
- **Baseflow:** the ARNO curve (linear below WS W2M, quadratic above).
- **Routing:** a gamma UH.

**Infiltration excess** (sub-daily steps, or on request):
- Input above the capacity runs off as Hortonian flow, routed with the model's quick UH.
- The capacity is the saturated conductivity (spec `soil_conductivity`) x (1 - urban fraction), in mm/h x the
  step.

**Channel routing:**
- **Main channel:**
  - Length from Hack's law, L = 1.4 A^0.6 (mi, mi2).
  - Bankfull width B = 1.5 A^0.4 m.
  - Slope = 5 % of the hillslope slope_mean, clipped to 2e-4 - 0.03.
  - Manning n = CH_N (0.035).
- **'uh':** a gamma UH with shape 3 and mean L/CH_VEL.
- **Muskingum** (McCarthy 1938). Daily values are end-of-step samples, with inflow linear between them and
  substeps h <= K/10.
- **Muskingum-Cunge** (Cunge 1969), variable-parameter (Ponce & Yevjevich 1978).
  - Wide channel: c = 5/3 v, K = dx/c, X = 1/2 (1 - Q/(B S0 c dx)).
  - Grid: Courant ~ 1, with all reaches advanced together each substep.
- **Kinematic wave** (Chow, Maidment & Mays 1988, ch. 9). A = alpha Q^0.6, solved with an implicit
  backward box scheme and Newton's method.
- **Networks:** `route_network` routes sub-basins through a tree of Muskingum-Cunge reaches.
- **Channel storage** is the running sum of inflow minus outflow, so the ledger is exact.

**Reservoirs:**
- **Daily mass balance:** S' = S + (I - R - Spill) dt - E A(S), with release R limited by dead storage.
- **Spill:**
  - `'free'` (default): instant overflow above capacity.
  - `'weir'`: an ogee weir, Q = Cw Lw h^1.5, over a surcharge pool (hourly substeps).
- **Geometry:** a power-law storage-depth curve V = C (h/H)^b, so A = dV/dh, or tables.
- **Hydropower:** P = eta rho g H Q_turbine, with H = mean pool level - tailwater.
- **Policies:**
  - SOP (standard operating policy)
  - rule curve (a seasonal target storage)
  - Hanasaki et al. (2006)
  - linear
  - run-of-river
  - any function of the daily `obs` dict, which holds storage, level, area, inflow, mean inflow, demand,
    capacity, day of year and history.
- **Network position:** `upstream_frac` (the share of the basin draining to the dam), with `inflow_from` for
  cascades.
- **Water use:** withdrawals (`policy_params={'withdraw': True}`) leave the system as consumptive use.

**Irrigation** (optional):
- **Demand:** irrigation_frac x max(PET - rain, 0) / efficiency on days above 10 C.
- **Withdrawal:** taken from the river, at most 60 % of its flow.
- **Return flow:** 30 % of the withdrawal comes back through a linear store (k = 0.05/day). The rest is
  consumptive use.

## The generator (training data)

- **Attributes.** Draws come from:
  - `prior='default'`: documented parametric distributions, CAMELS-US-like (`generator.PRIOR_NOTES`).
  - `prior='camels'`: the 52 real DEV basins from B's specs. Positive fields are jittered by exp(N(0, 0.15)) and
    the donor id is kept. Exam basins are never used.
- **Parameters.** Four shared latent factors (storage, permeability, quickness, snow) plus per-parameter scatter
  feed transfer functions for every model. One basin is the "same" basin under all four models.
- **Calibrated on dev** (recipe step 5):
  - `tools/fit_prior.py` compares the transfer-function medians with the KGE-calibrated parameters of the dev
    basins.
  - It then applies half of the median log-shift (SHRINK = 0.5), plus the extra scatter the calibrations show.
    The result is stored in `watersim/data/prior_adjust.json`; `adjust=False` turns it off.
  - The choice between no, full and half correction was made on the signature match below. Half was best;
    full correction made streams too flashy.
- **Weather.** `generate_weather` is a WGEN-type Markov-chain / gamma / AR(1) generator (with `fit_climate` it
  copies a real basin's climate statistics). The other source is real forcing from
  any basin in non-exam windows only (`realdata.random_forcing`, which refuses exam years). `forcing.perturb`
  adds climate shifts (P x U(0.85, 1.15), T + U(-1.5, 1.5) C).
- **Human influence.** 30 % of default basins get reservoirs (storage-depth exponent b ~ 2.5 exp(N(0, 0.35)), from the dev dams):
  - one, or two in cascade;
  - storage 0.02-1.5 years of mean flow;
  - a random policy: SOP, rule curve or Hanasaki.

  25 % get irrigation.

## Checks (`water/checks/*.json`, re-run with `run_all.sh`)

| check | file | result |
|---|---|---|
| GR4J vs **airGR 1.7.9** (CRAN source compiled in an isolated R), help example L0123001 | `reference_models.json` | max \|dQ\| **1.0e-7 mm** (airGR's single-precision output); production store 7.7e-13 mm; AE 1.8e-14 mm; NSE 1990-99 **0.79882** vs the published **0.7988** |
| HBV vs **TUWmodel 1.1-1** (independent HBV code, Parajka et al. 2007), lumped Vils example 1976-2008 | `reference_models.json` | snow, soil moisture, ET, rain, melt identical to **<= 1e-12 mm**. Runoff NSE **0.9990** (no routing) and **0.9998** (MAXBAS 10), volume ratio 1.00001; the response routines are discretised differently by design. |
| HBV-light published example | - | not available (Windows GUI, no downloadable test series); TUWmodel stands in |
| Water-balance closure, 121 runs: all models x snow x routing, generated basins with reservoirs, cascades and irrigation, 8 real dev basins, and a 4-sub-basin network | `balance.json` | 121/121 ok; worst closure **1.4e-15** relative; worst daily residual **8.3e-12 mm**. Restarting from `final_state` reproduces one continuous run exactly (0.0 difference, all 8 model x snow pairs). |
| Linear-reservoir recession (HBV lower box; SAC-SMA primary baseflow) | `analytic.json` | errors 4.4e-15 / 2.0e-15 mm vs closed form. The storage matches S0 exp(-kt) to 2.4e-15; the recession fit recovers k = 0.95, b = 1.000. |
| Unit hydrographs | `analytic.json` | gamma step response vs S-curve 6.5e-13; Nash cascade ODE (solve_ivp) vs gamma ordinates 1.8e-10; HBV triangle MAXBAS = 3 equals the hand values 2/9, 5/9, 2/9 exactly |
| Muskingum, X = 0, vs the exact linear-reservoir ODE (K = 0.5, 2, 5 days) | `analytic.json` | max relative error 0.05-0.17 %, NSE >= 0.99999, mass 1e-16 |
| Muskingum-Cunge vs the analytic diffusion wave (Hayami kernel), 300 km, c = 2.2 m/s, D = 4000 m2/s | `analytic.json` | NSE **0.9999998**; peak error -0.04 %; same peak day |
| Kinematic wave vs the characteristic solution (rising limb with lateral inflow) | `analytic.json` | NSE 0.99976; error at t_e/2 0.7 %; equilibrium exact; mass 1e-15 |
| Degree-day snow vs a 5-day hand calculation | `analytic.json` | SWE and outflow exact (0.0) |
| Energy-balance snow terms vs hand arithmetic | `analytic.json` | net energy relative error 0.0; sublimation exact; mass 6e-15 mm |
| PET vs FAO-56 worked examples | `analytic.json` | Example 8: Ra 32.194 vs 32.2. Example 18 (Brussels): Ra 41.088 vs 41.09, Rn 13.283 vs 13.28, **ET0 3.88 vs 3.9 mm/day**. |
| Reservoir | `analytic.json` | A = dV/dh to 6e-11; mass balance 1e-11 hm3; hydropower = eta rho g H Q exactly; weir steady state spills the inflow, head within 5 % of hand (one hourly substep) |
| Reservoirs on B's 15 DEV dams (ResOpsUS) | `reservoir_dev.json` | power-law storage-elevation fits real pairs with median R2 0.9993 (b median 2.5, range 1.4-6; the generator uses this). Replaying observed inflow and release from each water year's start ends within a median 5 % of capacity (no evaporation; includes data inconsistencies). Generic policies on observed inflow, median release NSE: SOP 0.20, Hanasaki 0.27, fitted rule curve 0.33 (inflow-as-release -0.49). |
| Weather generator fitted to each dev basin (`fit_climate`, 1999-2008) vs that basin's real forcing | `weather_generator.json` | median generated/real ratio: mean P 1.00, wet-day frequency 0.99, wet-wet persistence 0.99, P99 1.06, annual-total CV 1.01, dry-spell length 1.00; T mean -0.3 C. Through the same calibrated HBV: mean Q 0.93, Q95 0.92, BFI 1.01, flashiness 1.00. |
| Speed | `speed.json` | see below |
| Generator realism vs the 52 dev basins' observed signatures | `generator.json` | see below |
| Dev calibration | `dev_calibration.json`, `dev_calibrated_params.json` | see below |
| Prior skill (uncalibrated transfer functions) | `prior_skill.json` | see below |
| Snow vs SNOTEL SWE, 33 of B's 39 DEV stations (6 lack data in the windows), station P and T; calibration WY2001-10, validation WY2011-25 (B's SNOTEL protocol) | `snotel.json` | **Degree-day:** uncalibrated NSE 0.70, peak SWE -26 %, melt-out off by 18 days (validation medians); calibrated per station (TT, CFMAX, SFCF), validation NSE **0.89** (10-90 %: 0.79-0.96), peak -5 %, melt-out 6 days. **Energy balance:** uncalibrated NSE 0.82; calibrated validation NSE **0.93**, peak -2 %, melt-out 5 days. Point-calibrated TT has median +1.3 C (10-90 %: -0.3 to 2.5). Only one dev station lies within 30 km of a dev basin. |

## Speed (one thread, shared 2-core Xeon 2.1 GHz; `speed.json`)

| case | time |
|---|---|
| numba core, 5 elevation bands, per basin-year | 0.08 ms (HBV, degree-day) to 0.27 ms (VIC-lite, energy balance) |
| full `simulate()` of a basin-decade, labels off | 11-15 ms |
| full `simulate()` of a basin-decade, labels on | 47-67 ms (about 5-7 x 10^9 x real time) |
| `simulate_batch`, 1,000 parameter sets x 10 years | 1.2 s HBV, 1.5 s GR4J, 3.9 s SAC-SMA, 3.1 s VIC-lite (1.2-3.9 ms per basin-decade) |
| `simulate_many`, 200 generated basins x 10 years | 1.2 s |
| `simulate()` with reservoirs, irrigation and Muskingum-Cunge, 50 basins | 6.1 s (120 ms each; the per-day reservoir policy loop is Python) |
| calibration (40 x 80 = 3,240 runs) | 3-7 s per basin-model |
| first call in a fresh environment | about 10-20 s of numba compilation, then cached on disk |

The target was >= 10,000x real time; the core runs at about 1-4 x 10^11 x.

## Calibrated benchmarks on the dev basins (`dev_calibration.json`)

- **Basins:** the 52 CAMELS-US dev basins from B's frozen splits.
- **Forcing:** Caravan ERA5-Land with Caravan's FAO-56 PM PET. Observed flow is Caravan Q (USGS).
- **Calibration:** 1999-10-01..2008-09-30 (B's calibration period), 1-year spin-up from the preceding forcing.
- **Validation:** two NON-exam windows, 1980-10..1989-09 ("early") and 2008-10..2014-09 ("late"). The CAMELS
  validation decade 1989-10..1999-09 is B's temporal exam and was never read. Those years were blanked in the
  observations before anything ran.

Median over the dev basins (KGE objective):

| model + snow | cal NSE | cal KGE | val-early NSE | val-early KGE | val-late NSE | val-late KGE |
|---|---|---|---|---|---|---|
| HBV + degree-day | 0.516 | 0.708 | 0.439 | 0.629 | 0.440 | 0.658 |
| HBV + energy balance | 0.500 | 0.705 | 0.453 | 0.632 | 0.426 | 0.659 |
| GR4J + degree-day | 0.520 | 0.719 | 0.466 | 0.618 | 0.456 | 0.645 |
| SAC-SMA + degree-day | 0.559 | 0.728 | 0.440 | 0.619 | 0.433 | 0.634 |
| VIC-lite + degree-day | 0.585 | 0.732 | 0.450 | 0.593 | 0.408 | 0.629 |

NSE objective (`dev_calibration_nse.json`, `dev_calibrated_params_nse.json`), for comparison with the
NSE-based literature:

| model + snow | cal NSE | cal KGE | val-early NSE | val-early KGE | val-late NSE | val-late KGE | share of basins with cal NSE > 0.5 |
|---|---|---|---|---|---|---|---|
| HBV + degree-day | 0.560 | 0.597 | 0.494 | 0.585 | 0.578 | 0.626 | 65 % |
| GR4J + degree-day | 0.557 | 0.612 | 0.516 | 0.583 | 0.572 | 0.616 | 65 % |
| SAC-SMA + degree-day | 0.620 | 0.635 | 0.489 | 0.589 | 0.524 | 0.617 | 73 % |
| VIC-lite + degree-day | 0.655 | 0.608 | 0.473 | 0.579 | 0.522 | 0.622 | 75 % |

The best classical number on non-exam dev validation is a median NSE of about 0.5-0.58 (NSE objective) or a
median KGE of about 0.63-0.66 (KGE objective).

These numbers are low next to the CAMELS literature. Kratzert et al. (2019) report an LSTM median NSE of about
0.74 and Newman et al. (2017) about 0.6 for SAC-SMA; both used Daymet/Maurer/NLDAS forcing and are context only.
The reasons we can see:

- **Forcing.** ERA5-Land is coarser and noisier for humid eastern rain basins (cal NSE about 0.4 there, 0.8-0.9
  in snowmelt basins).
- **Calendar days.** Caravan's days are UTC and USGS days are local. A +-1 day shift was tested: lag 0 is best.
- **Objective.** These are KGE calibrations. The NSE objective raises NSE by about 0.05-0.1 on the basins tried.
- **Arid basins.** Arid and intermittent basins (aridity > 1.3) give NSE near or below 0.

`dev_calibrated_params.json` has every best parameter set; pass one as `simulate(..., params=...)`.

**Uncalibrated attribute prior** (`prior_skill.json`, median NSE / KGE on the dev basins, calibration window;
the "ungauged" prior that `simulate` uses when a spec has no parameters):

| version | HBV | GR4J | SAC-SMA | VIC-lite |
|---|---|---|---|---|
| v0, no dev correction | 0.29 / 0.29 | 0.33 / 0.34 | 0.31 / 0.35 | 0.32 / 0.34 |
| v2, half correction (kept) | 0.31 / 0.48 | 0.37 / 0.48 | 0.37 / 0.49 | 0.36 / 0.45 |

The full correction (v1, `prior_skill_v0_v1.json`) raised KGE to about 0.5 but dropped NSE to 0.01-0.34:
combining per-parameter medians made the hydrograph too flashy. The medians are fitted on these same basins, so
the table is in-sample; on non-exam validation (2008-14) the kept prior gives NSE 0.30-0.37 and KGE 0.43-0.52.

**Generator realism** (`generator.json`, 250 + 250 simulated basins vs the 52 dev basins' observed 1999-2008
signatures; the dev correction is half of the median, as above):
- Mean KS distance over 12 signatures: **0.18** with the default prior and stochastic weather, **0.14** with the
  camels prior and real forcing (0.24 / 0.25 before the dev correction).
- **90-92 %** of real values fall inside the simulated 95 % ranges.
- Only **2-4 %** of real basins lie outside the simulated multivariate cloud (11-19 % before the correction).
- **Still off:**
  - simulated flashiness is a bit low (median 0.21-0.25 vs 0.31);
  - the share of real basins with near-zero q5 is under-represented (intermittent streams);
  - the default prior is somewhat too arid and too snowy.

## Using it in Phase 2

- **Rules** (B's `splits.json`):
  - Train only on simulations.
  - Real forcing may be reused from any basin, but only in non-exam windows (before 1989-10-01, and
    1999-10-01..2015-09-30). `realdata` enforces this.
  - Real attributes come only from dev basins (`prior='camels'` uses those 52).
  - SNOTEL dev stations and reservoir dev dams are free.
- **Recommended mix for training banks:**
  - prior: about half `'camels'` (real dev attributes and the donor's non-exam forcing) and half `'default'` with
    `generate_weather`, or `fit_climate` of a random basin's non-exam forcing;
  - models: all four, so a network does not learn one model's quirks;
  - snow: both routines;
  - observation noise: `obs_noise=True` (or `noise.streamflow_obs`) on the targets, and `noise.forcing_error`
    on the inputs.
- **Classical baselines** for the exam: `calibrate.calibrate` on an exam basin's own calibration period (the
  gauged protocol allows it) with the settings of `checks/calibrate_dev.py`. The dev numbers above show what to
  expect.
- **Twin tests:** for a dev basin, `simulate(wd.spec(id), (wd.forcing(id), wd.forcing_meta(id)), ...)` with
  `dev_calibrated_params.json` gives its calibrated twin. Without parameters, the attribute prior gives an
  "ungauged" twin.
- **ReservoirPilot:** use `reservoir.Reservoir.step()` inside your own loop (state, inflow, release request,
  evaporation), or `simulate(..., reservoirs=[{..., 'policy': my_policy}])` with a function of `obs`. The
  storage-elevation power law fits the dev dams with R2 0.999.
- **SnowSense:** use `simulate(model='none', snow=...)` for snow only, `return_bands=True` for per-band SWE,
  and `calibrate(..., target='SWE', model='none', obs_shift=1)` for the SNOTEL classical baseline.

## Known limitations (what Phase 2 should know)

1. **Lumped soil.** Snow is per elevation band but soil and groundwater are lumped per basin; there is no
   hillslope or aspect effect, and no glacier melt (Caravan `glacier_frac` is ignored).
2. **Snow parameters are scale-dependent.** At SNOTEL points the calibrated TT is about +1.3 C; in basin
   calibrations TT shifts the other way (-0.7 C from the prior). Uncalibrated defaults underestimate point peak
   SWE by about 20-25 %.
3. **Energy-balance snow is daily.**
   - The surface temperature is approximated by min(T, 0), and there is no diurnal melt-refreeze cycle.
   - Turbulent fluxes use a simple stability factor.
   - With B's ERA5-Land forcing SRAD is NET shortwave over ERA5's own surface. Incoming shortwave is
     reconstructed as SRAD / (1 - albedo_ERA5), with the albedo approximated from ERA5's SWE (0.2 bare, up to
     0.75) and kept within 0.15-1.0 x clear-sky. This is an approximation.
   - At sub-daily steps the EB model uses the same daily-type physics; use degree-day for hourly runs.
4. **GR4J is daily only.** Hourly GR4H uses different constants and is not implemented. Reservoirs run at a daily
   step only.
5. **Channel geometry** comes from regional power laws, not surveyed channels; channel slope is 5 % of
   HydroATLAS hillslope slope.
6. **Reservoir policies** are generic (SOP, rule curve, Hanasaki). There is no flood-control pool logic beyond
   the rule curve, and no multi-purpose optimisation or real operating rules. Evaporation is on top of the
   land-surface ET of the same area (precipitation on the lake is already in the runoff).
7. **Irrigation** is a simple deficit demand on a share of the basin. No groundwater pumping, no inter-basin
   transfers, no urban withdrawals.
8. **Generator.** Transfer functions are hand-written and only partly corrected on 52 dev basins. Real
   attributes for the camels prior come only from those 52 (B's rule), so attribute variety is limited. Use
   `prior='default'`, or mix both priors.
9. **Calibration.** The calibrated benchmarks above use ERA5-Land and are clearly below Daymet-based literature
   numbers. Phase 2 must compare against these baselines on the same forcing, not against literature values.
10. **No sub-daily validation.** Infiltration excess is first-order (constant capacity, no Green-Ampt
   ponding-time dynamics). The hourly mode is tested only for mass balance and plausibility; there is no hourly
   real-data check.
11. **numba cache.** numba caches compiled kernels on disk and does not track changes across modules.
    `import watersim` clears the cache automatically when any source file changed (a SHA-1 stamp); call
    `ws.clear_cache()` if in doubt.

## Files

```
watersim/
  watersim/            package: simulate.py engine.py snow.py pet.py routing.py reservoir.py models/{hbv,gr4j,
                       sacsma,vic_lite}.py spec.py forcing.py weather.py generator.py labels.py signatures.py
                       metrics.py calibrate.py realdata.py data/prior_adjust.json
  tools/fit_prior.py   dev-only prior correction (SHRINK=0.5)
  .refenv/             isolated R 4.4 + gfortran (micromamba, conda-forge) with airGR and TUWmodel (reference only)
  STATUS.md            resume notes
water/checks/          check_*.py, calibrate_dev.py, summarize_dev.py, run_all.sh, ref/ (R scripts + CSVs), *.json
```

## References

- Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998) FAO Irrigation and Drainage Paper 56.
- Bergström, S. (1976) Development and application of a conceptual runoff model for Scandinavian catchments. SMHI RHO 7.
- Brutsaert, W. (1975) On a derivable formula for long-wave radiation from clear skies. WRR 11.
- Brutsaert, W., Nieber, J.L. (1977) Regionalized drought flow hydrographs from a mature glaciated plateau. WRR 13.
- Burnash, R.J.C. (1995) The NWS River Forecast System - catchment modeling. In Computer Models of Watershed Hydrology.
- Chow, V.T., Maidment, D.R., Mays, L.W. (1988) Applied Hydrology. McGraw-Hill.
- Coron, L. et al. (2017) The suite of lumped GR hydrological models in an R package. EMS 94 (airGR).
- Cunge, J.A. (1969) On the subject of a flood propagation computation method (Muskingum method). J. Hydraul. Res. 7.
- Eckhardt, K. (2005) How to construct recursive digital filters for baseflow separation. Hydrol. Process. 19.
- Hanasaki, N., Kanae, S., Oki, T. (2006) A reservoir operation scheme for global river routing models. J. Hydrol. 327.
- Hargreaves, G.H., Samani, Z.A. (1985) Reference crop evapotranspiration from temperature. Appl. Eng. Agric. 1.
- Kratzert, F. et al. (2019) Towards learning universal, regional, and local hydrological behaviors. HESS 23.
- Kratzert, F. et al. (2023) Caravan - a global community dataset for large-sample hydrology. Sci. Data 10.
- Ladson, A.R. et al. (2013) A standard approach to baseflow separation using the Lyne and Hollick filter. AJWR 17.
- Liang, X., Lettenmaier, D.P., Wood, E.F., Burges, S.J. (1994) A simple hydrologically based model... (VIC). JGR 99.
- McCarthy, G.T. (1938) The unit hydrograph and flood routing. US Army Corps of Engineers.
- Nash, J.E. (1957) The form of the instantaneous unit hydrograph. IAHS Publ. 45.
- Newman, A.J. et al. (2015, 2017) CAMELS / benchmarking of a physically based hydrologic model. HESS 19; JHM 18.
- Oudin, L. et al. (2005) Which potential evapotranspiration input for a lumped rainfall-runoff model? J. Hydrol. 303.
- Parajka, J., Merz, R., Blöschl, G. (2007) Uncertainty and multiple objective calibration in regional water balance modelling (TUWmodel). Hydrol. Process. 21.
- Penman, H.L. (1948) Natural evaporation from open water, bare soil and grass. Proc. R. Soc. A 193.
- Perrin, C., Michel, C., Andréassian, V. (2003) Improvement of a parsimonious model for streamflow simulation. J. Hydrol. 279.
- Ponce, V.M., Yevjevich, V. (1978) Muskingum-Cunge method with variable parameters. J. Hydraul. Div. 104.
- Priestley, C.H.B., Taylor, R.J. (1972) On the assessment of surface heat flux and evaporation. MWR 100.
- Richardson, C.W. (1981) Stochastic simulation of daily precipitation, temperature and solar radiation. WRR 17.
- Seibert, J., Vis, M. (2012) Teaching hydrological modeling with a user-friendly catchment-runoff-model software package. HESS 16.
- Tarboton, D.G., Luce, C.H. (1996) Utah Energy Balance snow accumulation and melt model (UEB).
- Todini, E. (1996) The ARNO rainfall-runoff model. J. Hydrol. 175.
- Unsworth, M.H., Monteith, J.L. (1975) Long-wave radiation at the ground. QJRMS 101.
- Walter, M.T. et al. (2005) Process-based snowmelt modeling: does it require more input data than temperature-index modeling? J. Hydrol. 300.
- Zhang, J., Sanderson, A.C. (2009) JADE: adaptive differential evolution with optional external archive. IEEE TEC 13.
