"""Random but physically consistent fleets and system states for training (Phase 2).

generate_fleet(seed, peak_MW, ...)  -> contract fleet DataFrame (thermal units with heat-rate curves, ramp, min up/down,
    start costs, inertia, governors; wind/solar/battery capacity), sized so firm capacity covers peak x reserve margin.
generate_states(n, seed, ...)       -> list of frequency system_state dicts + a labels DataFrame (inertia, IBR share,
    governor headroom, FFR, damping, largest unit), each a snapshot of a committed system at some demand level.

Ranges (all drawn uniformly or log-uniformly inside them unless stated):
  unit sizes per technology: EIA-860 (2019-2024) inter-quartile to 95th-percentile ranges for US units;
  heat rates: EIA-923 annual operating heat rates per technology (refdata/hr), median and spread;
  inertia constants H (s, own MVA): Kundur 1994 Table 3.2; ENTSO-E 2016 "Frequency stability evaluation criteria";
  NESO/ERCOT inertia reports: nuclear 4-6, coal 2.5-5, gas steam 3-5, CCGT 4-7 (whole block), OCGT 2.5-7,
  hydro 2-4, engines 0.5-2;
  governor share responsive, droop 3-6 %, deadband 0.015-0.036 Hz (GB/ERCOT), headroom fraction 5-20 %;
  committed spinning margin: 5-30 % of demand + 0.5-1.2 x the largest unit (operators cover the largest loss);
  load damping 0.5-4.5 %/% log-uniform (Kundur 1-2; GB 9 Aug 2019 fit 0.75; ERCOT dev-event effective fit 2.5-3.75);
  IBR share of generation 0-85 % (ERCOT 2025 peak instantaneous ~75 %; GB ~80 %),
  battery power 0-20 % of demand, FFR fraction of battery 0-100 %.
Resulting system inertia per MW of demand is checked against ERCOT (2019 minimum 134.6 GW s at 29.4 GW, i.e. 4.6 s)
and GB (9 Aug 2019: 210 GVA s at 29 GW, 7.2 s) in checks/generator.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fleet import HR_SHAPE, CO2_KG_PER_MMBTU, tech_default
from .frequency import make_system_state, SYSTEMS

# unit size ranges [MW] (p25, p95) by technology, EIA-860 operable US units 2019-2024 (summer capacity)
SIZE = {"nuclear": (1000, 1400), "coal_steam": (250, 900), "gas_cc": (250, 900), "gas_ct": (40, 230),
        "gas_steam": (80, 800), "gas_ice": (3, 20), "hydro_res": (20, 700), "oil": (2, 60), "biomass": (10, 60)}
# heat rate medians and relative spread (EIA-923 2017-2023 operating averages, MMBtu/MWh)
HR = {"nuclear": (10.45, 0.01), "coal_steam": (10.6, 0.07), "gas_cc": (7.6, 0.08), "gas_ct": (11.2, 0.12),
      "gas_steam": (11.5, 0.12), "gas_ice": (9.0, 0.06), "oil": (12.5, 0.15), "biomass": (13.5, 0.12)}
H_RANGE = {"nuclear": (4, 6), "coal_steam": (2.5, 5), "gas_steam": (3, 5), "gas_cc": (4, 7), "gas_ct": (2.5, 7),
           "hydro_res": (2, 4), "gas_ice": (0.5, 2), "oil": (2, 4), "biomass": (2.5, 4)}
GOV_SHARE = {"nuclear": 0.0, "coal_steam": 0.6, "gas_steam": 0.6, "gas_cc": 0.75, "gas_ct": 0.75, "hydro_res": 0.8,
             "gas_ice": 0.7, "oil": 0.5, "biomass": 0.4}
FUEL = {"nuclear": "NUC", "coal_steam": "SUB", "gas_cc": "NG", "gas_ct": "NG", "gas_steam": "NG", "gas_ice": "NG",
        "oil": "DFO", "biomass": "WDS", "hydro_res": "WAT"}


def _mix(rng):
    """Synchronous capacity shares (Dirichlet around US/GB BA mixes)."""
    techs = ["nuclear", "coal_steam", "gas_cc", "gas_ct", "gas_steam", "hydro_res", "gas_ice"]
    alpha = np.array([1.0, 1.5, 3.0, 1.2, 1.0, 0.8, 0.3])
    w = rng.dirichlet(alpha)
    return dict(zip(techs, w))


def generate_fleet(seed=0, peak_MW=None, reserve_margin=None, wind_frac=None, solar_frac=None, battery_frac=None,
                   region="ERCO"):
    """A random fleet in the generator contract (DataFrame).  Firm capacity = peak x (1 + margin)."""
    rng = np.random.default_rng(seed)
    peak = peak_MW or float(np.exp(rng.uniform(np.log(8000), np.log(90000))))
    rm = reserve_margin if reserve_margin is not None else rng.uniform(0.08, 0.30)
    mix = _mix(rng)
    rows = []
    for tech, share in mix.items():
        cap = share * peak * (1 + rm)
        lo, hi = SIZE[tech]
        while cap > 0.3 * lo:
            size = float(min(np.exp(rng.uniform(np.log(lo), np.log(hi))), max(cap, lo)))
            cap -= size
            hr_med, hr_sd = HR.get(tech, (0, 0))
            hr = hr_med * np.exp(rng.normal(0, hr_sd)) if hr_med else 0.0
            pmin_f = float(np.clip(tech_default(tech, "pmin_frac") * rng.uniform(0.7, 1.3), 0, 0.95))
            ramp_f = float(tech_default(tech, "ramp_frac_per_min") * np.exp(rng.normal(0, 0.4)))
            curve = None
            if tech in HR_SHAPE and hr:
                fr = np.array([a for a, _ in HR_SHAPE[tech]]); sh = np.array([b for _, b in HR_SHAPE[tech]])
                k = hr / np.interp(0.75, fr, sh)
                curve = [[float(f * size), float(b * k)] for f, b in HR_SHAPE[tech]]
            H = float(rng.uniform(*H_RANGE[tech]))
            fuel = FUEL[tech]
            fc = {"NUC": 0.65, "SUB": rng.uniform(1.4, 2.6), "NG": rng.uniform(2.0, 6.0), "DFO": rng.uniform(10, 25),
                  "WDS": rng.uniform(2, 4), "WAT": 0.0}[fuel]
            gov_on = rng.random() < GOV_SHARE.get(tech, 0.5)
            rows.append(dict(id=f"g{seed}_{len(rows)}", plant_id=None, synthetic=True, region=region, tech=tech,
                             pmax_MW=size, pmin_MW=pmin_f * size, ramp_up_MW_per_min=ramp_f * size,
                             ramp_down_MW_per_min=ramp_f * size,
                             min_up_h=float(round(tech_default(tech, "min_up_h") * rng.uniform(0.5, 1.5))),
                             min_down_h=float(round(tech_default(tech, "min_down_h") * rng.uniform(0.5, 1.5))),
                             start_cost_ul=None, heat_rate_curve=curve, fuel=fuel, **{"fuel_cost_$_per_MMBtu": fc},
                             co2_kg_per_MMBtu=CO2_KG_PER_MMBTU.get(fuel, 0.0), inertia_H_s=H,
                             mva_base=size / rng.uniform(0.85, 0.95),
                             governor=dict(droop=float(rng.uniform(0.03, 0.06)), deadband_Hz=None,
                                           T_s=None, headroom_frac=float(rng.uniform(0.05, 0.20)), enabled=gov_on),
                             ibr=False, ffr=None))
            rows[-1]["start_cost_$"] = tech_default(tech, "start_heat_MMBtu_per_MW") * size * max(fc, 2.0) * \
                rng.uniform(0.5, 1.5)
            rows[-1].pop("start_cost_ul")
    wf = wind_frac if wind_frac is not None else rng.uniform(0, 0.8)
    sf = solar_frac if solar_frac is not None else rng.uniform(0, 0.6)
    bf = battery_frac if battery_frac is not None else rng.uniform(0, 0.2)
    for tech, frac, n in (("wind", wf, 20), ("solar_pv", sf, 20), ("battery", bf, 10)):
        cap = frac * peak
        if cap < 1:
            continue
        for i, c in enumerate(rng.dirichlet(np.ones(n)) * cap):
            r = dict(id=f"g{seed}_{tech}_{i}", plant_id=None, synthetic=True, region=region, tech=tech, pmax_MW=float(c),
                     pmin_MW=0.0, ibr=True, inertia_H_s=0.0, fuel={"wind": "WND", "solar_pv": "SUN",
                                                                   "battery": "MWH"}[tech])
            if tech == "battery":
                r.update(energy_MWh=float(c * rng.choice([1, 2, 4])), eta_rt=float(rng.uniform(0.82, 0.9)),
                         soc_min=0.05, soc_max=0.95,
                         ffr=dict(MW=float(c * rng.uniform(0, 1)), trigger_Hz=None, delay_s=float(rng.uniform(0.1, 0.5))))
            if tech == "wind":
                r.update(hub_height_m=float(rng.uniform(70, 120)), specific_power=float(rng.uniform(220, 400)))
            if tech == "solar_pv":
                r.update(tracking=str(rng.choice(["single_axis", "fixed"])), dc_ac=float(rng.uniform(1.1, 1.45)))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.attrs.update(peak_MW=peak, reserve_margin=rm, seed=seed)
    return df


def generate_states(n=1000, seed=0, systems=("ERCO", "GB"), ibr_share=(0.0, 0.85), demand_frac=(0.35, 1.0),
                    loss=("largest", 0.3, 1.0), aggregate=False):
    """n random system snapshots ready for frequency_event (vectorised batch).

    Each: a random fleet (generate_fleet), a demand level, an instantaneous IBR share; synchronous units committed in
    random merit order until (demand - IBR - imports) + spinning reserve is covered; online units loaded between
    pmin and pmax.  Returns (states, losses_MW, labels DataFrame).

    aggregate: False (default, as released). True merges the units of each state per technology (aggregate_units)
    after the labels are computed: fleets of up to ~1,300 units otherwise pad a batched frequency_event; GridPulse's
    build measured ~40x faster batches (HANDOFF P0.6). Exact for the linear parts of the governor model; rate limits
    and per-unit headroom clips act on the group instead of each unit."""
    rng = np.random.default_rng(seed)
    states, losses, labels = [], [], []
    for i in range(n):
        sysname = systems[rng.integers(len(systems))]
        sysd = SYSTEMS[sysname]
        fl = generate_fleet(seed=int(rng.integers(1 << 30)), region=sysname)
        peak = fl.attrs["peak_MW"]
        demand = peak * rng.uniform(*demand_frac)
        ibr = rng.uniform(*ibr_share)
        sync_need = demand * (1 - ibr)
        th = fl[~fl.ibr.astype(bool)].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        # nuclear first (baseload), then random order biased by cost
        th = pd.concat([th[th.tech == "nuclear"], th[th.tech != "nuclear"]])
        # operators commit spinning capacity above the synchronous need: at least the largest unit, plus a margin
        spin = rng.uniform(0.05, 0.30) * demand + float(th.pmax_MW.max()) * rng.uniform(0.5, 1.2)
        units, cap, pmin_sum = [], 0.0, 0.0
        for _, u in th.iterrows():
            if cap >= sync_need + spin and pmin_sum > 0:
                break
            if pmin_sum + u.pmin_MW > sync_need:      # cannot commit below its minimum
                continue
            cap += u.pmax_MW; pmin_sum += u.pmin_MW
            units.append(u)
        if not units:
            continue
        # load online units: share of sync_need above pmin pro rata to range
        rngs = np.array([u.pmax_MW - u.pmin_MW for u in units])
        extra = max(sync_need - pmin_sum, 0.0)
        load = np.array([u.pmin_MW for u in units]) + rngs * min(extra / max(rngs.sum(), 1e-9), 1.0)
        db = float(rng.choice([0.015, 0.017, 0.036])) if sysname == "GB" else float(rng.choice([0.017, 0.036]))
        ulist = []
        for u, p in zip(units, load):
            g = dict(u.governor) if isinstance(u.governor, dict) else {}
            g["deadband_Hz"] = db
            ulist.append(dict(tech=u.tech, MW=float(p), pmax_MW=float(u.pmax_MW), pmin_MW=float(u.pmin_MW),
                              mva=float(u.mva_base), H_s=float(u.inertia_H_s), governor=g))
        bat = fl[fl.tech == "battery"]
        ffr = []
        bat_MW = float(bat.pmax_MW.sum()) * rng.uniform(0.2, 1.0)        # share online/available
        if bat_MW > 0:
            ffr_frac = rng.uniform(0, 1)
            trig = f0 = sysd["f0"]
            trig = (59.85 if f0 == 60 else 49.7) + rng.uniform(-0.05, 0.05)
            ffr.append(dict(MW=bat_MW * ffr_frac, trigger_Hz=float(trig), delay_s=float(rng.uniform(0.1, 0.5)),
                            ramp_s=float(rng.uniform(0.05, 0.5))))
            if rng.random() < 0.7:     # remainder as droop (primary frequency response) from batteries
                ulist.append(dict(tech="battery", MW=0.0, pmax_MW=bat_MW * (1 - ffr_frac), pmin_MW=-bat_MW,
                                  H_s=0.0, mva=bat_MW, governor=dict(enabled=True, droop=float(rng.uniform(0.01, 0.05)),
                                                                     deadband_Hz=db, up_MW=bat_MW * (1 - ffr_frac))))
        ibr_MW = demand * ibr
        ulist.append(dict(tech="wind", MW=float(ibr_MW), pmax_MW=float(ibr_MW), H_s=0.0, mva=float(ibr_MW)))
        si = []
        if rng.random() < 0.15:        # some grid-forming / synthetic inertia
            si.append(dict(H_s=float(rng.uniform(2, 6)), MVA=float(bat_MW + 0.1 * ibr_MW), MW_max=float(0.1 * bat_MW + 1)))
        D = float(np.exp(rng.uniform(np.log(0.5), np.log(4.5))))     # incl. the effective ERCOT fits (2.5-3.75)
        Hx = float(rng.uniform(0, 0.15)) * sum(u["H_s"] * u["mva"] for u in ulist)
        st = make_system_state(ulist, load_MW=demand, region=sysname, D=D, ffr=ffr, synthetic_inertia=si,
                               H_extra_MWs=Hx)
        big = max(u["MW"] for u in ulist if u["tech"] not in ("wind", "battery"))
        L = big * rng.uniform(loss[1], loss[2]) if loss[0] == "largest" else float(loss)
        states.append(st)
        losses.append(float(L))
        E = sum(u["H_s"] * u["mva"] for u in ulist) + Hx
        from .frequency import _compile
        P = _compile(st, sysd["f0"])
        labels.append(dict(system=sysname, f0=sysd["f0"], demand_MW=demand, ibr_share=ibr, inertia_MWs=E,
                           inertia_s_per_demand=E / demand, n_sync_units=len(units), loss_MW=L,
                           governor_headroom_MW=float(P["up"].sum()), ffr_MW=float(sum(f["MW"] for f in ffr)),
                           D=D, deadband_Hz=db, battery_MW=bat_MW, largest_unit_MW=big))
        if aggregate:
            from .frequency import aggregate_units
            st["units"] = aggregate_units(st["units"], by=("tech",), db=db, f0=sysd["f0"])
    return states, losses, pd.DataFrame(labels)
