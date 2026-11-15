"""Generator fleet: contract spec handling, technology defaults, fuel-input curves, fuel prices, clustering.

A fleet is a pandas DataFrame (or list of contract dicts) with one row per generator (contract fields; unknown = None).
`prepare_fleet` fills every missing field from TECH_DEFAULTS and records where each value came from in `src_fill`.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# reference tables (EIA-923 heat rates and fuel receipts, Henry Hub, MATPOWER cases): not in git; `mlratchet data grid`
# restores them. They live under the configurable grid root ($ML_RATCHET_GRID, default shared/grid).
REFDATA = os.path.join(os.environ.get("ML_RATCHET_GRID") or os.path.dirname(os.path.dirname(HERE)), "gridsim", "refdata")

THERMAL = ["nuclear", "coal_steam", "gas_cc", "gas_ct", "gas_steam", "gas_ice", "oil", "biomass", "geothermal", "other"]
VARIABLE = ["wind", "solar_pv", "hydro_ror"]
STORAGE = ["battery", "pumped"]
HYDRO = ["hydro_res"]
ALL_TECHS = THERMAL + VARIABLE + STORAGE + HYDRO

# EIA-930 fuel categories (`<BA>.NG.<FUEL>` columns)
EIA930_FUEL = {"nuclear": "NUC", "coal_steam": "COL", "gas_cc": "NG", "gas_ct": "NG", "gas_steam": "NG",
               "gas_ice": "NG", "oil": "OIL", "hydro_res": "WAT", "hydro_ror": "WAT", "pumped": "WAT",
               "wind": "WND", "solar_pv": "SUN", "battery": "OTH", "biomass": "OTH", "geothermal": "OTH",
               "other": "OTH"}

# kg CO2 per MMBtu (40 CFR 98 Subpart C Table C-1); biogenic fuels counted as 0 for power-sector CO2 like CAMPD/EIA.
CO2_KG_PER_MMBTU = {"NG": 53.06, "SUB": 97.17, "LIG": 97.72, "BIT": 93.28, "RC": 93.28, "WC": 93.28, "SGC": 93.28,
                    "PC": 102.41, "DFO": 73.96, "RFO": 75.10, "JF": 72.22, "KER": 75.20, "PG": 61.71, "OG": 59.00,
                    "BFG": 274.32, "LFG": 0.0, "OBG": 0.0, "WDS": 0.0, "AB": 0.0, "MSW": 41.69, "BLQ": 0.0, "TDF": 85.97,
                    "WH": 0.0, "PUR": 0.0, "OTH": 0.0, "SUN": 0, "WND": 0, "WAT": 0, "NUC": 0, "GEO": 0, "MWH": 0}

# Technology defaults.  Sources: RTS-GMLC gen.csv (Barrows et al. 2020) for ramp / min up-down / start heat by type;
# EIA Electric Power Annual 2019-2023 Table 8.1/8.2 for average heat rates; NREL ATB 2023 for VOM; NERC GADS
# 2017-2021 for forced-outage rates and planned-outage shares, replaced for coal, CC, CT and gas steam by ERCOT's own
# GADS statistics (Texas RE Assessment of Reliability Performance 2024/2025, Table A.4: coal EAF 69.5-76.9 %, gas steam
# 67-69 %, CC ~80 %, combustion turbines 84-88 %; forced share + scheduled share = 1 - EAF); inertia constants from
# Kundur (1994) Table 3.2
# and ENTSO-E/NESO inertia studies.  See README for the table.
TECH_DEFAULTS = {
    #               pmin   ramp %/min  up  down  start heat  HR full  VOM   fuel   price  EFOR  planned  H
    "nuclear":    (0.90, 0.005, 168, 48, 0.0, 10.45, 2.5, "NUC", 0.65, 0.02, 0.08, 5.0),
    "coal_steam": (0.40, 0.015, 8, 8, 18.0, 10.50, 4.5, "SUB", 1.95, 0.15, 0.11, 3.5),
    "gas_cc":     (0.40, 0.030, 4, 4, 6.0, 7.30, 2.5, "NG", 3.00, 0.08, 0.11, 5.0),
    "gas_ct":     (0.40, 0.100, 1, 1, 1.0, 11.00, 5.0, "NG", 3.00, 0.08, 0.06, 3.0),
    "gas_steam":  (0.30, 0.015, 6, 6, 15.0, 11.20, 4.0, "NG", 3.00, 0.13, 0.19, 3.5),
    "gas_ice":    (0.25, 0.200, 0, 0, 0.2, 8.90, 6.0, "NG", 3.00, 0.03, 0.03, 1.5),
    "oil":        (0.30, 0.100, 1, 1, 1.0, 12.00, 6.0, "DFO", 15.0, 0.08, 0.04, 2.0),
    "biomass":    (0.40, 0.015, 8, 8, 10.0, 13.50, 5.0, "WDS", 2.50, 0.10, 0.08, 3.0),
    "geothermal": (0.80, 0.010, 24, 24, 0.0, 0.00, 1.0, "GEO", 0.00, 0.05, 0.05, 3.0),
    "other":      (0.30, 0.020, 4, 4, 5.0, 10.50, 5.0, "OTH", 3.00, 0.08, 0.06, 3.0),
    "hydro_res":  (0.00, 1.000, 0, 0, 0.0, 0.00, 0.0, "WAT", 0.00, 0.03, 0.05, 3.0),
    "hydro_ror":  (0.00, 1.000, 0, 0, 0.0, 0.00, 0.0, "WAT", 0.00, 0.03, 0.05, 2.5),
    "pumped":     (0.00, 1.000, 0, 0, 0.0, 0.00, 0.0, "WAT", 0.00, 0.03, 0.05, 3.5),
    "wind":       (0.00, 1.000, 0, 0, 0.0, 0.00, 0.0, "WND", 0.00, 0.00, 0.00, 0.0),
    "solar_pv":   (0.00, 1.000, 0, 0, 0.0, 0.00, 0.0, "SUN", 0.00, 0.00, 0.00, 0.0),
    "battery":    (0.00, 1.000, 0, 0, 0.0, 0.00, 1.0, "MWH", 0.00, 0.02, 0.00, 0.0),
}
_DEF_KEYS = ["pmin_frac", "ramp_frac_per_min", "min_up_h", "min_down_h", "start_heat_MMBtu_per_MW", "hr_full",
             "vom", "fuel", "fuel_cost", "efor", "planned_frac", "H"]

# Outage defaults by physics version (the only columns that differ between versions).
#   1.0: the table above. Its coal, CC, CT and gas-steam rows come from Texas RE's 2024/2025 assessments (2024-2025
#        data, published during the exam years 2023-2025 of splits.json): a leak of exam-period information into
#        every dispatch default (shared/grid/HANDOFF.md P0.5).  Kept so v1.0 results stay reproducible.
#   1.1: the same four rows from sources published before 2023, describing years before 2023:
#        forced share = PJM "2017-2021 PJM Generating Unit Class Average Values" (created 20 May 2022), weighted EFORd:
#        fossil steam 13.593 % (coal and gas steam), combustion turbine 5.777 %, combined cycle 3.629 %;
#        planned share = Texas RE "2017 Assessment of Reliability Performance" (April 2018), Table 12: ERCOT
#        fleet scheduled outage factor 8.3 % (unweighted), the same for the four types (no per-type table there).
#        All other rows are unchanged.  EFORd is a demand-period rate, the 1.0 rows were time shares (1 - EAF - SOF):
#        the definitions differ slightly; both are used as a time share of capacity.
OUTAGE_DEFAULTS = {
    "1.0": {t: (TECH_DEFAULTS[t][9], TECH_DEFAULTS[t][10]) for t in ("coal_steam", "gas_cc", "gas_ct", "gas_steam")},
    "1.1": {"coal_steam": (0.136, 0.083), "gas_cc": (0.036, 0.083), "gas_ct": (0.058, 0.083),
            "gas_steam": (0.136, 0.083)},
}
# 1.2 (15 Nov 2026, opt-in): 1.1 plus real-time scarcity the day-ahead market cannot see (gridsim.market, used only by
# callers that ask for it). Every outage default, the dispatch and the frequency calibration are 1.1's.
OUTAGE_DEFAULTS["1.2"] = dict(OUTAGE_DEFAULTS["1.1"])


def calibration_path(version=None):
    """The dev-calibration file of a physics version: data/calibration.json (1.0), data/calibration_v1_1.json (1.1
    and 1.2: 1.2 changes no dispatch or frequency value; its RT scarcity parameters are in data/rt_scarcity_v1_2.json)."""
    v = version or physics_version()
    if v == "1.2":
        v = "1.1"
    return os.path.join(HERE, "data", "calibration.json" if v == "1.0" else f"calibration_v{v.replace('.', '_')}.json")


def calibration(version=None):
    p = calibration_path(version)
    return json.load(open(p)) if os.path.exists(p) else {}


def physics_version():
    """The GridSim physics version in use: $GRIDSIM_PHYSICS ("1.0", "1.1" or "1.2"), default "1.1"."""
    v = os.environ.get("GRIDSIM_PHYSICS", "1.1")
    if v not in OUTAGE_DEFAULTS:
        raise ValueError(f"GRIDSIM_PHYSICS={v!r}: known versions {sorted(OUTAGE_DEFAULTS)}")
    return v
# average-heat-rate shape vs load fraction (RTS-GMLC curves, normalised to full load)
HR_SHAPE = {"coal_steam": [(0.4, 1.089), (0.6, 1.014), (0.8, 0.995), (1.0, 1.0)],
            "gas_steam": [(0.3, 1.12), (0.6, 1.02), (0.8, 0.995), (1.0, 1.0)],
            "gas_cc": [(0.4, 1.09), (0.6, 1.03), (0.8, 1.005), (1.0, 1.0)],
            "gas_ct": [(0.4, 1.33), (0.6, 1.12), (0.8, 1.04), (1.0, 1.0)],
            "gas_ice": [(0.25, 1.20), (0.6, 1.06), (0.8, 1.02), (1.0, 1.0)],
            "oil": [(0.3, 1.25), (0.6, 1.06), (0.8, 1.0), (1.0, 1.0)],
            "biomass": [(0.4, 1.10), (0.7, 1.02), (1.0, 1.0)],
            "other": [(0.3, 1.15), (0.7, 1.03), (1.0, 1.0)],
            "nuclear": [(0.9, 1.0), (1.0, 1.0)]}


def tech_default(tech, key):
    if key in ("efor", "planned_frac") and tech in OUTAGE_DEFAULTS[physics_version()]:
        return OUTAGE_DEFAULTS[physics_version()][tech][0 if key == "efor" else 1]
    row = TECH_DEFAULTS.get(tech, TECH_DEFAULTS["other"])
    return row[_DEF_KEYS.index(key)]


def _parse(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def eia923_heat_rates(df, years):
    """Annual operating heat rate [MMBtu/MWh] per unit from EIA-923 (fuel for electricity / net generation) over
    `years`; combined-cycle parts are pooled per PUDL unit.  NaN where no usable record (CF < 3 % or HR outside 5-25)."""
    p = os.path.join(REFDATA, "hr", "unit_heat_rates.parquet")
    if not os.path.exists(p) or "plant_id" not in df:
        return np.full(len(df), np.nan)
    h = pd.read_parquet(p)
    h = h[h.year.isin(list(years))]
    h = h[(h.netgen_mwh > 0) & (h.fuel_mmbtu > 0)]
    gen = h.groupby(["plant_id", "generator_id"])[["fuel_mmbtu", "netgen_mwh"]].sum()
    unit = h.dropna(subset=["unit_id_pudl"]).groupby(["plant_id", "unit_id_pudl"])[["fuel_mmbtu", "netgen_mwh"]].sum()
    out = np.full(len(df), np.nan)
    ny = max(len(list(years)), 1)
    for i, r in enumerate(df.itertuples()):
        rec = None
        try:
            if r.tech == "gas_cc" and getattr(r, "unit_id_pudl", None) is not None and not pd.isna(r.unit_id_pudl):
                rec = unit.loc[(int(r.plant_id), float(r.unit_id_pudl))]
                cap = df.pmax_MW[(df.plant_id == r.plant_id) & (df.unit_id_pudl == r.unit_id_pudl)].sum()
            else:
                rec = gen.loc[(int(r.plant_id), str(r.generator_id))]
                cap = r.pmax_MW
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        cfac = rec.netgen_mwh / ny / (max(cap, 1e-6) * 8760)
        hr = rec.fuel_mmbtu / rec.netgen_mwh
        if cfac > 0.03 and 5.0 < hr < 25.0:
            out[i] = hr
    return out


def prepare_fleet(fleet, region=None, start=None, end=None, hr_source="eia923", hr_years=None):
    """Return a filled DataFrame of units online at any time in [start, end] (strings or Timestamps, optional).

    Missing fields come from TECH_DEFAULTS; `src_fill` lists which fields were defaulted.
    hr_source 'eia923': each fossil unit's heat-rate curve is the technology shape scaled so that the heat rate at
    75 % load equals its EIA-923 annual operating heat rate over `hr_years` (default: the 3 years up to `end`);
    'spec': keep the spec's curve.
    """
    if isinstance(fleet, str):
        fleet = pd.read_parquet(fleet) if fleet.endswith(".parquet") else pd.DataFrame(json.load(open(fleet)))
    df = pd.DataFrame(fleet).copy() if not isinstance(fleet, pd.DataFrame) else fleet.copy()
    if region is not None and "region" in df:
        df = df[df.region == region]
    if "id" not in df:
        df["id"] = [f"u{i}" for i in range(len(df))]
    df["online"] = pd.to_datetime(df.get("online_date"), errors="coerce") if "online_date" in df else pd.NaT
    df["retired"] = pd.to_datetime(df.get("retired_date"), errors="coerce") if "retired_date" in df else pd.NaT
    if start is not None:
        df = df[df.retired.isna() | (df.retired >= pd.Timestamp(start).tz_localize(None))]
    if end is not None:
        df = df[df.online.isna() | (df.online <= pd.Timestamp(end).tz_localize(None))]
    df = df.reset_index(drop=True)
    fills = [[] for _ in range(len(df))]

    def fill(col, values_fn):
        if col not in df:
            df[col] = np.nan
        vals = df[col].astype(object).to_numpy(copy=True)
        for i in range(len(df)):
            v = vals[i]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals[i] = values_fn(i)
                fills[i].append(col)
        df[col] = vals

    techs = df["tech"].values
    pmax = pd.to_numeric(df["pmax_MW"], errors="coerce").values
    fill("pmin_MW", lambda i: tech_default(techs[i], "pmin_frac") * pmax[i])
    fill("ramp_up_MW_per_min", lambda i: tech_default(techs[i], "ramp_frac_per_min") * pmax[i])
    fill("ramp_down_MW_per_min", lambda i: tech_default(techs[i], "ramp_frac_per_min") * pmax[i])
    fill("min_up_h", lambda i: tech_default(techs[i], "min_up_h"))
    fill("min_down_h", lambda i: tech_default(techs[i], "min_down_h"))
    fill("fuel", lambda i: tech_default(techs[i], "fuel"))
    fill("fuel_cost_$_per_MMBtu", lambda i: tech_default(techs[i], "fuel_cost"))
    fuels = df["fuel"].values
    fill("co2_kg_per_MMBtu", lambda i: CO2_KG_PER_MMBTU.get(str(fuels[i]), 0.0))
    fill("vom_$_per_MWh", lambda i: tech_default(techs[i], "vom"))
    fill("inertia_H_s", lambda i: tech_default(techs[i], "H"))
    fill("mva_base", lambda i: pmax[i] / 0.9)
    fill("efor", lambda i: tech_default(techs[i], "efor"))
    fill("planned_frac", lambda i: tech_default(techs[i], "planned_frac"))
    fc = pd.to_numeric(df["fuel_cost_$_per_MMBtu"], errors="coerce").values
    fill("start_cost_$", lambda i: tech_default(techs[i], "start_heat_MMBtu_per_MW") * pmax[i] * max(fc[i], 2.0))
    # heat-rate curve: parse or default (shape x full-load HR; full-load HR from the EIA-923 average if given)
    hrs = []
    hr923 = pd.to_numeric(df.get("hr_eia923_2018_2022_MMBtu_per_MWh", pd.Series(np.nan, index=df.index)),
                          errors="coerce").values
    for i in range(len(df)):
        c = _parse(df["heat_rate_curve"].iloc[i]) if "heat_rate_curve" in df else None
        t = techs[i]
        if isinstance(c, list) and len(c) >= 2:
            hrs.append([[float(a), float(b)] for a, b in c])
        elif t in HR_SHAPE and tech_default(t, "hr_full") > 0:
            full = hr923[i] if np.isfinite(hr923[i]) and 5 < hr923[i] < 25 else tech_default(t, "hr_full")
            # EIA-923 averages include part-load operation; divide by a typical part-load penalty to get full-load HR
            hrs.append([[f * pmax[i], s * full] for f, s in HR_SHAPE[t]])
            fills[i].append("heat_rate_curve")
        else:
            hrs.append(None)
    df["heat_rate_curve"] = hrs
    df["hr_avg_eia923"] = np.nan
    if hr_source == "eia923":
        # without an end date: the three years up to 2023 in v1.0 (an exam year), up to 2022 in v1.1
        yend = pd.Timestamp(end).year if end is not None else (2023 if physics_version() == "1.0" else 2022)
        yrs = hr_years or range(yend - 2, yend + 1)
        hr_avg = eia923_heat_rates(df, yrs)
        df["hr_avg_eia923"] = hr_avg
        for i in np.where(np.isfinite(hr_avg))[0]:
            t = techs[i]
            if t in HR_SHAPE and t != "nuclear":
                fr = np.array([a for a, _ in HR_SHAPE[t]]); sh = np.array([b for _, b in HR_SHAPE[t]])
                scale = hr_avg[i] / np.interp(0.75, fr, sh)
                hrs[i] = [[f * pmax[i], b * scale] for f, b in HR_SHAPE[t]]
                fills[i] = [x for x in fills[i] if x != "heat_rate_curve"] + ["heat_rate_curve<-eia923"]
        df["heat_rate_curve"] = hrs
    for k in ("governor", "ffr"):
        if k in df:
            df[k] = [_parse(v) for v in df[k]]
    df["src_fill"] = fills
    for c in ["pmax_MW", "pmin_MW", "ramp_up_MW_per_min", "ramp_down_MW_per_min", "min_up_h", "min_down_h",
              "fuel_cost_$_per_MMBtu", "co2_kg_per_MMBtu", "vom_$_per_MWh", "inertia_H_s", "mva_base", "efor",
              "planned_frac", "start_cost_$"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["pmin_MW"] = np.minimum(df["pmin_MW"], df["pmax_MW"])
    return df


def merge_combined_cycles(df):
    """Merge combined-cycle parts (cc_part CT/CA/CS) of one plant + PUDL unit into one block (pmax and pmin summed,
    heat-rate curve = capacity-weighted mean of the parts' curves with EIA-923 heat rates, else tech default)."""
    if "cc_part" not in df or "unit_id_pudl" not in df:
        return df
    is_cc = (df.tech == "gas_cc") & df.cc_part.notna()
    key = df.plant_id.astype(str) + "_" + df.unit_id_pudl.fillna(-1).astype(int).astype(str)
    key = key.where(df.unit_id_pudl.notna(), df.plant_id.astype(str) + "_p")
    out = [df[~is_cc]]
    rows = []
    for k, g in df[is_cc].groupby(key[is_cc]):
        r = g.iloc[0].copy()
        r["id"] = "cc_" + k
        pmax = g.pmax_MW.sum()
        r["pmax_MW"] = pmax
        r["pmin_MW"] = min(g.pmin_MW.sum(), 0.6 * pmax)
        for c in ["ramp_up_MW_per_min", "ramp_down_MW_per_min", "start_cost_$", "mva_base"]:
            r[c] = g[c].sum()
        r["inertia_H_s"] = float((g.inertia_H_s * g.mva_base).sum() / g.mva_base.sum())
        r["min_up_h"], r["min_down_h"] = g.min_up_h.max(), g.min_down_h.max()
        # block heat rate: fuel is burnt only in the CTs (+ duct burners); EIA-923 splits it over parts, so use the
        # generation-weighted mean of the parts' full-load average heat rates that look like CC values (5.5-11)
        hu = g.hr_avg_eia923.dropna() if "hr_avg_eia923" in g else pd.Series(dtype=float)
        if len(hu):          # pooled unit heat rate (same for all parts)
            full = float(hu.iloc[0]) / np.interp(0.75, [a for a, _ in HR_SHAPE["gas_cc"]], [b for _, b in HR_SHAPE["gas_cc"]])
        else:
            hr = pd.to_numeric(g.get("hr_eia923_2018_2022_MMBtu_per_MWh"), errors="coerce")
            ok = hr.between(5.5, 11.0)
            full = float((hr[ok] * g.pmax_MW[ok]).sum() / g.pmax_MW[ok].sum()) if ok.any() else tech_default("gas_cc", "hr_full")
        r["heat_rate_curve"] = [[f * pmax, s * full] for f, s in HR_SHAPE["gas_cc"]]
        r["online"] = g.online.max() if g.online.notna().any() else pd.NaT       # block works when all parts exist
        r["retired"] = g.retired.min() if g.retired.notna().any() else pd.NaT
        r["n_parts"] = len(g)
        r["cc_part"] = "block"
        rows.append(r)
    if rows:
        out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True)


def fuel_curve(curve, pmax, pmin, n_seg=3):
    """Convex fuel-input model from an average-heat-rate curve [[MW, MMBtu/MWh], ...].

    Fits F(P) = a + b P + c P^2 (c >= 0, a >= 0) to F = P * HR(P), then returns
    no-load fuel a [MMBtu/h], fuel at pmin, segment widths [MW] and incremental heat rates [MMBtu/MWh] on n_seg
    equal segments from pmin to pmax (non-decreasing -> convex).
    """
    if curve is None or pmax <= 0:
        return dict(F0=0.0, Fmin=0.0, widths=np.full(n_seg, max(pmax - pmin, 0) / n_seg), ihr=np.zeros(n_seg))
    P = np.array([c[0] for c in curve], float)
    F = P * np.array([c[1] for c in curve], float)
    if len(P) >= 3 and np.ptp(P) > 0:
        A = np.vstack([np.ones_like(P), P, P ** 2]).T
        coef, *_ = np.linalg.lstsq(A, F, rcond=None)
        a, b, c = coef
        if c < 0 or a < 0:
            A2 = np.vstack([np.ones_like(P), P]).T
            (a, b), *_ = np.linalg.lstsq(A2, F, rcond=None)
            c = 0.0
            if a < 0:
                a, b = 0.0, float(F.sum() / P.sum())
    elif len(P) >= 2 and np.ptp(P) > 0:
        b = (F[-1] - F[0]) / (P[-1] - P[0]); a = max(F[0] - b * P[0], 0.0); c = 0.0
    else:
        a, b, c = 0.0, float(F[-1] / max(P[-1], 1e-9)), 0.0
    edges = np.linspace(pmin, pmax, n_seg + 1)
    Fe = a + b * edges + c * edges ** 2
    widths = np.diff(edges)
    ihr = np.where(widths > 0, np.diff(Fe) / np.where(widths > 0, widths, 1), b)
    ihr = np.maximum.accumulate(np.maximum(ihr, 0))
    return dict(F0=float(a), Fmin=float(a + b * pmin + c * pmin ** 2), widths=widths, ihr=ihr)


# ---------------------------------------------------------------------- fuel prices
_HH = None


def henry_hub_daily():
    global _HH
    if _HH is None:
        p = os.path.join(REFDATA, "fuel", "henry_hub_daily.csv")
        s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["hh_usd_per_MMBtu"]
        _HH = s.asfreq("D").ffill()
    return _HH


def fuel_prices_daily(units, start, end, mode="eia923"):
    """Daily fuel price per unit [$/MMBtu] for days in [start, end] -> DataFrame (days x unit index).

    mode 'eia923': gas = the plant's (else state's) monthly delivered cost from EIA-923 receipts, plus the daily
    Henry Hub deviation from its monthly mean; coal/oil = plant (else state) monthly delivered cost; others = the
    spec's constant.  mode 'constant': the spec's fuel_cost_$_per_MMBtu for every day.
    """
    days = pd.date_range(pd.Timestamp(start).tz_localize(None).normalize(),
                         pd.Timestamp(end).tz_localize(None).normalize(), freq="D")
    base = units["fuel_cost_$_per_MMBtu"].fillna(0).values
    out = np.tile(base, (len(days), 1))
    if mode == "constant":
        return pd.DataFrame(out, index=days, columns=units.index)
    p = os.path.join(REFDATA, "fuel", "out_eia923__monthly_fuel_receipts_costs.parquet")
    rc = pd.read_parquet(p, columns=["report_date", "plant_id_eia", "state", "fuel_type_code_pudl",
                                     "fuel_cost_per_mmbtu", "fuel_received_mmbtu"])
    rc["report_date"] = pd.to_datetime(rc.report_date)
    rc = rc[(rc.report_date >= days[0] - pd.Timedelta(days=400)) & (rc.report_date <= days[-1])]
    rc = rc[rc.fuel_cost_per_mmbtu.between(0.05, 60)]
    rc["w"] = rc.fuel_received_mmbtu.clip(lower=1)
    rc["wc"] = rc.w * rc.fuel_cost_per_mmbtu
    g_pl = rc.groupby(["plant_id_eia", "fuel_type_code_pudl", "report_date"])[["wc", "w"]].sum()
    pl = (g_pl.wc / g_pl.w)
    states = units.get("state", pd.Series("TX", index=units.index)).fillna("TX")
    g_st = rc.groupby(["state", "fuel_type_code_pudl", "report_date"])[["wc", "w"]].sum()
    stp = (g_st.wc / g_st.w)
    months = days.to_period("M").to_timestamp()
    hh = henry_hub_daily().reindex(days).ffill().bfill()
    hh_m = hh.groupby(months).transform("mean")
    group = {"NG": "gas", "SUB": "coal", "LIG": "coal", "BIT": "coal", "RC": "coal", "WC": "coal", "PC": "coal",
             "DFO": "oil", "RFO": "oil", "JF": "oil", "KER": "oil", "OG": "gas", "PG": "gas"}
    um = pd.DatetimeIndex(sorted(set(months)))
    for j, (idx, u) in enumerate(units.iterrows()):
        fg = group.get(str(u.get("fuel")))
        if fg is None:
            continue
        ser = None
        try:
            s = pl.loc[(int(u.plant_id), fg)]
            if (s.index >= um[0] - pd.Timedelta(days=60)).sum() >= 1:
                ser = s
        except (KeyError, TypeError, ValueError):
            pass
        if ser is None:
            try:
                ser = stp.loc[(states.loc[idx], fg)]
            except KeyError:
                continue
        ser = ser.sort_index()
        full = pd.Series(ser.values, index=ser.index).reindex(um.union(ser.index)).sort_index().ffill().bfill()
        mval = full.reindex(months).values
        if fg == "gas":
            mval = mval + (hh.values - hh_m.values)
        out[:, j] = np.maximum(mval, 0.05)
    return pd.DataFrame(out, index=days, columns=units.index)


# ---------------------------------------------------------------------- clustering
def cluster_fleet(units, n_clusters=None, max_units_exact=1, seed=0):
    """Group thermal units into clusters of similar units (same tech, similar full-load heat rate and size).

    n_clusters: dict tech -> number of clusters (default: coal/nuclear per unit, gas_cc 24, gas_steam 10, gas_ct 10,
    gas_ice 3, oil 2, biomass/other/geothermal 2).  Returns the units frame with a `cluster` column.
    """
    n_def = {"nuclear": None, "coal_steam": None, "gas_cc": 24, "gas_steam": 10, "gas_ct": 10, "gas_ice": 3,
             "oil": 2, "biomass": 2, "other": 2, "geothermal": 2}
    if n_clusters:
        n_def.update(n_clusters)
    units = units.copy()
    units["cluster"] = -1
    cid = 0
    for tech in THERMAL:
        m = np.where(units.tech.values == tech)[0]
        if len(m) == 0:
            continue
        k = n_def.get(tech)
        if k is None or k >= len(m):
            units.loc[units.index[m], "cluster"] = np.arange(cid, cid + len(m))
            cid += len(m)
            continue
        hr = np.array([c[-1][1] if c else 10.0 for c in units.heat_rate_curve.values[m]])
        cost = hr * units["fuel_cost_$_per_MMBtu"].values[m] + units["vom_$_per_MWh"].values[m]
        size = np.log(units.pmax_MW.values[m])
        # split by size (two classes when sizes differ a lot), then by cost quantiles within each class
        big = size > np.median(size) if np.ptp(size) > np.log(3) and len(m) > 4 else np.zeros(len(m), bool)
        classes = [np.where(~big)[0], np.where(big)[0]]
        kk = [max(1, int(round(k * len(c) / len(m)))) for c in classes]
        for c, kc in zip(classes, kk):
            if len(c) == 0:
                continue
            order = c[np.argsort(cost[c])]
            for part in np.array_split(order, min(kc, len(order))):
                units.loc[units.index[m[part]], "cluster"] = cid
                cid += 1
    return units
