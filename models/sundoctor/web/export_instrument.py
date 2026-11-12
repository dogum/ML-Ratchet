"""Export SunDoctor's live instrument for the website: float16 weights, real fortnights from the two exam farms exactly as
the reading pipeline grids them (solar time, per kW of nameplate), and reference outputs for the browser parity test.

    python models/sundoctor/web/export_instrument.py      # writes site/public/instruments/sundoctor/

Everything the network sees is rebuilt in the browser by `channels()` (ported to net.ts) from the arrays exported here:
power P, irradiance G, air temperature Ta (the pipeline's solar grid of the examples/ CSVs) and the clear-sky geometry
cz, ca, gclear (sundoctor.clearsky_poa). The farm's healthy reference level (`ref`, from the full record, causal) is
exported per window and mode. PU is the power the array would have made without its AC limit (training/inject.py
`unclipped`), so a string can be killed under the clipping ceiling faithfully.

Answer keys come from the model's own exam (lab/index.html DATA): sibling inverters count the dead strings in California;
the plant's own instruments key snow, stuck trackers and outages in Colorado.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import sundoctor as S  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("sundoctor")
D = 14

FARMS = {
    "ca": dict(csv="pvdaq2107_inverter11_2019H2.csv", label="California", system="PVDAQ 2107 · inverter 11",
               kw=dict(lat=38.996, lon=-122.134, tilt=25, azimuth=180, dc_kw=37.2, tz="US/Pacific"),
               mount="fixed", n=6, unit="string", ceiling=30.0 / 37.2, ceiling_note="30 kW inverter rating"),
    "co": dict(csv="pvdaq9068_module2_2018winter.csv", label="Colorado", system="PVDAQ 9068 · module 2",
               kw=dict(lat=40.3864, lon=-104.5512, mount="tracker", max_angle=60, gcr=0.35, backtrack=False, dc_kw=592.25, tz="US/Mountain"),
               mount="tracker", n=3, unit="combiner", ceiling=None, ceiling_note="plateau in the record"),
}

# the fortnights: chosen to hold the known events (first index day is the window's first solar day)
WINDOWS = [
    ("ca", "2019-09-20", "A string dies", "one of six strings fails on 27 Sep 2019"),
    ("ca", "2019-10-09", "…and is repaired", "a crew restores it on 15–16 Oct"),
    ("ca", "2019-07-08", "Summer", "bright, hot, clipping at noon"),
    ("ca", "2019-11-15", "Low winter sun", "rows start to shade each other"),
    ("co", "2018-01-14", "Snowstorm", "snow buries the field on 21 Jan 2018"),
    ("co", "2018-02-05", "Snow, then frozen trackers", "trackers freeze in place 11–14 Feb"),
    ("co", "2018-02-11", "Late-February storm", "snow on 20–21 Feb"),
]


def unclipped(p, pe, clip):
    """training/inject.py unclipped(): clipped steps from the physics prior scaled by the day's measured/expected ratio"""
    r = np.where((p < 0.95 * clip) & (pe > 0.1), p / np.maximum(pe, 1e-3), np.nan)
    with np.errstate(all="ignore"):
        k = np.nanmedian(r, 1, keepdims=True)
    return np.where(p >= 0.95 * clip, np.maximum(p, pe * np.nan_to_num(k, nan=1.0)), p)


def inject(P, PU, n, native, ceil=None, kill=None, snow=None):
    """the instrument's faults, exactly as ui.ts/net.ts apply them (see net.ts inject())"""
    X = P.copy()
    if kill is not None:
        X[kill:] = np.minimum(PU[kill:] * ((n - 1) / n), native)
    if ceil is not None:
        X = np.minimum(X, ceil)
    if snow is not None:
        X[snow] = np.where(np.isfinite(X[snow]), 0.0, X[snow])
        if snow + 1 < len(X):
            X[snow + 1, :48] = np.where(np.isfinite(X[snow + 1, :48]), 0.0, X[snow + 1, :48])
    return X


def lab_keys():
    """answer keys from the model's own exam page (lab/index.html, const DATA)"""
    line = next(l for l in open(os.path.join(MODEL, "lab", "index.html"), encoding="utf-8") if l.startswith("const DATA = "))
    data = json.loads(line[line.index("{"): line.rindex("}") + 1])
    keys = {"ca": {}, "co": {}}
    for c in data["cases"]:
        if c["inverter"] == 11:
            for d, k in zip(c["dates"], c["k_true"]):
                keys["ca"].setdefault(d, {})["dead"] = k
    for c in data["co"]:
        if c["module"] == 2:
            for i, d in enumerate(c["dates"]):
                keys["co"][d] = {k: c["key"][k][i] for k in ("snow", "stuck", "outage")}
    return keys


def pipeline(doc, farm, home):
    """run the doctor's own diagnose_csv and capture what it hands the network: the solar grid and the reference level"""
    cap = []
    orig = doc.run
    doc.run = lambda *a, **k: (cap.append((a, k)), orig(*a, **k))[1]
    f = FARMS[farm]
    r = doc.diagnose_csv(os.path.join(MODEL, "examples", f["csv"]), home=home, **f["kw"])
    doc.run = orig
    (P, G, Ta, lat, tilt, az, doy, ref, _home, mount), trk = cap[-1]
    cz, ca, gc = S.clearsky_poa(lat, tilt, az, doy[:, None], mount=mount, **trk)
    return dict(P=P, G=G, Ta=Ta, cz=cz, ca=ca, gc=gc, ref=ref, dates=[str(d) for d in r["date"]], out=r)


def forward(net, x):
    day, g = net.forward(x[None])
    return np.concatenate([day[0].ravel(), g[0]])


def main():
    doc = S.SunDoctor()
    net = doc.net
    W = dict(net.W)
    net.W = roundtrip(W, "f16")  # the browser sees float16 weights; reference outputs use the same values
    write_pack(OUT / "weights", W, dtype="f16", meta=net.meta)

    keys = lab_keys()
    runs = {(farm, home): pipeline(doc, farm, home) for farm in FARMS for home in (False, True)}
    natives = {}
    for farm, f in FARMS.items():
        P = runs[(farm, False)]["P"]
        natives[farm] = f["ceiling"] if f["ceiling"] else float(np.nanmax(P))

    tens, meta, cases = {}, [], []
    f32 = lambda a: np.asarray(a, np.float32).astype(np.float64)  # what the browser sees
    for i, (farm, start, name, desc) in enumerate(WINDOWS):
        R, Rh, f = runs[(farm, False)], runs[(farm, True)], FARMS[farm]
        a = R["dates"].index(start)
        sl = slice(a, a + D)
        P, G, Ta, cz, ca, gc = (f32(R[k][sl]) for k in ("P", "G", "Ta", "cz", "ca", "gc"))
        t0 = np.where(np.isfinite(Ta), Ta, 15.0)
        pe = np.where(np.isfinite(G), S.expected_power(np.maximum(np.nan_to_num(G), 0), t0, ca), S.expected_power(gc, t0, ca))
        PU = f32(unclipped(P, pe, natives[farm]))
        for k, v in (("p", P), ("pu", PU), ("g", G), ("ta", Ta), ("cz", cz), ("ca", ca), ("gc", gc)):
            tens[f"{k}{i}"] = v
        ref = {"poa": R["ref"][a + D // 2], "home": Rh["ref"][a + D // 2]}
        ref = {k: (float(v) if np.isfinite(v) else None) for k, v in ref.items()}
        dates = R["dates"][sl]
        kk = keys[farm]
        key = {c: [kk.get(d, {}).get(c) for d in dates] for c in (("dead",) if farm == "ca" else ("snow", "stuck", "outage"))}
        meta.append({
            "farm": farm, "label": f["label"], "system": f["system"], "name": name, "desc": desc, "dates": dates,
            "dc_kw": f["kw"]["dc_kw"], "mount": f["mount"], "n": f["n"], "unit": f["unit"], "ceiling": natives[farm],
            "ceiling_note": f["ceiling_note"], "ref": ref, "key": key,
        })

        def case(variant, home=False, **inj):
            x = S.channels(inject(P, PU, f["n"], natives[farm], **inj), G, Ta, cz, ca, gc,
                           ref["home" if home else "poa"], home)
            return {"example": i, "variant": variant, "home": home, "inj": inj, "outputs": forward(net, x).tolist()}

        cases.append(case("raw"))
        cases.append(case("home", home=True))
        if i in (0, 4):  # deterministic perturbations: a lower inverter ceiling, a string lost on day 5, snow on day 3
            cases.append(case("clip0.55", ceil=0.55))
            cases.append(case("kill5", kill=5))
            cases.append(case("snow3", snow=3))
            cases.append(case("all-home", home=True, ceil=0.6, kill=8, snow=2))

    write_pack(OUT / "examples", tens, dtype="f32", meta=meta)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    size = sum(os.path.getsize(OUT / n) for n in os.listdir(OUT))
    print(f"wrote {OUT}: {len(meta)} fortnights, {len(cases)} parity cases, {size / 1e6:.2f} MB in all")


if __name__ == "__main__":
    main()
