"""Export TripDetective's live instrument for the website: float16 weights, the 13 real exam cars of the lab page
(VED exam set, fixed random draw within VED types, seed 3; up to 8 trips each, the first window of each trip), the
page's simulated lab (three RoadSim cars over two real Ann Arbor routes, precomputed logs), and reference outputs
for the browser parity test.

The trips and the lab logs come from the original page (lab/index.html, written by training/webdata.py and
training/build_lab.py); the raw training workspace is not in the repository. The reading pipeline in the browser
is the model's own net_input() + scalars(), ported; parity cases cover that too.

    python models/tripdetective/web/export_instrument.py      # writes site/public/instruments/tripdetective/
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import tripdetective as TD  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("tripdetective")
PAGE = os.path.join(MODEL, "lab", "index.html")
# integer scales of the page's compact trips (training/webdata.py Q); the browser decodes q / scale exactly like this
Q = dict(v=100, grade=1000, hw=100, fuel=1000, rpm=1000, batt=10, soc=1000, aux=1000)
ROW = dict(v=0, grade=1, hw=3, fuel=5, rpm=6, batt=7, soc=8, aux=9)
LABQ = dict(fuel=1000, rpm=1000, batt=10, soc=1000)
NTRIP = 8


def page_blobs():
    blobs = {}
    with open(PAGE, encoding="utf-8") as f:
        for line in f:
            for tag in ('id="D"', 'id="LAB"'):
                if line.startswith(f"<script {tag}"):
                    s = line[line.index(">") + 1: line.rindex("</script>")]
                    blobs[tag[4:-1]] = json.loads(s.replace("<\\/", "</"))
    return blobs["D"], blobs["LAB"]


class Int16Store:
    """Quantised integer channels, concatenated; written as bytes (u8) and read back as int16 in the browser."""

    def __init__(self):
        self.parts, self.n = [], 0

    def add(self, q):
        q = np.asarray(q, np.int64)
        assert np.abs(q).max(initial=0) < 32768
        off = self.n
        self.parts.append(q.astype("<i2"))
        self.n += len(q)
        return off

    def bytes(self):
        a = np.concatenate(self.parts) if self.parts else np.zeros(0, "<i2")
        return np.frombuffer(a.tobytes(), np.uint8)


# ---------------- the same input changes the browser makes (net.ts applyDrop / cut) ----------------
def apply(raw, fl, drop=(), cut=None):
    raw = raw.copy(); fl = np.array(fl, np.float32)
    if "fuel" in drop: raw[5] = 0; fl[0] = 0; fl[1] = 0
    if "rpm" in drop: raw[6] = 0; fl[2] = 0
    if "batt" in drop: raw[7] = 0; raw[8] = 0; fl[3] = 0; raw[9] = 0; fl[4] = 0   # A/C was only ever logged with the battery
    if "aux" in drop: raw[9] = 0; fl[4] = 0
    if cut: raw = raw[:, :cut]
    return raw, fl


def run_net(td, raws, fls):
    """The model's own encode() + heads() over every window of every trip (as read_car does)."""
    E = []
    for raw, fl in zip(raws, fls):
        for s, e in TD.windows(raw.shape[1]):
            E.append(td.encode(*TD.net_input(raw[:, s:e], fl, feat=td.feat)))
    h = td.heads(E)
    out = [h["arch"], h["strat"], h["mu"], h["sig"]]
    for mu, sg in h["led"]:
        out += [mu, sg]
    return np.concatenate([np.ravel(o) for o in out]).astype(float)


def main():
    td = TD.TripDetective()
    floats = {k: v for k, v in td.W.items() if k not in ("meta", "n_blocks", "down") and not k.endswith("_s")}
    nb = int(td.W["n_blocks"])
    meta = dict(td.meta, n_blocks=nb, strides=[int(td.W[f"c{i}_s"]) for i in range(nb)],
                params=int(sum(v.size for v in floats.values())))
    write_pack(OUT / "weights", floats, dtype="f16", meta=meta)
    td.W.update(roundtrip(floats, "f16"))  # the browser sees float16 weights; reference outputs use the same values

    D, LAB = page_blobs()
    store = Int16Store()
    valid_bytes = []
    nvalid = 0
    F60 = {r["vid"]: r["F60_N"] for r in D["exam"]["roadload"]["cars"]}

    # ---------------- real exam cars ----------------
    cars, raws_all, fls_all = [], [], []
    for c in D["cars"]:
        trips, raws, fls = [], [], []
        for t in c["trips"][:NTRIP]:
            n = t["n"]
            raw = np.zeros((len(TD.RAW), n), np.float64)
            ch = {}
            for k, i in ROW.items():
                q = np.asarray(t[k], np.int64)
                raw[i] = q / Q[k]
                ch[k] = store.add(q) if np.any(q) else None
            raw[2] = t["T"]; raw[4] = t["rho"]
            va = np.frombuffer(t["valid"].encode(), np.uint8) == ord("1")
            raw[10] = va
            valid_bytes.append(va.astype(np.uint8)); voff = nvalid; nvalid += n
            raw = raw.astype(np.float32)
            raws.append(raw); fls.append(np.array(t["flags"], np.float32))
            trips.append(dict(id=t["id"], n=n, km=t["km"], T=t["T"], rho=t["rho"], flags=t["flags"], ch=ch, valid=voff))
        cars.append(dict(vid=c["vid"], ved_type=c["ved_type"], arch=c["arch_rd"], arch_conf=c["arch_conf"],
                         weight_lb=c["weight_lb"], curb_kg=c["curb_kg"], disp=c["disp"], engine=c["engine"],
                         trans=c["trans"], likely=c["likely"], F60_N=F60.get(c["vid"]), trips=trips))
        raws_all.append(raws); fls_all.append(fls)

    # ---------------- simulated lab ----------------
    routes, rtens = [], {}
    for r, R in enumerate(LAB["routes"]):
        rr = np.zeros((5, R["n"]), np.float32)
        rr[0] = np.asarray(R["v"], np.float64)
        rr[1] = np.asarray(R["grade"], np.float64) * 100          # rise/run -> %, as raw_from_log
        rr[2] = np.asarray(R["T"], np.float64)
        rr[3] = np.asarray(R["hw"], np.float64)                   # headwind already along the heading
        rr[4] = np.asarray(R["rho"], np.float64)
        rtens[f"route{r}"] = rr
        routes.append(dict(id=R["id"], n=R["n"]))
    configs = []
    for cfg in LAB["configs"]:
        logs = []
        for lg in cfg["logs"]:
            logs.append({k: store.add(v) for k, v in lg.items()})
        configs.append(dict(arch=cfg["arch"], strat=cfg["strat"], dm=cfg["dm"], fc=cfg["fc"], mass=cfg["mass"],
                            CdA=cfg["CdA"], Crr=cfg["Crr"], disp=cfg["disp"], ledger=cfg["ledger"], logs=logs))

    def lab_raw(ci, r):
        cfg, R = LAB["configs"][ci], rtens[f"route{r}"]
        n = R.shape[1]
        raw = np.zeros((len(TD.RAW), n), np.float32)
        raw[0], raw[1], raw[2], raw[3], raw[4] = R
        for k, v in cfg["logs"][r].items():
            raw[ROW[k]] = np.asarray(v, np.float64) / LABQ[k]
        raw[10] = 1
        fl = np.array([0, 0, 0, 1, 0] if cfg["arch"] == "bev" else [1, 1, 1, 0, 0], np.float32)
        return raw, fl

    tensors = {"q16": store.bytes(), "valid": np.concatenate(valid_bytes)}
    tensors.update(rtens)
    ex_meta = dict(cars=cars, lab=dict(routes=routes, configs=configs, mass=LAB["mass"], cda=LAB["cda"], strat=LAB["strat"]),
                   scales=Q, labScales=LABQ)

    # ---------------- reference cases ----------------
    cases = []

    def exam_case(ci, variant, trips, drop=(), cut=None):
        rs, fs = zip(*[apply(raws_all[ci][t], fls_all[ci][t], drop, cut) for t in trips])
        cases.append(dict(example=ci, variant=variant, set="exam", trips=list(trips), drop=list(drop), cut=cut,
                          outputs=run_net(td, rs, fs).tolist()))

    for ci, c in enumerate(cars):
        nt = len(c["trips"])
        exam_case(ci, "one", [0])
        exam_case(ci, "pool3", [(1 + k) % nt for k in range(3)])
        typ = c["ved_type"]
        if typ == "ICE":
            exam_case(ci, "nofuel", [0], ("fuel",))
            exam_case(ci, "cut300", [2], (), 300)
        elif typ == "HEV":
            exam_case(ci, "norpm", [0], ("rpm",))
            exam_case(ci, "nofuel", [1], ("fuel",))
        elif typ == "PHEV":
            exam_case(ci, "nobatt", [0], ("batt",))
            exam_case(ci, "noaux", [1], ("aux",))
        else:
            exam_case(ci, "speedonly", [0], ("batt",))
    exam_case(0, "pool8", list(range(len(cars[0]["trips"]))))
    exam_case(len(cars) - 1, "pool8_cut240", list(range(len(cars[-1]["trips"]))), (), 240)
    for ci in (6, 47, 61, 121, 139):
        rs, fs = zip(*[lab_raw(ci, r) for r in (0, 1)])
        cfg = LAB["configs"][ci]
        cases.append(dict(example=ci, variant=f"lab_{cfg['arch']}_{cfg['strat']}_{cfg['dm']}_{cfg['fc']}", set="lab",
                          outputs=run_net(td, rs, fs).tolist()))

    ex_meta["python"] = {"note": "numpy model, float16 weights, all trips of each car pooled",
                         "cars": [_reading(td, raws_all[ci], fls_all[ci]) for ci in range(len(cars))]}
    write_pack(OUT / "examples", tensors, dtype="u8", keep={k: "f32" for k in rtens}, meta=ex_meta)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})

    size = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {len(cars)} cars, {sum(len(c['trips']) for c in cars)} trips, {len(configs)} lab configs, "
          f"{len(cases)} reference cases, {size / 1e6:.2f} MB")
    for ci, c in enumerate(cars):
        p = ex_meta["python"]["cars"][ci]
        print(f"  {c['vid']:8s} {c['ved_type']:4s} curb {p['curb_kg']:6.0f} kg (published {c['curb_kg']}), "
              f"engine {p['disp']:.2f} L (published {c['disp']}), type {max(p['type'], key=p['type'].get)}")


def _reading(td, raws, fls):
    E = [td.encode(*TD.net_input(r, f, feat=td.feat)) for r, f in zip(raws, fls)]
    h = td.heads(E)
    typ = {}
    for a, p in zip(td.meta["archs"], h["arch"]):
        typ[TD.VEDTYPE[a]] = typ.get(TD.VEDTYPE[a], 0) + float(p)
    return dict(curb_kg=float(1800 * np.exp(h["mu"][0]) - TD.CURB_OFFSET), disp=float(2.5 * np.exp(h["mu"][1])), type=typ)


if __name__ == "__main__":
    main()
