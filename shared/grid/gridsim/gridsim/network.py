"""Transmission network: MATPOWER case parsing, DC and AC (Newton-Raphson) power flow, PTDF, DC-OPF.

Conventions follow MATPOWER (Zimmerman et al. 2011) so results can be checked against pandapower/PYPOWER:
per-unit on `baseMVA`, angles in degrees at the interface, branch model = pi model with an ideal
transformer (ratio, phase shift) at the from end.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# MATPOWER column indices (0-based)
BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN = range(13)
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX = range(13)
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)
PQ, PV, REF, NONE = 1, 2, 3, 4


def _parse_matrix(text: str, name: str):
    m = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % re.escape(name), text, re.S)
    if not m:
        return None
    rows = []
    for line in m.group(1).split("\n"):
        line = line.split("%")[0].strip().rstrip(";").strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if part:
                rows.append([float(v) for v in part.replace(",", " ").split()])
    width = max(len(r) for r in rows)
    out = np.zeros((len(rows), width))
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _parse_cell(text: str, name: str):
    m = re.search(r"mpc\.%s\s*=\s*\{(.*?)\};" % re.escape(name), text, re.S)
    if not m:
        return None
    return re.findall(r"'([^']*)'", m.group(1))


def load_matpower(path_or_text: str) -> dict:
    """Parse a MATPOWER version-2 case file (.m) into a ppc-style dict of numpy arrays.

    Extra cell arrays (`genfuel`, `gentype`, `bus_name`) are kept when present (ACTIVSg cases have them).
    """
    text = path_or_text
    if "\n" not in path_or_text and path_or_text.endswith(".m"):
        with open(path_or_text) as fh:
            text = fh.read()
    base = float(re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", text).group(1))
    ppc = {"baseMVA": base, "version": "2"}
    for k in ("bus", "gen", "branch", "gencost"):
        ppc[k] = _parse_matrix(text, k)
    for k in ("genfuel", "gentype", "bus_name"):
        v = _parse_cell(text, k)
        if v is not None:
            ppc[k] = v
    return ppc


@dataclass
class Network:
    """Internal-indexed network. Build with `Network.from_ppc(ppc)` or `Network.from_matpower(path)`."""
    baseMVA: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    gencost: np.ndarray | None = None
    genfuel: list | None = None
    bus_ids: np.ndarray = field(default=None)
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_ppc(cls, ppc: dict) -> "Network":
        bus = np.array(ppc["bus"], float).copy()
        gen = np.array(ppc["gen"], float).copy()
        branch = np.array(ppc["branch"], float).copy()
        if branch.shape[1] < 13:
            branch = np.hstack([branch, np.zeros((branch.shape[0], 13 - branch.shape[1]))])
            branch[:, ANGMIN], branch[:, ANGMAX] = -360, 360
        gencost = None if ppc.get("gencost") is None else np.array(ppc["gencost"], float).copy()
        ids = bus[:, BUS_I].astype(int)
        idx = {b: i for i, b in enumerate(ids)}
        bus[:, BUS_I] = np.arange(len(ids))
        gen[:, GEN_BUS] = [idx[int(b)] for b in gen[:, GEN_BUS]]
        branch[:, F_BUS] = [idx[int(b)] for b in branch[:, F_BUS]]
        branch[:, T_BUS] = [idx[int(b)] for b in branch[:, T_BUS]]
        genfuel = ppc.get("genfuel")
        net = cls(float(ppc["baseMVA"]), bus, gen, branch, gencost, genfuel, ids)
        net._fix_types()
        return net

    @classmethod
    def from_matpower(cls, path: str) -> "Network":
        return cls.from_ppc(load_matpower(path))

    def _fix_types(self):
        """Out-of-service branches/gens off; PV buses with no online gen become PQ (as pandapower does)."""
        on = self.gen[:, GEN_STATUS] > 0
        has_gen = np.zeros(self.nb, bool)
        has_gen[self.gen[on, GEN_BUS].astype(int)] = True
        t = self.bus[:, BUS_TYPE]
        self.bus[(t == PV) & ~has_gen, BUS_TYPE] = PQ
        if not np.any(self.bus[:, BUS_TYPE] == REF):
            cand = np.where(has_gen)[0]
            self.bus[cand[np.argmax(self.gen[on][:, PMAX][:len(cand)])] if len(cand) else 0, BUS_TYPE] = REF

    @property
    def nb(self):
        return self.bus.shape[0]

    @property
    def nl(self):
        return self.branch.shape[0]

    @property
    def ref(self):
        return np.where(self.bus[:, BUS_TYPE] == REF)[0]

    @property
    def pv(self):
        return np.where(self.bus[:, BUS_TYPE] == PV)[0]

    @property
    def pq(self):
        return np.where(self.bus[:, BUS_TYPE] == PQ)[0]

    @property
    def on_gen(self):
        return np.where(self.gen[:, GEN_STATUS] > 0)[0]

    @property
    def rate_MW(self):
        """Branch rating A in MW/MVA (np.inf where 0 = unlimited)."""
        r = self.branch[:, RATE_A].copy()
        r[r <= 0] = np.inf
        return r

    def copy(self):
        return Network(self.baseMVA, self.bus.copy(), self.gen.copy(), self.branch.copy(),
                       None if self.gencost is None else self.gencost.copy(), self.genfuel, self.bus_ids)

    # ------------------------------------------------------------------ matrices
    def _tap(self):
        br = self.branch
        tap = np.where(br[:, TAP] == 0, 1.0, br[:, TAP])
        return tap * np.exp(1j * np.deg2rad(br[:, SHIFT]))

    def ybus(self):
        """Ybus, Yf, Yt (sparse complex), as MATPOWER makeYbus."""
        if "ybus" in self._cache:
            return self._cache["ybus"]
        br, nb, nl = self.branch, self.nb, self.nl
        stat = br[:, BR_STATUS]
        Ys = stat / (br[:, BR_R] + 1j * br[:, BR_X])
        Bc = stat * br[:, BR_B]
        tap = self._tap()
        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / (tap * np.conj(tap))
        Yft = -Ys / np.conj(tap)
        Ytf = -Ys / tap
        Ysh = (self.bus[:, GS] + 1j * self.bus[:, BS]) / self.baseMVA
        f = br[:, F_BUS].astype(int)
        t = br[:, T_BUS].astype(int)
        i = np.r_[np.arange(nl), np.arange(nl)]
        Yf = sp.csr_matrix((np.r_[Yff, Yft], (i, np.r_[f, t])), (nl, nb))
        Yt = sp.csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f, t])), (nl, nb))
        Cf = sp.csr_matrix((np.ones(nl), (np.arange(nl), f)), (nl, nb))
        Ct = sp.csr_matrix((np.ones(nl), (np.arange(nl), t)), (nl, nb))
        Y = (Cf.T @ Yf + Ct.T @ Yt + sp.diags(Ysh)).tocsr()
        self._cache["ybus"] = (Y, Yf, Yt)
        return Y, Yf, Yt

    def bdc(self):
        """DC model matrices (MATPOWER makeBdc): Bbus, Bf, Pbusinj, Pfinj (p.u.)."""
        if "bdc" in self._cache:
            return self._cache["bdc"]
        br, nb, nl = self.branch, self.nb, self.nl
        stat = br[:, BR_STATUS]
        tap = np.where(br[:, TAP] == 0, 1.0, br[:, TAP])
        b = stat / br[:, BR_X] / tap
        f = br[:, F_BUS].astype(int)
        t = br[:, T_BUS].astype(int)
        i = np.r_[np.arange(nl), np.arange(nl)]
        Cft = sp.csr_matrix((np.r_[np.ones(nl), -np.ones(nl)], (i, np.r_[f, t])), (nl, nb))
        Bf = sp.csr_matrix((np.r_[b, -b], (i, np.r_[f, t])), (nl, nb))
        Bbus = (Cft.T @ Bf).tocsc()
        Pfinj = b * (-np.deg2rad(br[:, SHIFT]))
        Pbusinj = Cft.T @ Pfinj
        self._cache["bdc"] = (Bbus, Bf.tocsr(), Pbusinj, Pfinj)
        return self._cache["bdc"]

    def ptdf(self, slack=None):
        """Power transfer distribution factors (nl x nb), MW flow per MW injected at a bus and withdrawn at the slack.

        `slack` may be a bus index (default: the reference bus) or a weight vector over buses (distributed slack).
        """
        key = ("ptdf", None if slack is None or np.isscalar(slack) else "dist", slack if np.isscalar(slack) else None)
        if key in self._cache:
            return self._cache[key]
        Bbus, Bf, _, _ = self.bdc()
        ref = int(self.ref[0]) if (slack is None or not np.isscalar(slack)) else int(slack)
        nr = np.r_[0:ref, ref + 1:self.nb]
        lu = spla.splu(Bbus[nr][:, nr].tocsc())
        X = lu.solve(np.asarray(Bf[:, nr].todense()).T.copy(), trans="T")  # (nb-1) x nl
        H = np.zeros((self.nl, self.nb))
        H[:, nr] = X.T
        if slack is not None and not np.isscalar(slack):
            w = np.asarray(slack, float) / np.sum(slack)
            H = H - (H @ w)[:, None]
        self._cache[key] = H
        return H

    def bus_injections(self, Pg_MW=None, Pd_MW=None):
        """Net real injection per bus [MW] = generation - load (bus-indexed)."""
        Pg = self.gen[:, PG] if Pg_MW is None else Pg_MW
        Pd = self.bus[:, PD] if Pd_MW is None else Pd_MW
        on = self.gen[:, GEN_STATUS] > 0
        inj = -np.asarray(Pd, float).copy()
        np.add.at(inj, self.gen[on, GEN_BUS].astype(int), np.asarray(Pg)[on])
        return inj


# ---------------------------------------------------------------------- power flow
def dc_powerflow(net: Network, Pg_MW=None, Pd_MW=None):
    """DC power flow (lossless, |V| = 1, small angles). Slack takes the imbalance."""
    Bbus, Bf, Pbusinj, Pfinj = net.bdc()
    base = net.baseMVA
    P = net.bus_injections(Pg_MW, Pd_MW) / base - Pbusinj - net.bus[:, GS] / base
    ref = net.ref
    nr = np.setdiff1d(np.arange(net.nb), ref)
    Va = np.zeros(net.nb)
    Va[ref] = np.deg2rad(net.bus[ref, VA])
    rhs = P[nr] - Bbus[nr][:, ref] @ Va[ref]
    Va[nr] = spla.spsolve(Bbus[nr][:, nr].tocsc(), rhs)
    Pf = (Bf @ Va + Pfinj) * base
    Pbus_calc = (Bbus @ Va + Pbusinj) * base + net.bus[:, GS]
    on = net.gen[:, GEN_STATUS] > 0
    Pg = (net.gen[:, PG] if Pg_MW is None else np.asarray(Pg_MW, float)).copy()
    gref = np.where(on & np.isin(net.gen[:, GEN_BUS].astype(int), ref))[0]
    if len(gref):
        r = int(net.gen[gref[0], GEN_BUS])
        Pd = net.bus[:, PD] if Pd_MW is None else Pd_MW
        Pg[gref[0]] += Pbus_calc[r] - (net.bus_injections(Pg, Pd)[r])
    return {"Va_deg": np.rad2deg(Va), "Vm": np.ones(net.nb), "Pf_MW": Pf, "Pt_MW": -Pf, "Pg_MW": Pg,
            "losses_MW": 0.0, "loading": np.abs(Pf) / net.rate_MW, "converged": True}


def _dSbus_dV(Y, V):
    Ibus = Y @ V
    diagV = sp.diags(V)
    diagI = sp.diags(Ibus)
    diagVn = sp.diags(V / np.abs(V))
    dS_dVm = diagV @ np.conj(Y @ diagVn) + np.conj(diagI) @ diagVn
    dS_dVa = 1j * diagV @ np.conj(diagI - Y @ diagV)
    return dS_dVm, dS_dVa


def ac_powerflow(net: Network, Pg_MW=None, Pd_MW=None, Qd_MVAr=None, tol=1e-8, max_it=30, flat_start=False,
                 enforce_q_lims=False):
    """Full AC power flow by Newton-Raphson in polar coordinates (sparse Jacobian)."""
    base = net.baseMVA
    Y, Yf, Yt = net.ybus()
    on = net.gen[:, GEN_STATUS] > 0
    gbus = net.gen[on, GEN_BUS].astype(int)
    Pg = (net.gen[:, PG] if Pg_MW is None else np.asarray(Pg_MW, float)).copy()
    Qg = net.gen[:, QG].copy()
    Pd = net.bus[:, PD] if Pd_MW is None else np.asarray(Pd_MW, float)
    Qd = net.bus[:, QD] if Qd_MVAr is None else np.asarray(Qd_MVAr, float)
    btype = net.bus[:, BUS_TYPE].copy()
    ref = np.where(btype == REF)[0]
    Vm = np.ones(net.nb) if flat_start else net.bus[:, VM].copy()
    Va = np.zeros(net.nb) if flat_start else np.deg2rad(net.bus[:, VA])
    Vm[gbus] = net.gen[on, VG]
    Va[ref] = np.deg2rad(net.bus[ref, VA])
    V = Vm * np.exp(1j * Va)

    def sbus():
        S = -(Pd + 1j * Qd).astype(complex)
        np.add.at(S, gbus, Pg[on] + 1j * Qg[on])
        return S / base

    it, converged, total_it = 0, False, 0
    for _qloop in range(10):
        pv = np.where(btype == PV)[0]
        pq = np.where(btype == PQ)[0]
        pvpq = np.r_[pv, pq]
        S = sbus()
        for it in range(max_it + 1):
            mis = V * np.conj(Y @ V) - S
            F = np.r_[mis[pvpq].real, mis[pq].imag]
            if np.max(np.abs(F)) < tol:
                converged = True
                break
            if it == max_it:
                break
            dVm, dVa = _dSbus_dV(Y, V)
            J11 = dVa[pvpq][:, pvpq].real
            J12 = dVm[pvpq][:, pq].real
            J21 = dVa[pq][:, pvpq].imag
            J22 = dVm[pq][:, pq].imag
            J = sp.bmat([[J11, J12], [J21, J22]], format="csc")
            dx = -spla.spsolve(J, F)
            Va[pvpq] += dx[:len(pvpq)]
            Vm[pq] += dx[len(pvpq):]
            V = Vm * np.exp(1j * Va)
            Vm, Va = np.abs(V), np.angle(V)
        total_it += it
        # generator Q and slack P
        Sb = V * np.conj(Y @ V) * base + (Pd + 1j * Qd)
        gon = np.where(on)[0]
        for b in np.unique(gbus):
            gs = gon[gbus == b]
            qtot = Sb[b].imag
            share = np.ones(len(gs)) / len(gs)
            Qg[gs] = qtot * share
        for r in ref:
            gs = gon[gbus == r]
            if len(gs):
                Pg[gs[0]] = Sb[r].real - Pg[gs[1:]].sum()
        if not enforce_q_lims or not converged:
            break
        viol = [g for g in gon if btype[int(net.gen[g, GEN_BUS])] == PV and
                (Qg[g] > net.gen[g, QMAX] + 1e-6 or Qg[g] < net.gen[g, QMIN] - 1e-6)]
        if not viol:
            break
        for g in viol:
            Qg[g] = np.clip(Qg[g], net.gen[g, QMIN], net.gen[g, QMAX])
            btype[int(net.gen[g, GEN_BUS])] = PQ
        converged = False
    br = net.branch
    f = br[:, F_BUS].astype(int)
    t = br[:, T_BUS].astype(int)
    Sf = V[f] * np.conj(Yf @ V) * base
    St = V[t] * np.conj(Yt @ V) * base
    rate = net.rate_MW
    return {"Va_deg": np.rad2deg(Va), "Vm": Vm, "Pf_MW": Sf.real, "Qf_MVAr": Sf.imag, "Pt_MW": St.real,
            "Qt_MVAr": St.imag, "Pg_MW": Pg, "Qg_MVAr": Qg, "losses_MW": float(np.sum(Sf.real + St.real)),
            "loading": np.maximum(np.abs(Sf), np.abs(St)) / rate, "converged": bool(converged),
            "iterations": int(total_it)}


def powerflow(network, injections=None, method="dc", **kw):
    """Contract entry point.

    network: Network, ppc dict, or path to a MATPOWER .m file.
    injections: None (use the case's Pg/Pd) or dict with any of `Pg_MW` (per gen), `Pd_MW`, `Qd_MVAr` (per bus).
    method: 'dc' or 'ac'. Returns flows, angles, voltages, losses (MW, MVAr, degrees, p.u.).
    """
    net = _as_network(network)
    inj = injections or {}
    if method == "dc":
        return dc_powerflow(net, inj.get("Pg_MW"), inj.get("Pd_MW"))
    if method == "ac":
        return ac_powerflow(net, inj.get("Pg_MW"), inj.get("Pd_MW"), inj.get("Qd_MVAr"), **kw)
    raise ValueError(method)


def _as_network(network):
    if isinstance(network, Network):
        return network
    if isinstance(network, dict):
        return Network.from_ppc(network)
    return Network.from_matpower(network)


# ---------------------------------------------------------------------- DC-OPF
def dc_opf(network, Pd_MW=None, cost=None, line_limits=True, rate_scale=1.0, quadratic=True):
    """DC optimal power flow (PTDF form) solved with HiGHS (QP if highspy is available, else LP with linear costs).

    cost: optional (ngen, 3) array of [c2, c1, c0] in $/MW^2h, $/MWh, $/h (default: MATPOWER gencost, model 2).
    Returns Pg, flows, LMPs ($/MWh per bus: energy + congestion), total cost, binding lines.
    """
    net = _as_network(network)
    base = net.baseMVA
    on = np.where(net.gen[:, GEN_STATUS] > 0)[0]
    ng = len(on)
    if cost is None:
        gc = net.gencost
        c = np.zeros((net.gen.shape[0], 3))
        for g in range(net.gen.shape[0]):
            n = int(gc[g, 3])
            coef = gc[g, 4:4 + n]
            if int(gc[g, 0]) != 2:
                raise NotImplementedError("piecewise-linear gencost (model 1) not supported here")
            coef = np.r_[np.zeros(3 - min(n, 3)), coef[-3:]]
            c[g] = coef
    else:
        c = np.asarray(cost, float)
    c = c[on]
    if not quadratic:
        c = c.copy(); c[:, 0] = 0
    Pd = net.bus[:, PD] if Pd_MW is None else np.asarray(Pd_MW, float)
    Bbus, Bf, Pbusinj, Pfinj = net.bdc()
    fixed = -Pd - net.bus[:, GS] - Pbusinj * base           # MW per bus, not counting generation
    H = net.ptdf()
    Cg = np.zeros((net.nb, ng))
    Cg[net.gen[on, GEN_BUS].astype(int), np.arange(ng)] = 1
    Hg = H @ Cg
    f0 = H @ fixed + Pfinj * base
    rate = net.rate_MW * rate_scale
    lim = np.where(np.isfinite(rate) & line_limits)[0]
    pmin, pmax = net.gen[on, PMIN], net.gen[on, PMAX]
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)       # all gridsim HiGHS instances use one thread (HiGHS fixes it per process)
    # rows: balance, then line limits (lower/upper in one ranged row)
    A = np.vstack([np.ones((1, ng)), Hg[lim]])
    lo = np.r_[-fixed.sum(), -rate[lim] - f0[lim]]
    hi = np.r_[-fixed.sum(), rate[lim] - f0[lim]]
    A = sp.csc_matrix(A)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = ng, A.shape[0]
    lp.col_cost_ = c[:, 1]
    lp.col_lower_, lp.col_upper_ = pmin, pmax
    lp.row_lower_, lp.row_upper_ = lo, hi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    lp.offset_ = float(c[:, 2].sum())
    model = highspy.HighsModel()
    model.lp_ = lp
    if np.any(c[:, 0] != 0):
        hess = highspy.HighsHessian()
        hess.dim_ = ng
        hess.format_ = highspy.HessianFormat.kTriangular
        hess.start_ = np.arange(ng + 1)
        hess.index_ = np.arange(ng)
        hess.value_ = 2 * c[:, 0]
        model.hessian_ = hess
    h.passModel(model)
    h.run()
    st = h.getModelStatus()
    sol = h.getSolution()
    Pg_on = np.array(sol.col_value)
    duals = np.array(sol.row_dual)
    lam = duals[0]
    mu = duals[1:]
    # LMP_b = d cost / d load_b. Load at b shifts the balance row by -1 and each limited flow by -H[l,b]
    # (HiGHS row duals are d objective / d row bound), hence lam + H^T mu.  Checked against PYPOWER rundcopf.
    lmp = lam + H[lim].T @ mu
    Pg = np.zeros(net.gen.shape[0])
    Pg[on] = Pg_on
    flows = Hg @ Pg_on + f0
    return {"status": h.modelStatusToString(st), "Pg_MW": Pg, "Pf_MW": flows, "lmp": lmp, "lambda": lam,
            "cost": float(h.getInfo().objective_function_value),
            "binding": lim[np.abs(mu) > 1e-6], "loading": np.abs(flows) / net.rate_MW}
