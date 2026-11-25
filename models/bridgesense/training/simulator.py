"""First-principles thermo-mechanical modal FEM bridge simulator (BridgeSim).

Physics versions:
  p1: Textbook linear thermoelastic 2-D modal FEM (beam, multi-span girder, arch-deck, truss)
      + synthetic Gaussian OMA noise.
  p2: Adds 1-D Fourier deck heat diffusion (thermal lag), asphalt/ballast freezing phase transition
      below ~0 C, bearing axial thermal restraint (geometric stiffness K_G), and realistic warm-only
      baseline -> cold-query extrapolation episodes.
  p3: Adds real healthy OMA tracking noise & correlated missing-mode dropout masks from the
      unlabelled healthy autumn noise pool, plus dev-twin-calibrated thermal/freezing priors.

Zero real labels ever enter the simulator. Pure numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


# ---------------------------------------------------------------------------
# 2-D Euler-Bernoulli Frame / Beam / Truss Element Assembly
# ---------------------------------------------------------------------------

def _beam2d_local(EA: float, EI: float, rhoA: float, L: float, N_axial: float = 0.0):
    """Return local (6x6) elastic stiffness, consistent mass, and geometric stiffness matrices.

    DOFs per node: [u, w, theta] (axial, transverse, rotation).
    N_axial > 0 is tension (stiffens bending); N_axial < 0 is compression (softens bending).
    """
    L2 = L * L
    L3 = L2 * L
    k = np.zeros((6, 6), dtype=np.float64)
    # Axial
    a = EA / L
    k[0, 0] = k[3, 3] = a
    k[0, 3] = k[3, 0] = -a
    # Bending (Euler-Bernoulli)
    b = EI / L3
    kb = b * np.array([
        [12.0,     6.0 * L,   -12.0,     6.0 * L],
        [6.0 * L,  4.0 * L2,  -6.0 * L,  2.0 * L2],
        [-12.0,   -6.0 * L,    12.0,    -6.0 * L],
        [6.0 * L,  2.0 * L2,  -6.0 * L,  4.0 * L2],
    ], dtype=np.float64)
    idx_b = [1, 2, 4, 5]
    k[np.ix_(idx_b, idx_b)] = kb

    # Consistent mass matrix
    m = np.zeros((6, 6), dtype=np.float64)
    ma = rhoA * L / 6.0
    m[0, 0] = m[3, 3] = 2.0 * ma
    m[0, 3] = m[3, 0] = ma
    mb = (rhoA * L / 420.0) * np.array([
        [156.0,      22.0 * L,   54.0,     -13.0 * L],
        [22.0 * L,   4.0 * L2,   13.0 * L,  -3.0 * L2],
        [54.0,       13.0 * L,  156.0,     -22.0 * L],
        [-13.0 * L, -3.0 * L2,  -22.0 * L,   4.0 * L2],
    ], dtype=np.float64)
    m[np.ix_(idx_b, idx_b)] = mb

    # Geometric stiffness (axial force N_axial)
    kg = np.zeros((6, 6), dtype=np.float64)
    if N_axial != 0.0:
        g = N_axial / (30.0 * L)
        kgb = g * np.array([
            [36.0,      3.0 * L,   -36.0,     3.0 * L],
            [3.0 * L,   4.0 * L2,  -3.0 * L, -1.0 * L2],
            [-36.0,    -3.0 * L,    36.0,    -3.0 * L],
            [3.0 * L,  -1.0 * L2,  -3.0 * L,  4.0 * L2],
        ], dtype=np.float64)
        kg[np.ix_(idx_b, idx_b)] = kgb
    return k, m, kg


def _rot2d(c: float, s: float) -> np.ndarray:
    R = np.zeros((6, 6), dtype=np.float64)
    R[0, 0] = R[1, 1] = R[3, 3] = R[4, 4] = c
    R[0, 1] = R[3, 4] = s
    R[1, 0] = R[4, 3] = -s
    R[2, 2] = R[5, 5] = 1.0
    return R


@dataclass
class BridgeFEM:
    """Assembled 2-D bridge finite-element model split into physical stiffness/mass components."""
    kind: str                    # 'girder' | 'arch' | 'truss'
    material: str                # 'steel' | 'concrete' | 'composite'
    span_m: float
    n_dof: int
    free_dofs: np.ndarray
    K_elems: list[np.ndarray]    # per-element global elastic stiffness (reduced)
    K_surf: np.ndarray           # deck surfacing/ballast composite stiffness (reduced)
    K_geom_unit: np.ndarray      # geometric stiffness per unit axial compression N=-1 (reduced)
    K_bear: np.ndarray           # support / pier bearing springs (reduced)
    M_elems: list[np.ndarray]    # per-element global consistent mass (reduced)
    M_base: np.ndarray           # total baseline mass (reduced)
    elem_groups: list[str]       # 'deck' | 'arch' | 'hanger' | 'chord' | 'diag' | 'pier'
    deck_dofs: np.ndarray        # vertical deck DOFs (reduced indices) for added mass


def assemble_uniform_beam(L: float, EI: float, EA: float, rhoA: float, n_elem: int = 16,
                          bc: str = "simply_supported") -> tuple[np.ndarray, np.ndarray]:
    """Assemble exact 1-D/2-D uniform Euler-Bernoulli beam for analytical verification."""
    n_nodes = n_elem + 1
    n_dof = 3 * n_nodes
    K = np.zeros((n_dof, n_dof), dtype=np.float64)
    M = np.zeros((n_dof, n_dof), dtype=np.float64)
    Le = L / n_elem
    ke, me, _ = _beam2d_local(EA, EI, rhoA, Le)
    for e in range(n_elem):
        dofs = [3 * e, 3 * e + 1, 3 * e + 2, 3 * (e + 1), 3 * (e + 1) + 1, 3 * (e + 1) + 2]
        K[np.ix_(dofs, dofs)] += ke
        M[np.ix_(dofs, dofs)] += me
    fixed = set()
    if bc == "simply_supported":
        fixed.update([0, 1, 3 * (n_nodes - 1) + 1])
    elif bc == "cantilever":
        fixed.update([0, 1, 2])
    elif bc == "clamped":
        fixed.update([0, 1, 2, 3 * (n_nodes - 1), 3 * (n_nodes - 1) + 1, 3 * (n_nodes - 1) + 2])
    else:
        raise ValueError(bc)
    free = np.array([i for i in range(n_dof) if i not in fixed], dtype=int)
    return K[np.ix_(free, free)], M[np.ix_(free, free)]


def solve_modes(K: np.ndarray, M: np.ndarray, n_modes: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Solve K phi = w^2 M phi -> (freqs_hz, phi_M_normalized)."""
    n_req = min(n_modes, K.shape[0])
    vals, vecs = eigh(K, M, subset_by_index=[0, n_req - 1])
    vals = np.maximum(vals, 1e-12)
    freqs = np.sqrt(vals) / (2.0 * np.pi)
    return freqs, vecs


def build_parametric_bridge(rng: np.random.Generator, kind: str | None = None,
                            material: str | None = None) -> BridgeFEM:
    """Build a randomized physically realistic 2-D bridge FEM (girder, arch-deck, or truss)."""
    if kind is None:
        kind = str(rng.choice(["girder", "arch", "truss"], p=[0.40, 0.35, 0.25]))
    if material is None:
        if kind == "girder":
            material = str(rng.choice(["concrete", "composite", "steel"], p=[0.60, 0.25, 0.15]))
        elif kind == "arch":
            material = str(rng.choice(["steel", "composite"], p=[0.75, 0.25]))
        else:
            material = "steel"

    nodes: list[tuple[float, float]] = []
    elems: list[tuple[int, int, float, float, float, str]] = []  # (i, j, EA, EI, rhoA, group)
    fixed: set[int] = set()
    springs: list[tuple[int, float]] = []  # (dof, stiffness)

    if kind == "girder":
        n_spans = int(rng.choice([1, 2, 3], p=[0.35, 0.35, 0.30]))
        span = float(rng.uniform(25.0, 65.0))
        total_L = span * n_spans
        n_per_span = 6 if n_spans >= 2 else 12
        n_elem = n_per_span * n_spans
        for k in range(n_elem + 1):
            nodes.append((total_L * k / n_elem, 0.0))
        E = 3.5e10 if material == "concrete" else (2.1e11 if material == "steel" else 7.0e10)
        E *= float(rng.uniform(0.85, 1.15))
        depth = span * float(rng.uniform(0.045, 0.075))
        width = float(rng.uniform(8.0, 14.0))
        A = width * depth * (0.35 if material == "concrete" else 0.08)
        I = width * (depth ** 3) * (0.045 if material == "concrete" else 0.012)
        rho = 2500.0 if material == "concrete" else (7850.0 if material == "steel" else 4200.0)
        rhoA = rho * A * float(rng.uniform(1.1, 1.5))  # include non-structural deck/ballast mass
        for e in range(n_elem):
            elems.append((e, e + 1, E * A, E * I, rhoA, "deck"))
        # Boundary conditions with finite bearing springs
        k_vert_scale = E * I / ((span / 4.0) ** 3)
        k_rot_scale = E * I / span
        fixed.add(0)  # pin x at left abutment
        springs.append((1, k_vert_scale * float(rng.uniform(50.0, 400.0))))
        springs.append((2, k_rot_scale * float(rng.uniform(0.05, 1.5))))
        end_node = n_elem
        springs.append((3 * end_node, (E * A / total_L) * float(rng.uniform(0.01, 0.3))))
        springs.append((3 * end_node + 1, k_vert_scale * float(rng.uniform(50.0, 400.0))))
        springs.append((3 * end_node + 2, k_rot_scale * float(rng.uniform(0.05, 1.5))))
        # Intermediate piers
        for s_idx in range(1, n_spans):
            pn = s_idx * n_per_span
            springs.append((3 * pn + 1, k_vert_scale * float(rng.uniform(30.0, 300.0))))
            springs.append((3 * pn + 2, k_rot_scale * float(rng.uniform(0.1, 2.0))))
        deck_node_ids = list(range(n_elem + 1))

    elif kind == "arch":
        # Tied bowstring arch-deck bridge (e.g. KW51 family: 60-130 m span)
        span = float(rng.uniform(60.0, 130.0))
        rise = span * float(rng.uniform(0.14, 0.22))
        n_bays = 8
        # Deck nodes 0..n_bays, arch interior nodes (n_bays+1)..(2*n_bays-1)
        for k in range(n_bays + 1):
            nodes.append((span * k / n_bays, 0.0))
        for k in range(1, n_bays):
            x = span * k / n_bays
            y = 4.0 * rise * (x / span) * (1.0 - x / span)
            nodes.append((x, y))
        E = 2.1e11 * float(rng.uniform(0.92, 1.08))
        A_deck = float(rng.uniform(0.28, 0.55))
        I_deck = float(rng.uniform(0.06, 0.22))
        rhoA_deck = 7850.0 * A_deck + float(rng.uniform(5000.0, 12000.0))  # steel + track/ballast
        A_arch = float(rng.uniform(0.14, 0.30))
        I_arch = float(rng.uniform(0.03, 0.12))
        rhoA_arch = 7850.0 * A_arch * 1.15
        A_hang = float(rng.uniform(0.015, 0.045))
        I_hang = float(rng.uniform(2e-4, 2e-3))
        rhoA_hang = 7850.0 * A_hang * 1.1
        # Deck elements
        for k in range(n_bays):
            elems.append((k, k + 1, E * A_deck, E * I_deck, rhoA_deck, "deck"))
        # Arch rib elements
        arch_ids = [0] + [n_bays + k for k in range(1, n_bays)] + [n_bays]
        for k in range(n_bays):
            elems.append((arch_ids[k], arch_ids[k + 1], E * A_arch, E * I_arch, rhoA_arch, "arch"))
        # Vertical + diagonal hangers
        for k in range(1, n_bays):
            elems.append((k, n_bays + k, E * A_hang, E * I_hang, rhoA_hang, "hanger"))
            if k < n_bays - 1:
                elems.append((k, n_bays + k + 1, E * A_hang * 0.7, E * I_hang * 0.6, rhoA_hang * 0.7, "hanger"))
        fixed.update([0, 1, 3 * n_bays + 1])
        springs.append((3 * n_bays, (E * A_deck / span) * float(rng.uniform(0.01, 0.25))))
        deck_node_ids = list(range(n_bays + 1))

    else:
        # Steel Warren/Pratt truss bridge (e.g. HBTA family: 25-60 m span)
        span = float(rng.uniform(25.0, 60.0))
        height = span * float(rng.uniform(0.11, 0.18))
        n_bays = 6
        # Bottom chord 0..n_bays, top chord (n_bays+1)..(2*n_bays-1)
        for k in range(n_bays + 1):
            nodes.append((span * k / n_bays, 0.0))
        for k in range(1, n_bays):
            nodes.append((span * k / n_bays, height))
        E = 2.1e11 * float(rng.uniform(0.92, 1.08))
        A_ch = float(rng.uniform(0.04, 0.12))
        I_ch = float(rng.uniform(0.002, 0.012))
        rhoA_bot = 7850.0 * A_ch + float(rng.uniform(1200.0, 3500.0))
        rhoA_top = 7850.0 * A_ch * 1.15
        A_dg = float(rng.uniform(0.015, 0.05))
        I_dg = float(rng.uniform(4e-4, 3e-3))
        rhoA_dg = 7850.0 * A_dg * 1.1
        for k in range(n_bays):
            elems.append((k, k + 1, E * A_ch, E * I_ch, rhoA_bot, "chord"))
        top_ids = [0] + [n_bays + k for k in range(1, n_bays)] + [n_bays]
        for k in range(n_bays):
            elems.append((top_ids[k], top_ids[k + 1], E * A_ch, E * I_ch, rhoA_top, "chord"))
        for k in range(1, n_bays):
            elems.append((k, n_bays + k, E * A_dg, E * I_dg, rhoA_dg, "diag"))
            if k < n_bays - 1:
                elems.append((k, n_bays + k + 1, E * A_dg, E * I_dg, rhoA_dg, "diag"))
        fixed.update([0, 1, 3 * n_bays + 1])
        springs.append((3 * n_bays, (E * A_ch / span) * float(rng.uniform(0.01, 0.25))))
        deck_node_ids = list(range(n_bays + 1))

    n_dof = 3 * len(nodes)
    free = np.array([i for i in range(n_dof) if i not in fixed], dtype=int)
    free_map = {d: idx for idx, d in enumerate(free)}
    nf = len(free)

    K_elems: list[np.ndarray] = []
    M_elems: list[np.ndarray] = []
    elem_groups: list[str] = []
    K_surf = np.zeros((nf, nf), dtype=np.float64)
    K_geom_unit = np.zeros((nf, nf), dtype=np.float64)
    M_base = np.zeros((nf, nf), dtype=np.float64)
    K_bear = np.zeros((nf, nf), dtype=np.float64)

    for (i, j, EA, EI, rhoA, grp) in elems:
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        dx, dy = xj - xi, yj - yi
        Le = float(np.hypot(dx, dy))
        c, s = dx / Le, dy / Le
        R = _rot2d(c, s)
        ke_l, me_l, kg_l = _beam2d_local(EA, EI, rhoA, Le, N_axial=-1.0 if grp in ("deck", "chord") else 0.0)
        ke_g = R.T @ ke_l @ R
        me_g = R.T @ me_l @ R
        kg_g = R.T @ kg_l @ R

        dofs = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        ke_r = np.zeros((nf, nf), dtype=np.float64)
        me_r = np.zeros((nf, nf), dtype=np.float64)
        for a_loc, da in enumerate(dofs):
            if da not in free_map:
                continue
            ia = free_map[da]
            for b_loc, db in enumerate(dofs):
                if db not in free_map:
                    continue
                ib = free_map[db]
                ke_r[ia, ib] += ke_g[a_loc, b_loc]
                me_r[ia, ib] += me_g[a_loc, b_loc]
                if grp in ("deck", "chord"):
                    K_geom_unit[ia, ib] += kg_g[a_loc, b_loc]
        K_elems.append(ke_r)
        M_elems.append(me_r)
        elem_groups.append(grp)
        M_base += me_r
        if grp in ("deck", "chord"):
            # Surfacing / ballast composite flexural stiffness fraction (~3-12% at 20 C, stiffens when frozen)
            K_surf += ke_r

    for d, kv in springs:
        if d in free_map:
            idx = free_map[d]
            K_bear[idx, idx] += kv

    deck_dofs = np.array([free_map[3 * nid + 1] for nid in deck_node_ids if (3 * nid + 1) in free_map], dtype=int)
    return BridgeFEM(
        kind=kind, material=material, span_m=span, n_dof=n_dof, free_dofs=free,
        K_elems=K_elems, K_surf=K_surf, K_geom_unit=K_geom_unit, K_bear=K_bear,
        M_elems=M_elems, M_base=M_base, elem_groups=elem_groups, deck_dofs=deck_dofs,
    )


# ---------------------------------------------------------------------------
# 1-D Fourier Heat Conduction Through Bridge Deck Slab (Crank-Nicolson)
# ---------------------------------------------------------------------------

def thermal_diffusion_1d(T_air: np.ndarray, dt_s: float, alpha_m2s: float, thickness_m: float,
                         h_over_k: float = 12.0, n_z: int = 11,
                         bottom_conv_frac: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Solve dT/dt = alpha d^2T/dz^2 on z in [0, H] with Robin top BC and insulated/convective bottom.

    Returns (T_struct_mean, T_surface_top) for each time step of T_air.
    """
    n_t = len(T_air)
    dz = thickness_m / (n_z - 1)
    r = alpha_m2s * dt_s / (dz * dz)
    # Build Crank-Nicolson tridiagonal operators: A T^{n+1} = B T^n + rhs
    L = np.zeros((n_z, n_z), dtype=np.float64)
    for i in range(1, n_z - 1):
        L[i, i - 1] = 1.0
        L[i, i] = -2.0
        L[i, i + 1] = 1.0
    # Top boundary (z=0): -k dT/dz = h (T_air - T_0) => ghost node T_{-1} = T_1 + 2 dz (h/k)(T_air - T_0)
    biot = h_over_k * dz
    L[0, 0] = -2.0 * (1.0 + biot)
    L[0, 1] = 2.0
    # Bottom boundary (z=H): insulated (0) or weaker convective exchange (bottom_conv_frac * h)
    biot_b = bottom_conv_frac * biot
    L[-1, -2] = 2.0
    L[-1, -1] = -2.0 * (1.0 + biot_b)

    I = np.eye(n_z, dtype=np.float64)
    A = I - 0.5 * r * L
    B = I + 0.5 * r * L
    A_inv = np.linalg.inv(A)
    M_step = A_inv @ B

    # Forcing vector per unit T_air
    f_vec = np.zeros(n_z, dtype=np.float64)
    f_vec[0] = 2.0 * r * biot
    f_vec[-1] = 2.0 * r * biot_b
    g_step = A_inv @ f_vec

    # Exact spectral recursion using lfilter: M_step = V diag(lam) V^{-1}
    # Since L is similar to a real symmetric tridiagonal matrix (via D = diag(sqrt(2),1..1,sqrt(2))),
    # all eigenvalues of M_step are real and distinct in (0, 1).
    from scipy.signal import lfilter
    d_sim = np.ones(n_z, dtype=np.float64)
    d_sim[0] = np.sqrt(0.5)
    d_sim[-1] = np.sqrt(0.5)
    # D L D^{-1} is symmetric!
    L_sym = (d_sim[:, None] * L) / d_sim[None, :]
    evals_L, Q = eigh(L_sym)
    # Transform: M_step = D^{-1} Q diag((1 + 0.5 r lam)/(1 - 0.5 r lam)) Q^T D
    lam_M = (1.0 + 0.5 * r * evals_L) / (1.0 - 0.5 * r * evals_L)
    # Forced term in modal coordinates: u_t = T_air_mid[t] - T0_init
    T_init = float(T_air[0])
    T_mid = np.empty(n_t, dtype=np.float64)
    T_mid[0] = T_init
    if n_t > 1:
        T_mid[1:] = 0.5 * (T_air[1:] + T_air[:-1])
    u = T_mid - T_init

    # Steady-state fixed point for constant T_air = T_init is T = T_init * ones
    # Modal projection of forcing vector g_step:
    c_g = Q.T @ (d_sim * g_step)
    # Output linear functionals (trapezoidal mean and top surface)
    w_mean = np.ones(n_z, dtype=np.float64) / (n_z - 1)
    w_mean[0] *= 0.5
    w_mean[-1] *= 0.5
    w_surf = np.zeros(n_z, dtype=np.float64)
    w_surf[0] = 0.65
    w_surf[1] = 0.35

    # Readout weights in modal coordinates: w^T D^{-1} Q
    r_mean = (w_mean / d_sim) @ Q
    r_surf = (w_surf / d_sim) @ Q

    T_struct = np.full(n_t, T_init, dtype=np.float64)
    T_surf = np.full(n_t, T_init, dtype=np.float64)
    for k in range(n_z):
        if abs(c_g[k]) < 1e-16:
            continue
        zk = lfilter([c_g[k]], [1.0, -lam_M[k]], u)
        T_struct += r_mean[k] * zk
        T_surf += r_surf[k] * zk
    return T_struct, T_surf


# ---------------------------------------------------------------------------
# Fast Modal Perturbation & Episode Generator
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def simulate_temperature_series(rng: np.random.Generator, n_hours: int,
                                T_mean: float | None = None,
                                season_amp: float | None = None) -> np.ndarray:
    """Generate realistic hourly air temperature (seasonal + synoptic AR(1) + diurnal)."""
    t = np.arange(n_hours, dtype=np.float64)
    if T_mean is None:
        T_mean = float(rng.uniform(4.0, 15.0))
    if season_amp is None:
        season_amp = float(rng.uniform(6.0, 14.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    seasonal = T_mean + season_amp * np.sin(2.0 * np.pi * t / (365.25 * 24.0) + phase)
    diurnal_amp = float(rng.uniform(1.8, 5.5)) * (0.7 + 0.3 * np.sin(2.0 * np.pi * t / (365.25 * 24.0) + phase + 0.3))
    diurnal = diurnal_amp * np.sin(2.0 * np.pi * (t - 9.0) / 24.0)
    # Synoptic weather AR(1) (cold snaps & warm fronts, tau ~ 48-96 h)
    phi = float(rng.uniform(0.975, 0.992))
    sig_syn = float(rng.uniform(2.5, 5.5)) * np.sqrt(1.0 - phi * phi)
    syn = np.zeros(n_hours, dtype=np.float64)
    eps = rng.normal(0.0, sig_syn, size=n_hours)
    for i in range(1, n_hours):
        syn[i] = phi * syn[i - 1] + eps[i]
    return seasonal + diurnal + syn


def compute_bridge_frequencies(
    fem: BridgeFEM,
    T_struct: np.ndarray,
    T_surf: np.ndarray,
    T0: float,
    alpha_E: float,
    surf_share: float,
    freeze_gain: float,
    T_freeze: float,
    w_freeze: float,
    asph_beta: float,
    restraint_N_per_K: float,
    dK_elem: np.ndarray | None = None,
    dM_deck_frac: float = 0.0,
    n_modes: int = 14,
    mode_freeze_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute modal frequencies f(t, m) via Rayleigh-Ritz on the top baseline modes.

    Returns (f_tm [T, M], phi_basis [nf, n_sub]) where M = min(n_modes, max_modes).
    """
    K0 = sum(fem.K_elems) + fem.K_bear + surf_share * fem.K_surf
    M0 = fem.M_base
    n_sub = min(K0.shape[0], max(n_modes + 24, 60))
    _, Phi = solve_modes(K0, M0, n_modes=n_sub)  # [nf, n_sub], M0-orthonormal

    # Project component matrices into modal subspace (exact Rayleigh-Ritz, ~50x faster than full eigh)
    K_struct_red = Phi.T @ (sum(fem.K_elems) + fem.K_bear) @ Phi
    K_surf_red = Phi.T @ fem.K_surf @ Phi
    K_geom_red = Phi.T @ fem.K_geom_unit @ Phi

    K_delta_red = np.zeros((n_sub, n_sub), dtype=np.float64)
    if dK_elem is not None:
        K_delta = np.zeros_like(K0)
        for dk, Ke in zip(dK_elem, fem.K_elems):
            if dk != 0.0:
                K_delta += dk * Ke
        K_delta_red = Phi.T @ K_delta @ Phi

    M_red = np.eye(n_sub, dtype=np.float64)
    if dM_deck_frac != 0.0:
        M_add = np.zeros_like(M0)
        for dm_idx in fem.deck_dofs:
            M_add[dm_idx, dm_idx] += dM_deck_frac * np.mean(np.diag(M0)[fem.deck_dofs])
        M_red = M_red + Phi.T @ M_add @ Phi

    # Modal-level stiffness factors per timestep
    # Note: K_struct(T) = (1 - alpha_E*(T_struct - T0)) * K_struct
    #       K_surf(T) = surf_share * exp(-asph_beta*(T_surf - T0)) * (1 + freeze_gain * sigmoid((T_freeze - T_surf)/w_freeze)) * K_surf
    #       K_geom(T) = restraint_N_per_K * (T_struct - T0) * K_geom_unit   (positive dT -> compression -> lowers freq)
    n_t = len(T_struct)
    M_out = min(n_modes, n_sub)
    freqs = np.empty((n_t, M_out), dtype=np.float64)

    # Evaluate on unique / subsampled or direct loop (n_t <= 100 in episode windows, so direct loop is < 5 ms!)
    c_struct = np.clip(1.0 - alpha_E * (T_struct - T0), 0.70, 1.30)
    c_surf = surf_share * np.exp(np.clip(-asph_beta * (T_surf - T0), -1.5, 1.5)) * (
        1.0 + freeze_gain * _sigmoid((T_freeze - T_surf) / max(w_freeze, 0.3))
    )
    c_geom = restraint_N_per_K * (T_struct - T0)

    for t in range(n_t):
        Kt = c_struct[t] * K_struct_red + c_surf[t] * K_surf_red + c_geom[t] * K_geom_red + K_delta_red
        # Symmetrize & solve small (n_sub x n_sub) eigenproblem
        Kt = 0.5 * (Kt + Kt.T)
        vals = eigh(Kt, M_red, eigvals_only=True, subset_by_index=[0, M_out - 1])
        vals = np.maximum(vals, 1e-10)
        freqs[t] = np.sqrt(vals) / (2.0 * np.pi)

    if mode_freeze_weights is not None:
        # Mode-specific ballast/asphalt participation heterogeneity (e.g. arch vs deck modes in KW51)
        f0_ref = freqs[0:1]
        frz = _sigmoid((T_freeze - T_surf) / max(w_freeze, 0.3))[:, None]
        freqs = freqs * (1.0 + frz * mode_freeze_weights[None, :M_out])

    return freqs, Phi


def compute_physics_anchor(
    T_ref: np.ndarray,
    df_ref: np.ndarray,
    mask_ref: np.ndarray,
    T_q: np.ndarray,
    df_q: np.ndarray,
    mask_q: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Classical per-mode OLS linear thermal model fitted ONLY on the reference window.

    Identical math in Python inference and TypeScript browser instrument.
    Inputs:
      T_ref [N_ref], df_ref [N_ref, M] (fractional shift (f - f0)/f0), mask_ref [N_ref, M] (0/1)
      T_q   [N_q],   df_q   [N_q, M],                                   mask_q   [N_q, M]
    """
    M = df_ref.shape[1]
    slopes = np.zeros(M, dtype=np.float64)
    intercepts = np.zeros(M, dtype=np.float64)
    ref_resid_std = np.full(M, 0.005, dtype=np.float64)
    T_ref_mean = float(np.nanmean(T_ref)) if len(T_ref) else 10.0
    T_ref_min = float(np.nanmin(T_ref)) if len(T_ref) else 10.0
    T_ref_max = float(np.nanmax(T_ref)) if len(T_ref) else 10.0

    for m in range(M):
        ok = (mask_ref[:, m] > 0.5) & np.isfinite(df_ref[:, m]) & np.isfinite(T_ref)
        if np.sum(ok) >= 4:
            x = T_ref[ok] - T_ref_mean
            y = df_ref[ok, m]
            vx = float(np.dot(x, x))
            if vx > 1e-4:
                a = float(np.dot(x, y) / (vx + 1e-2))
                # Physical sanity clip on linear thermal slope ([-0.015, +0.005] / K)
                a = float(np.clip(a, -0.015, 0.005))
            else:
                a = 0.0
            b = float(np.mean(y) - a * np.mean(x))
            slopes[m] = a
            intercepts[m] = b
            r = y - (a * x + b)
            ref_resid_std[m] = float(max(np.std(r), 5e-4))
        elif np.sum(ok) >= 1:
            intercepts[m] = float(np.mean(df_ref[ok, m]))

    pred_ref = intercepts[None, :] + slopes[None, :] * (T_ref[:, None] - T_ref_mean)
    pred_q = intercepts[None, :] + slopes[None, :] * (T_q[:, None] - T_ref_mean)
    resid_ref = np.where(mask_ref > 0.5, df_ref - pred_ref, 0.0)
    resid_q = np.where(mask_q > 0.5, df_q - pred_q, 0.0)

    # Per-mode mean query residual and z-score
    q_counts = np.sum(mask_q > 0.5, axis=0)
    mode_mean_resid = np.zeros(M, dtype=np.float64)
    mode_mean_raw = np.zeros(M, dtype=np.float64)
    for m in range(M):
        if q_counts[m] > 0:
            ok = mask_q[:, m] > 0.5
            mode_mean_resid[m] = float(np.mean(resid_q[ok, m]))
            mode_mean_raw[m] = float(np.mean(df_q[ok, m]))

    active = q_counts > 0
    if np.any(active):
        # Implied stiffness change from linear-thermal residual: dK/K ~= 2 * df/f (in %)
        mean_resid = float(np.mean(mode_mean_resid[active]))
        anchor_dk_pct = 200.0 * mean_resid
        anchor_env_pct = 100.0 * float(np.mean(mode_mean_raw[active]) - mean_resid)
        mode_z = mode_mean_resid / ref_resid_std
        max_abs_z = float(np.max(np.abs(mode_z[active])))
        mean_abs_z = float(np.mean(np.abs(mode_z[active])))
        # Inter-mode spread of structural residual (large spread = localized damage/retrofit vs uniform shift)
        resid_std_modes = float(np.std(mode_mean_resid[active])) if np.sum(active) >= 2 else 0.0
    else:
        anchor_dk_pct = 0.0
        anchor_env_pct = 0.0
        max_abs_z = 0.0
        mean_abs_z = 0.0
        resid_std_modes = 0.0

    # Cold extrapolation below baseline minimum temperature
    T_q_min = float(np.nanmin(T_q)) if len(T_q) else T_ref_min
    T_q_mean = float(np.nanmean(T_q)) if len(T_q) else T_ref_mean
    cold_extrap = max(0.0, T_ref_min - T_q_min)
    warm_extrap = max(0.0, float(np.nanmax(T_q)) - T_ref_max) if len(T_q) else 0.0
    freeze_frac_q = float(np.mean(T_q < 1.0)) if len(T_q) else 0.0
    freeze_frac_ref = float(np.mean(T_ref < 1.0)) if len(T_ref) else 0.0

    return {
        "slopes": slopes,
        "intercepts": intercepts,
        "ref_resid_std": ref_resid_std,
        "resid_ref": resid_ref,
        "resid_q": resid_q,
        "mode_mean_resid": mode_mean_resid,
        "mode_mean_raw": mode_mean_raw,
        "anchor_dk_pct": anchor_dk_pct,
        "anchor_env_pct": anchor_env_pct,
        "max_abs_z": max_abs_z,
        "mean_abs_z": mean_abs_z,
        "resid_std_modes": resid_std_modes,
        "T_ref_mean": T_ref_mean,
        "T_ref_min": T_ref_min,
        "T_ref_max": T_ref_max,
        "T_q_mean": T_q_mean,
        "T_q_min": T_q_min,
        "cold_extrap": cold_extrap,
        "warm_extrap": warm_extrap,
        "freeze_frac_q": freeze_frac_q,
        "freeze_frac_ref": freeze_frac_ref,
    }


def simulate_episode(
    rng: np.random.Generator,
    physics_version: str = "p2",
    n_ref: int = 48,
    n_q: int = 48,
    max_modes: int = 14,
    noise_pool: dict | None = None,
    cal: dict | None = None,
) -> dict[str, np.ndarray | float | int | str]:
    """Simulate one bridge monitoring episode (healthy baseline window + query window).

    Returns fixed-shape arrays ready for BridgeSense training & evaluation.
    """
    cal = cal or {}
    fem = build_parametric_bridge(rng)
    # Number of tracked modes in this episode (2..14, reflecting 2-mode concrete to 14-mode KW51)
    n_modes_pool = [2, 3, 4, 6, 8, 10, 12, 14]
    n_modes_probs = [0.16, 0.08, 0.10, 0.12, 0.12, 0.12, 0.12, 0.18]
    M_active = int(rng.choice(n_modes_pool, p=n_modes_probs))
    M_active = min(M_active, fem.K_surf.shape[0] - 2)

    # Material-specific thermo-mechanical priors
    if fem.material == "steel":
        alpha_E = float(rng.uniform(2.0e-4, 6.0e-4))
        alpha_th = float(rng.uniform(8e-6, 1.4e-5))  # thermal diffusivity m^2/s (effective deck+ballast)
        H_slab = float(rng.uniform(0.15, 0.45))
        surf_share = float(rng.uniform(0.01, 0.06)) if physics_version != "p1" else 0.0
        freeze_gain = float(rng.uniform(0.2, 2.2)) if physics_version != "p1" else 0.0
        asph_beta = float(rng.uniform(0.002, 0.015)) if physics_version != "p1" else 0.0
    elif fem.material == "concrete":
        alpha_E = float(rng.uniform(1.0e-3, 3.5e-3))
        alpha_th = float(rng.uniform(4e-7, 1.2e-6))  # concrete diffusivity ~ 0.6e-6 m^2/s
        H_slab = float(rng.uniform(0.25, 0.65))
        surf_share = float(rng.uniform(0.03, 0.14)) if physics_version != "p1" else 0.0
        freeze_gain = float(rng.uniform(0.6, 3.8)) if physics_version != "p1" else 0.0
        asph_beta = float(rng.uniform(0.010, 0.035)) if physics_version != "p1" else 0.0
    else:  # composite
        alpha_E = float(rng.uniform(4.0e-4, 1.8e-3))
        alpha_th = float(rng.uniform(8e-7, 3.0e-6))
        H_slab = float(rng.uniform(0.20, 0.55))
        surf_share = float(rng.uniform(0.02, 0.10)) if physics_version != "p1" else 0.0
        freeze_gain = float(rng.uniform(0.4, 3.0)) if physics_version != "p1" else 0.0
        asph_beta = float(rng.uniform(0.005, 0.025)) if physics_version != "p1" else 0.0

    # Apply dev-calibrated scaling if present (p3)
    freeze_scale = float(cal.get("freeze_scale", 1.0))
    freeze_gain *= freeze_scale
    T_freeze = float(rng.uniform(cal.get("T_freeze_lo", -3.0), cal.get("T_freeze_hi", 1.5)))
    w_freeze = float(rng.uniform(0.8, 2.8))

    # Bearing axial thermal restraint (p2+): small fraction of Euler buckling load per K
    if physics_version != "p1" and rng.random() < 0.65:
        K0_est = sum(fem.K_elems)
        eig1 = float(solve_modes(K0_est + fem.K_bear, fem.M_base, n_modes=1)[0][0])
        # Scale so 10 K warming shifts f1 by 0.2% to 2.5%
        restraint_N_per_K = float(rng.uniform(0.0, 1.2e4)) * (fem.span_m / 40.0)
        # Normalize relative to first mode so it never buckles unphysically
        w1_sq = (2.0 * np.pi * eig1) ** 2
        g_norm = float(np.linalg.norm(fem.K_geom_unit, ord=2)) + 1e-12
        max_N_per_K = 0.004 * w1_sq * float(np.mean(np.diag(fem.M_base))) / g_norm
        restraint_N_per_K = float(rng.uniform(0.0, 1.0)) * max_N_per_K
    else:
        restraint_N_per_K = 0.0

    # Generate a 1-year hourly air temperature trace and pick baseline + query windows
    n_year = 24 * 365
    T_air_yr = simulate_temperature_series(rng, n_year)
    if physics_version != "p1":
        T_struct_yr, T_surf_yr = thermal_diffusion_1d(T_air_yr, dt_s=3600.0, alpha_m2s=alpha_th, thickness_m=H_slab)
    else:
        T_struct_yr = T_air_yr.copy()
        T_surf_yr = T_air_yr.copy()

    # Precompute 24h and 72h causal moving averages of air temperature (what the operator observes!)
    cs = np.concatenate([[0.0], np.cumsum(T_air_yr)])
    idx_all = np.arange(n_year)
    T24_yr = (cs[idx_all + 1] - cs[np.maximum(0, idx_all + 1 - 24)]) / np.minimum(idx_all + 1, 24)
    T72_yr = (cs[idx_all + 1] - cs[np.maximum(0, idx_all + 1 - 72)]) / np.minimum(idx_all + 1, 72)

    # Choose baseline (reference) window and query window
    # 35% of p2/p3 episodes deliberately have warm-only baseline -> cold/freezing query (hardest real SHM regime!)
    cold_idx = np.where(T_air_yr[24 * 10:-24 * 10] < 0.5)[0] + 24 * 10
    warm_idx = np.where(T_air_yr[24 * 10:-24 * 10] > 7.0)[0] + 24 * 10
    if physics_version != "p1" and rng.random() < 0.35 and len(cold_idx) > 48 and len(warm_idx) > 200:
        ref_center = int(rng.choice(warm_idx))
        q_center = int(rng.choice(cold_idx))
        ref_span_h = int(rng.integers(24 * 14, 24 * 60))
        q_span_h = int(rng.integers(24 * 3, 24 * 14))
    else:
        ref_span_h = int(rng.integers(24 * 10, 24 * 90))
        q_span_h = int(rng.integers(24 * 2, 24 * 21))
        ref_center = int(rng.integers(ref_span_h // 2 + 72, n_year - ref_span_h // 2 - 1))
        q_center = int(rng.integers(q_span_h // 2 + 72, n_year - q_span_h // 2 - 1))

    ref_lo = max(72, ref_center - ref_span_h // 2)
    ref_hi = min(n_year - 1, ref_lo + ref_span_h)
    q_lo = max(72, q_center - q_span_h // 2)
    q_hi = min(n_year - 1, q_lo + q_span_h)

    idx_ref = np.sort(rng.choice(np.arange(ref_lo, ref_hi), size=n_ref, replace=(ref_hi - ref_lo < n_ref)))
    idx_q = np.sort(rng.choice(np.arange(q_lo, q_hi), size=n_q, replace=(q_hi - q_lo < n_q)))

    T_ref_air = T_air_yr[idx_ref]
    T_q_air = T_air_yr[idx_q]
    T0 = float(np.median(T_ref_air))

    # Decide structural state: healthy (0), damage (-dK), retrofit (+dK), or added mass (+dM)
    u = float(rng.random())
    n_elems = len(fem.K_elems)
    dK_elem = np.zeros(n_elems, dtype=np.float64)
    dM_deck_frac = 0.0
    if u < 0.48:
        scenario = "healthy"
        is_changed = 0
    elif u < 0.74:
        scenario = "damage"
        is_changed = 1
        # Localized or multi-element stiffness loss (1 to 4 elements, e.g. crack, truss member, pier)
        k_hit = int(rng.integers(1, min(5, n_elems) + 1))
        hit_ids = rng.choice(n_elems, size=k_hit, replace=False)
        sev = float(rng.uniform(0.06, 0.65))
        for hi in hit_ids:
            dK_elem[hi] = -sev * float(rng.uniform(0.6, 1.0))
    elif u < 0.90:
        scenario = "retrofit"
        is_changed = 1
        # Connection / hanger / joint stiffening (like KW51 diagonal-to-deck bolting)
        k_hit = int(rng.integers(max(2, n_elems // 5), max(3, n_elems // 2) + 1))
        hit_ids = rng.choice(n_elems, size=min(k_hit, n_elems), replace=False)
        gain = float(rng.uniform(0.06, 0.45))
        for hi in hit_ids:
            dK_elem[hi] = gain * float(rng.uniform(0.6, 1.0))
        # In p3: real steel retrofits also add localized plate/box mass (e.g., KW51 diagonal boxes)
        if physics_version == "p3" and rng.random() < 0.5:
            dM_deck_frac = float(rng.uniform(0.005, 0.035))
    else:
        scenario = "mass"
        is_changed = 1
        dM_deck_frac = float(rng.uniform(0.025, 0.18))

    # Mode-specific freezing heterogeneity (e.g. deck/ballast modes jump more than steel arch modes)
    if physics_version != "p1" and rng.random() < 0.75:
        mode_frz_w = rng.uniform(0.0, float(cal.get("mode_freeze_extra", 0.045)), size=M_active) * freeze_scale
    else:
        mode_frz_w = None

    # Compute exact counterfactual frequencies:
    # 1. Healthy baseline window
    f_ref_clean, _ = compute_bridge_frequencies(
        fem, T_struct_yr[idx_ref], T_surf_yr[idx_ref], T0,
        alpha_E, surf_share, freeze_gain, T_freeze, w_freeze, asph_beta, restraint_N_per_K,
        dK_elem=None, dM_deck_frac=0.0, n_modes=M_active, mode_freeze_weights=mode_frz_w,
    )
    # 2. Query window WITHOUT structural change (pure thermal counterfactual!)
    f_q_thermal, _ = compute_bridge_frequencies(
        fem, T_struct_yr[idx_q], T_surf_yr[idx_q], T0,
        alpha_E, surf_share, freeze_gain, T_freeze, w_freeze, asph_beta, restraint_N_per_K,
        dK_elem=None, dM_deck_frac=0.0, n_modes=M_active, mode_freeze_weights=mode_frz_w,
    )
    # 3. Query window WITH actual structural state
    if is_changed:
        f_q_clean, _ = compute_bridge_frequencies(
            fem, T_struct_yr[idx_q], T_surf_yr[idx_q], T0,
            alpha_E, surf_share, freeze_gain, T_freeze, w_freeze, asph_beta, restraint_N_per_K,
            dK_elem=dK_elem, dM_deck_frac=dM_deck_frac, n_modes=M_active, mode_freeze_weights=mode_frz_w,
        )
    else:
        f_q_clean = f_q_thermal.copy()

    # Baseline median frequency per mode (f_{m,0})
    f0 = np.median(f_ref_clean, axis=0)
    df_ref_clean = (f_ref_clean - f0[None, :]) / f0[None, :]
    df_q_thermal = (f_q_thermal - f0[None, :]) / f0[None, :]
    df_q_clean = (f_q_clean - f0[None, :]) / f0[None, :]

    # Ground-truth physical quantities (computed BEFORE noise!)
    # Effective global stiffness-equivalent shift dK/K = ((f_actual / f_thermal)^2 - 1) * 100 (%)
    # (negative for damage or added mass, positive for retrofit, 0 for healthy)
    mode_dk_pct = 100.0 * (np.mean((f_q_clean / f_q_thermal) ** 2, axis=0) - 1.0)
    true_dk_pct = float(np.mean(mode_dk_pct))
    true_env_pct = float(100.0 * np.mean(df_q_thermal))

    # If a tiny random perturbation ended up below 0.5% effective shift, re-label honestly or amplify
    if is_changed and abs(true_dk_pct) < 0.6 and float(np.max(np.abs(mode_dk_pct))) < 1.2:
        # Amplify slightly so structural-change episodes are physically distinguishable from OMA noise
        scale_up = 1.8 / max(float(np.max(np.abs(mode_dk_pct))), 0.15)
        df_struct = (f_q_clean - f_q_thermal) / f0[None, :]
        df_q_clean = df_q_thermal + df_struct * min(scale_up, 4.0)
        f_q_adj = f0[None, :] * (1.0 + df_q_clean)
        mode_dk_pct = 100.0 * (np.mean((f_q_adj / f_q_thermal) ** 2, axis=0) - 1.0)
        true_dk_pct = float(np.mean(mode_dk_pct))

    # Inject OMA observation noise & missing-mode masks
    df_ref_obs = np.zeros((n_ref, max_modes), dtype=np.float32)
    df_q_obs = np.zeros((n_q, max_modes), dtype=np.float32)
    mask_ref = np.zeros((n_ref, max_modes), dtype=np.float32)
    mask_q = np.zeros((n_q, max_modes), dtype=np.float32)

    if physics_version == "p3" and noise_pool is not None and len(noise_pool["resid_windows"]) > 0:
        # Draw real OMA residual noise & real missingness masks from healthy KW51 autumn pool
        n_pool = len(noise_pool["resid_windows"])
        w_ref_id = int(rng.integers(0, n_pool))
        w_q_id = int(rng.integers(0, n_pool))
        n_src_m = noise_pool["resid_windows"].shape[2]
        m_pick = rng.choice(n_src_m, size=M_active, replace=(M_active > n_src_m))
        noise_scale = float(rng.uniform(0.5, 1.5)) * float(cal.get("noise_scale", 1.0))

        nr = noise_pool["resid_windows"][w_ref_id, :n_ref, :][:, m_pick] * noise_scale
        nq = noise_pool["resid_windows"][w_q_id, :n_q, :][:, m_pick] * noise_scale
        mr = noise_pool["mask_windows"][w_ref_id, :n_ref, :][:, m_pick].astype(np.float32)
        mq = noise_pool["mask_windows"][w_q_id, :n_q, :][:, m_pick].astype(np.float32)
        # Ensure each active mode has at least 4 valid points in ref and 2 in query
        for m in range(M_active):
            if mr[:, m].sum() < 5:
                mr[rng.choice(n_ref, size=max(6, n_ref // 3), replace=False), m] = 1.0
            if mq[:, m].sum() < 3:
                mq[rng.choice(n_q, size=max(4, n_q // 3), replace=False), m] = 1.0

        # Dev-twin-calibrated slow unmodelled environmental/moisture/solar-heating drift between windows
        # (prevents false alarms on real 0.15-0.30% healthy autumn-to-spring wander)
        g_drift_std = float(cal.get("global_drift_std", 0.0014))
        m_drift_std = float(cal.get("mode_drift_std", 0.0018))
        drift_q = rng.normal(0.0, g_drift_std) + rng.normal(0.0, m_drift_std, size=(1, M_active))
        df_ref_obs[:, :M_active] = np.where(mr > 0.5, df_ref_clean + nr, 0.0)
        df_q_obs[:, :M_active] = np.where(mq > 0.5, df_q_clean + nq + drift_q, 0.0)
        mask_ref[:, :M_active] = mr
        mask_q[:, :M_active] = mq
    else:
        # Synthetic OMA noise (Gaussian + mild AR(1) + occasional tracking outlier)
        sig_oma = float(rng.uniform(0.0012, 0.0060))
        nr = rng.normal(0.0, sig_oma, size=(n_ref, M_active))
        nq = rng.normal(0.0, sig_oma, size=(n_q, M_active))
        if physics_version != "p1":
            # Add occasional 1-2% mode-tracking glitches (1% of readings)
            glitch_r = (rng.random((n_ref, M_active)) < 0.012) * rng.normal(0.0, 0.015, size=(n_ref, M_active))
            glitch_q = (rng.random((n_q, M_active)) < 0.012) * rng.normal(0.0, 0.015, size=(n_q, M_active))
            nr += glitch_r
            nq += glitch_q
        miss_rate = float(rng.uniform(0.0, 0.40)) if physics_version != "p1" else float(rng.uniform(0.0, 0.15))
        mr = (rng.random((n_ref, M_active)) >= miss_rate).astype(np.float32)
        mq = (rng.random((n_q, M_active)) >= miss_rate).astype(np.float32)
        for m in range(M_active):
            if mr[:, m].sum() < 5:
                mr[rng.choice(n_ref, size=6, replace=False), m] = 1.0
            if mq[:, m].sum() < 3:
                mq[rng.choice(n_q, size=4, replace=False), m] = 1.0
        df_ref_obs[:, :M_active] = np.where(mr > 0.5, df_ref_clean + nr, 0.0)
        df_q_obs[:, :M_active] = np.where(mq > 0.5, df_q_clean + nq, 0.0)
        mask_ref[:, :M_active] = mr
        mask_q[:, :M_active] = mq

    # Re-center observed frequencies by observed reference median (exactly as done on real data!)
    for m in range(M_active):
        ok = mask_ref[:, m] > 0.5
        med_m = float(np.median(df_ref_obs[ok, m]))
        df_ref_obs[:, m] = np.where(ok, (1.0 + df_ref_obs[:, m]) / (1.0 + med_m) - 1.0, 0.0)
        ok_q = mask_q[:, m] > 0.5
        df_q_obs[:, m] = np.where(ok_q, (1.0 + df_q_obs[:, m]) / (1.0 + med_m) - 1.0, 0.0)

    # Observed air temperature features (air T, 24h lag, 72h lag) + small thermometer noise
    t_noise_r = rng.normal(0.0, 0.25, size=n_ref)
    t_noise_q = rng.normal(0.0, 0.25, size=n_q)
    T_ref_feat = np.stack([T_ref_air + t_noise_r, T24_yr[idx_ref], T72_yr[idx_ref]], axis=-1).astype(np.float32)
    T_q_feat = np.stack([T_q_air + t_noise_q, T24_yr[idx_q], T72_yr[idx_q]], axis=-1).astype(np.float32)

    # Random permutation of active mode slots (enforces strict mode-order invariance)
    perm = rng.permutation(M_active)
    df_ref_obs[:, :M_active] = df_ref_obs[:, perm]
    df_q_obs[:, :M_active] = df_q_obs[:, perm]
    mask_ref[:, :M_active] = mask_ref[:, perm]
    mask_q[:, :M_active] = mask_q[:, perm]
    mode_dk_padded = np.zeros(max_modes, dtype=np.float32)
    mode_dk_padded[:M_active] = mode_dk_pct[perm]

    return {
        "T_ref": T_ref_feat,          # [N_ref, 3]: T_air, T_24h, T_72h
        "T_q": T_q_feat,              # [N_q, 3]
        "df_ref": df_ref_obs,         # [N_ref, M_max]
        "df_q": df_q_obs,             # [N_q, M_max]
        "mask_ref": mask_ref,         # [N_ref, M_max]
        "mask_q": mask_q,             # [N_q, M_max]
        "n_modes": M_active,
        "is_changed": int(is_changed),
        "true_dk_pct": float(true_dk_pct),
        "true_env_pct": float(true_env_pct),
        "mode_dk_pct": mode_dk_padded,
        "scenario": scenario,
        "kind": fem.kind,
        "material": fem.material,
    }
