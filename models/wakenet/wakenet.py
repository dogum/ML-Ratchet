"""WakeNet: how much power is each wind turbine losing to its neighbours' wakes?

From a wind farm's turbine coordinates, hub heights, rotor power/thrust curves, mechanical yaw
angles, online flags, and the upstream freestream wind measured on the unwaked front row, WakeNet
predicts every turbine's waked power ratio P_i / P_front, wake power loss (kW and %), rotor-effective
wind speed, and farm-wide wake loss — each with a 68% error bar widened on dev data only.

Standalone numpy only — never imports torch, floris, or mlratchet.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(mod_name: str):
    path = os.path.join(HERE, f"{mod_name}.py")
    spec = importlib.util.spec_from_file_location(f"_wakenet_{mod_name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


WP = _load_sibling("physics")
PL = _load_sibling("pipeline")


def _erf(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return np.sign(x) * y


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x * 0.7071067811865476))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b


class WakeNet:
    """Physics-anchored directed Graph Neural Network for wind-farm wake losses (numpy-only)."""

    def __init__(self, weights: str = os.path.join(HERE, "wakenet_weights.npz")):
        z = np.load(weights)
        self.meta = json.loads(str(z["meta"]))
        self.W = {k: z[k].astype(np.float32) for k in z.files if k != "meta"}

    def forward_graph(
        self,
        node_x: np.ndarray,
        edge_x: np.ndarray,
        edge_mask: np.ndarray,
        online: np.ndarray,
        is_free: np.ndarray,
        eta_anchor: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        """Single-farm graph forward pass matching training/model.py EdgeWakeGNN to ~1e-6."""
        W = self.W
        arch = self.meta["arch"]
        n_layers = int(arch["n_layers"])
        anchored = bool(arch.get("anchored", True))
        use_edges = bool(arch.get("use_edges", True))
        agg = str(arch.get("agg", "gated_sum"))

        nx = np.asarray(node_x, dtype=np.float32)
        ex = np.asarray(edge_x, dtype=np.float32)
        emask = np.asarray(edge_mask, dtype=bool)
        on = np.asarray(online, dtype=bool)
        free = np.asarray(is_free, dtype=bool)
        anc = np.asarray(eta_anchor, dtype=np.float32)

        on_f = on.astype(np.float32)
        free_f = free.astype(np.float32)
        udef_anc = nx[:, 13].copy()

        if not anchored:
            nx = nx.copy()
            nx[:, 12:17] = 0.0
            ex = ex.copy()
            ex[:, :, [3, 6, 7, 8]] = 0.0

        N = nx.shape[0]
        h = _gelu(_linear(nx, W["node_enc.weight"], W["node_enc.bias"])) * on_f[:, None]

        if use_edges and n_layers > 0:
            e_emb = _gelu(_linear(ex, W["edge_enc.weight"], W["edge_enc.bias"])) * emask[:, :, None].astype(np.float32)
            H = h.shape[1]
            for l in range(n_layers):
                p = f"layers.{l}."
                hj = np.broadcast_to(h[:, None, :], (N, N, H))
                hi = np.broadcast_to(h[None, :, :], (N, N, H))
                z_pair = np.concatenate([hj, hi, e_emb], axis=-1)
                m_hid = _gelu(_linear(z_pair, W[p + "msg_fc1.weight"], W[p + "msg_fc1.bias"]))
                msg = _linear(m_hid, W[p + "msg_fc2.weight"], W[p + "msg_fc2.bias"])
                logit = _linear(m_hid, W[p + "gate_fc.weight"], W[p + "gate_fc.bias"])[:, :, 0]

                if agg == "softmax_attn":
                    scores = np.where(emask, logit, -1e9)
                    s_max = np.max(scores, axis=0, keepdims=True)
                    exps = np.where(emask, np.exp(scores - s_max), 0.0)
                    denom_e = np.maximum(np.sum(exps, axis=0, keepdims=True), 1e-12)
                    weights = exps / denom_e
                else:
                    weights = _sigmoid(logit) * emask.astype(np.float32)

                agg_in = np.einsum("ji,jih->ih", weights, msg, optimize=True)
                on_cnt = max(float(np.sum(on_f)), 1.0)
                h_farm = np.sum(h * on_f[:, None], axis=0, keepdims=True) / on_cnt
                h_farm_exp = np.broadcast_to(h_farm, (N, H))

                u_in = np.concatenate([h, agg_in, h_farm_exp], axis=-1)
                dh = _linear(_gelu(_linear(u_in, W[p + "upd_fc1.weight"], W[p + "upd_fc1.bias"])), W[p + "upd_fc2.weight"], W[p + "upd_fc2.bias"])
                h = (h + dh) * on_f[:, None]
        else:
            for l in range(n_layers):
                p = f"layers.{l}."
                dh = _linear(_gelu(_linear(h, W[p + "0.weight"], W[p + "0.bias"])), W[p + "1.weight"], W[p + "1.bias"])
                h = (h + 0.35 * dh) * on_f[:, None]

        nh = _gelu(_linear(np.concatenate([h, nx], axis=-1), W["node_head1.weight"], W["node_head1.bias"]))
        nout = _linear(nh, W["node_head2.weight"], W["node_head2.bias"])

        raw_eta = nout[:, 0]
        log_sig_eta = np.clip(nout[:, 1], -5.0, -0.2)
        d_udef = nout[:, 2]

        if anchored:
            eta_unnorm = (anc + 0.5 * np.tanh(raw_eta)) * on_f
            u_def_pred = np.clip(udef_anc + 0.35 * np.tanh(d_udef), -0.15, 0.95) * on_f
        else:
            eta_unnorm = _sigmoid(raw_eta + 1.2) * 1.25 * on_f
            u_def_pred = np.clip(_sigmoid(d_udef - 1.0), 0.0, 0.95) * on_f

        f_cnt = max(float(np.sum(free_f)), 1.0)
        f_mean = max(float(np.sum(eta_unnorm * free_f)) / f_cnt, 0.25)
        eta_pred = np.clip(eta_unnorm / f_mean, 0.0, 1.35) * on_f

        on_cnt = max(float(np.sum(on_f)), 1.0)
        farm_wake_pred = float(1.0 - np.sum(eta_pred) / on_cnt)

        h_mean = np.sum(h * on_f[:, None], axis=0) / on_cnt
        waked_f = (on & (~free)).astype(np.float32)
        w_cnt = max(float(np.sum(waked_f)), 1.0)
        h_waked = np.sum(h * waked_f[:, None], axis=0) / w_cnt
        f_scalars = np.array([farm_wake_pred, float(np.sum(waked_f)) / on_cnt, float(nx[0, 2]), float(nx[0, 6])], dtype=np.float32)

        fh = _gelu(_linear(np.concatenate([h_mean, h_waked, f_scalars], axis=-1), W["farm_head1.weight"], W["farm_head1.bias"]))
        log_sig_farm = float(np.clip(_linear(fh, W["farm_head2.weight"], W["farm_head2.bias"])[0], -5.5, -0.5))

        return {
            "eta_pred": eta_pred.astype(np.float32),
            "log_sig_eta": log_sig_eta.astype(np.float32),
            "u_def_pred": u_def_pred.astype(np.float32),
            "farm_wake_pred": farm_wake_pred,
            "log_sig_farm": log_sig_farm,
        }

    def read(
        self,
        east_m: np.ndarray,
        north_m: np.ndarray,
        hub_h_m: np.ndarray | float,
        turbine: dict | str,
        u_inf: float,
        wind_dir_deg: float,
        ti_inf: float = 0.09,
        sigma_wd_deg: float = 6.0,
        yaw_deg: np.ndarray | float = 0.0,
        online: np.ndarray | None = None,
        elev_m: np.ndarray | float = 0.0,
        air_density: float = 1.225,
        p_front_kw: float | None = None,
        calibrated: bool = True,
    ) -> dict:
        """Predict per-turbine and farm-wide wake losses with calibrated 68% error bars."""
        turb = WP.TURBINES[turbine] if isinstance(turbine, str) else turbine
        g = PL.build_graph(
            east_m, north_m, hub_h_m, turb,
            u_inf=u_inf, wind_dir_deg=wind_dir_deg, ti_inf=ti_inf, sigma_wd_deg=sigma_wd_deg,
            yaw_deg=yaw_deg, online=online, elev_m=elev_m, air_density=air_density,
        )
        raw = self.forward_graph(
            g["node_x"], g["edge_x"], g["edge_mask"],
            g["online"], g["is_free"], g["eta_anchor"],
        )

        sf = self.meta.get("sigma_factor", {})
        f_eta = float(sf.get("waked_eta", 1.0)) if calibrated else 1.0
        f_free = float(sf.get("free_eta", f_eta)) if calibrated else 1.0
        f_farm = float(sf.get("farm_wake", 1.0)) if calibrated else 1.0

        on = g["online"]
        free = g["is_free"]
        scale_vec = np.where(free, f_free, f_eta)
        eta_sig = np.where(on, np.exp(raw["log_sig_eta"]) * scale_vec, 0.0)
        farm_sig = float(np.exp(raw["log_sig_farm"]) * f_farm)

        # Reference unwaked front-row power (measured if supplied, else physics curve at front row)
        if p_front_kw is None:
            rho_sc = (float(air_density) / float(turb.get("ref_air_density", 1.225))) ** (1.0 / 3.0)
            p_ref = float(np.interp(float(u_inf) * rho_sc, turb["wind_speed"], turb["power_kw"]))
        else:
            p_ref = float(p_front_kw)

        eta = raw["eta_pred"]
        p_kw = eta * p_ref
        p_sig_kw = eta_sig * p_ref
        u_ms = np.where(on, g["u_free_ms"] * (1.0 - raw["u_def_pred"]), 0.0)

        return {
            "eta": eta,
            "eta_sig": eta_sig,
            "wake_loss": np.where(on, 1.0 - eta, 0.0),
            "wake_loss_sig": eta_sig,
            "power_kw": p_kw,
            "power_sig_kw": p_sig_kw,
            "u_rotor_ms": u_ms,
            "eta_anchor": g["eta_anchor"],
            "is_free": free,
            "online": on,
            "p_front_kw": p_ref,
            "farm_power_kw": float(np.sum(p_kw[on])),
            "farm_free_kw": float(np.sum(on) * p_ref),
            "farm_wake_loss": float(raw["farm_wake_pred"]),
            "farm_wake_sig": farm_sig,
        }
