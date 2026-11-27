"""Pure-numpy gradient-boosted decision tree runtime and exact TreeSHAP.

No xgboost, lightgbm or shap imports at inference time. A trained ensemble is stored as flat
numpy arrays (one contiguous node table across all trees), evaluated with a vectorised tree walk,
and explained with exact path-dependent TreeSHAP (Lundberg et al., Nature Machine Intelligence
2020, Algorithm 2).

The same flat-array layout is shared with ``site/src/lib/nn/trees.ts`` so tree models can run
live in a browser tab with exact Python-to-TypeScript parity.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

import numpy as np


@dataclass
class TreeEnsemble:
    """Self-contained gradient-boosted tree ensemble in flat numpy arrays.

    Node indices ``left[u]`` and ``right[u]`` are local to each tree (0 .. n_nodes_t - 1), with
    ``feat[u] == -1`` marking a leaf node. ``cover[u]`` stores the training sample weight / count
    flowing through node ``u`` (required for exact path-dependent TreeSHAP).
    """

    tree_offsets: np.ndarray        # int32 [n_trees + 1]
    feat: np.ndarray                # int32 [total_nodes], -1 for leaf
    thresh: np.ndarray              # float64 [total_nodes]
    default_left: np.ndarray        # uint8 [total_nodes], 1 if NaN -> left
    left: np.ndarray                # int32 [total_nodes], local node index or -1
    right: np.ndarray               # int32 [total_nodes], local node index or -1
    value: np.ndarray               # float64 [total_nodes]
    cover: np.ndarray               # float64 [total_nodes]
    base_score: np.ndarray          # float64 [n_outputs]
    tree_output: np.ndarray         # int32 [n_trees]
    n_features: int
    n_outputs: int = 1
    objective: str = "regression"   # "regression" | "binary" | "multiclass"
    feature_names: tuple[str, ...] | None = None
    monotone_cst: tuple[int, ...] | None = None

    @property
    def n_trees(self) -> int:
        return int(len(self.tree_offsets) - 1)

    @property
    def n_nodes(self) -> int:
        return int(len(self.feat))

    @property
    def n_leaves(self) -> int:
        return int(np.sum(self.feat < 0))

    @property
    def n_params(self) -> int:
        """Compact parameter budget: internal splits (feature + threshold) + leaf values."""
        n_internal = self.n_nodes - self.n_leaves
        return int(2 * n_internal + self.n_leaves)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Evaluate raw margin predictions in pure numpy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        np.ndarray of shape (n_samples,) if n_outputs == 1, else (n_samples, n_outputs).
        """
        X2 = np.asarray(X, dtype=np.float64)
        single = X2.ndim == 1
        if single:
            X2 = X2[None, :]
        n = X2.shape[0]
        if X2.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X2.shape[1]}")

        out = np.broadcast_to(self.base_score[None, :], (n, self.n_outputs)).copy()
        row_idx = np.arange(n)

        for t in range(self.n_trees):
            s = int(self.tree_offsets[t])
            e = int(self.tree_offsets[t + 1])
            f_t = self.feat[s:e]
            th_t = self.thresh[s:e]
            dl_t = self.default_left[s:e]
            l_t = self.left[s:e]
            r_t = self.right[s:e]
            v_t = self.value[s:e]

            nodes = np.zeros(n, dtype=np.int32)
            while True:
                f = f_t[nodes]
                active = f >= 0
                if not np.any(active):
                    break
                act_rows = row_idx[active]
                act_nodes = nodes[active]
                vals = X2[act_rows, f[active]]
                is_nan = np.isnan(vals)
                go_left = np.where(is_nan, dl_t[act_nodes] != 0, vals <= th_t[act_nodes])
                nodes[active] = np.where(go_left, l_t[act_nodes], r_t[act_nodes])

            out[:, int(self.tree_output[t])] += v_t[nodes]

        if self.n_outputs == 1:
            return float(out[0, 0]) if single else out[:, 0]
        return out[0] if single else out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict in target space (identity / sigmoid / softmax)."""
        raw = self.predict_raw(X)
        if self.objective == "binary":
            arr = np.asarray(raw, dtype=np.float64)
            p = 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))
            return float(p) if np.ndim(raw) == 0 else p
        if self.objective == "multiclass":
            arr = np.asarray(raw, dtype=np.float64)
            z = arr - np.max(arr, axis=-1, keepdims=True)
            e = np.exp(z)
            return e / np.sum(e, axis=-1, keepdims=True)
        return raw

    def expected_value(self, output_idx: int = 0) -> float:
        """Exact training-weighted root expectation (base_score + sum of tree root values)."""
        ev = float(self.base_score[output_idx])
        for t in range(self.n_trees):
            if int(self.tree_output[t]) != output_idx:
                continue
            s = int(self.tree_offsets[t])
            e = int(self.tree_offsets[t + 1])
            ev += _tree_expected_value(
                self.feat[s:e], self.left[s:e], self.right[s:e],
                self.value[s:e], self.cover[s:e],
            )
        return ev

    def treeshap(self, X: np.ndarray, output_idx: int = 0) -> tuple[np.ndarray, float]:
        """Exact path-dependent TreeSHAP (Lundberg et al. 2020, Algorithm 2) in pure numpy.

        Satisfies the exact Shapley efficiency axiom:
            ``expected_value + phi.sum(axis=1) == predict_raw(X)`` (for ``output_idx``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)
        output_idx : output / class index (default 0)

        Returns
        -------
        phi : np.ndarray of shape (n_samples, n_features) (or (n_features,) for 1-D input)
        expected_value : float
        """
        X2 = np.asarray(X, dtype=np.float64)
        single = X2.ndim == 1
        if single:
            X2 = X2[None, :]
        n = X2.shape[0]
        if X2.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X2.shape[1]}")

        phi = np.zeros((n, self.n_features), dtype=np.float64)
        ev = float(self.base_score[output_idx])
        batch_size = 2048

        for t in range(self.n_trees):
            if int(self.tree_output[t]) != output_idx:
                continue
            s = int(self.tree_offsets[t])
            e = int(self.tree_offsets[t + 1])
            f_t = self.feat[s:e]
            th_t = self.thresh[s:e]
            dl_t = self.default_left[s:e]
            l_t = self.left[s:e]
            r_t = self.right[s:e]
            v_t = self.value[s:e]
            c_t = self.cover[s:e]
            ev += _tree_expected_value(f_t, l_t, r_t, v_t, c_t)
            f_list = f_t.tolist()
            th_list = th_t.tolist()
            dl_list = dl_t.tolist()
            l_list = l_t.tolist()
            r_list = r_t.tolist()
            v_list = v_t.tolist()
            c_list = c_t.tolist()
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                _treeshap_tree_batch(
                    X2[start:end], f_list, th_list, dl_list, l_list, r_list, v_list, c_list, phi[start:end]
                )

        return (phi[0], ev) if single else (phi, ev)

    def save_npz(self, path: str) -> None:
        """Serialize the ensemble into a compact compressed ``.npz`` file."""
        meta = {
            "n_features": self.n_features,
            "n_outputs": self.n_outputs,
            "objective": self.objective,
            "feature_names": list(self.feature_names) if self.feature_names is not None else None,
            "monotone_cst": list(self.monotone_cst) if self.monotone_cst is not None else None,
        }
        np.savez_compressed(
            path,
            tree_offsets=self.tree_offsets.astype(np.int32),
            feat=self.feat.astype(np.int32),
            thresh=self.thresh.astype(np.float64),
            default_left=self.default_left.astype(np.uint8),
            left=self.left.astype(np.int32),
            right=self.right.astype(np.int32),
            value=self.value.astype(np.float64),
            cover=self.cover.astype(np.float64),
            base_score=self.base_score.astype(np.float64),
            tree_output=self.tree_output.astype(np.int32),
            meta_json=np.array(json.dumps(meta)),
        )

    @classmethod
    def load_npz(cls, path: str) -> "TreeEnsemble":
        """Load a ``TreeEnsemble`` previously written by :meth:`save_npz`."""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            fnames = tuple(meta["feature_names"]) if meta.get("feature_names") is not None else None
            mcst = tuple(int(v) for v in meta["monotone_cst"]) if meta.get("monotone_cst") is not None else None
            return cls(
                tree_offsets=z["tree_offsets"].astype(np.int32),
                feat=z["feat"].astype(np.int32),
                thresh=z["thresh"].astype(np.float64),
                default_left=z["default_left"].astype(np.uint8),
                left=z["left"].astype(np.int32),
                right=z["right"].astype(np.int32),
                value=z["value"].astype(np.float64),
                cover=z["cover"].astype(np.float64),
                base_score=z["base_score"].astype(np.float64),
                tree_output=z["tree_output"].astype(np.int32),
                n_features=int(meta["n_features"]),
                n_outputs=int(meta.get("n_outputs", 1)),
                objective=str(meta.get("objective", "regression")),
                feature_names=fnames,
                monotone_cst=mcst,
            )

    def to_pack_dict(self) -> tuple[dict[str, np.ndarray], dict]:
        """Return ``(tensors, meta)`` for ``mlratchet.webexport`` / ``site/src/lib/nn/trees.ts``."""
        tensors = {
            "tree_offsets": self.tree_offsets.astype(np.int32),
            "feat": self.feat.astype(np.int32),
            "thresh": self.thresh.astype(np.float32),
            "default_left": self.default_left.astype(np.uint8),
            "left": self.left.astype(np.int32),
            "right": self.right.astype(np.int32),
            "value": self.value.astype(np.float32),
            "cover": self.cover.astype(np.float32),
            "base_score": self.base_score.astype(np.float32),
            "tree_output": self.tree_output.astype(np.int32),
        }
        meta = {
            "n_trees": self.n_trees,
            "n_nodes": self.n_nodes,
            "n_features": self.n_features,
            "n_outputs": self.n_outputs,
            "objective": self.objective,
            "feature_names": list(self.feature_names) if self.feature_names is not None else None,
            "monotone_cst": list(self.monotone_cst) if self.monotone_cst is not None else None,
        }
        return tensors, meta


def group_shap(
    phi: np.ndarray,
    groups: Mapping[str, Sequence[int]],
) -> tuple[list[str], np.ndarray]:
    """Sum exact per-feature SHAP values into named physical feature groups.

    Because Shapley values are additive, grouping fine-grained features (for example, all
    aerodynamic work columns in EnergyNet) preserves exact efficiency when ``groups`` partitions
    all features.
    """
    arr = np.asarray(phi, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    names = list(groups.keys())
    out = np.zeros((arr.shape[0], len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        idx = list(groups[name])
        if idx:
            out[:, j] = np.sum(arr[:, idx], axis=1)
    return names, (out[0] if single else out)


def from_lightgbm(
    booster,
    feature_names: Sequence[str] | None = None,
    monotone_cst: Sequence[int] | None = None,
) -> TreeEnsemble:
    """Export a trained LightGBM ``Booster`` (or sklearn wrapper) to a pure-numpy ``TreeEnsemble``."""
    b = booster.booster_ if hasattr(booster, "booster_") else booster
    dump = b.dump_model()
    obj_str = str(dump.get("objective", "regression")).lower()
    if "multiclass" in obj_str:
        objective = "multiclass"
    elif "binary" in obj_str:
        objective = "binary"
    else:
        objective = "regression"

    n_features = int(dump["max_feature_idx"]) + 1
    n_outputs = int(dump.get("num_tree_per_iteration", 1))
    if feature_names is None and dump.get("feature_names"):
        fnames = tuple(str(s) for s in dump["feature_names"])
    elif feature_names is not None:
        fnames = tuple(str(s) for s in feature_names)
    else:
        fnames = tuple(f"f{i}" for i in range(n_features))

    mcst = tuple(int(v) for v in monotone_cst) if monotone_cst is not None else None

    tree_offsets = [0]
    feat: list[int] = []
    thresh: list[float] = []
    default_left: list[int] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []
    cover: list[float] = []
    tree_output: list[int] = []

    for idx, tinfo in enumerate(dump["tree_info"]):
        t_out = int(tinfo["tree_index"]) % n_outputs
        tree_output.append(t_out)
        # Flatten tree in pre-order with local node indices
        local_feat: list[int] = []
        local_thresh: list[float] = []
        local_dl: list[int] = []
        local_left: list[int] = []
        local_right: list[int] = []
        local_val: list[float] = []
        local_cov: list[float] = []

        def _visit(node: dict) -> int:
            u = len(local_feat)
            local_feat.append(-1)
            local_thresh.append(0.0)
            local_dl.append(1)
            local_left.append(-1)
            local_right.append(-1)
            local_val.append(0.0)
            local_cov.append(0.0)

            if "split_index" in node:
                if node.get("decision_type", "<=") not in ("<=", "<"):
                    raise ValueError(f"unsupported LightGBM decision_type: {node.get('decision_type')}")
                local_feat[u] = int(node["split_feature"])
                local_thresh[u] = float(node["threshold"])
                local_dl[u] = 1 if bool(node.get("default_left", True)) else 0
                local_val[u] = float(node.get("internal_value", 0.0))
                local_cov[u] = float(node.get("internal_count", node.get("internal_weight", 1.0)))
                lc = _visit(node["left_child"])
                rc = _visit(node["right_child"])
                local_left[u] = lc
                local_right[u] = rc
            else:
                local_val[u] = float(node["leaf_value"])
                local_cov[u] = float(node.get("leaf_count", node.get("leaf_weight", 1.0)))
            return u

        _visit(tinfo["tree_structure"])
        feat.extend(local_feat)
        thresh.extend(local_thresh)
        default_left.extend(local_dl)
        left.extend(local_left)
        right.extend(local_right)
        value.extend(local_val)
        cover.extend(local_cov)
        tree_offsets.append(len(feat))

    return TreeEnsemble(
        tree_offsets=np.asarray(tree_offsets, dtype=np.int32),
        feat=np.asarray(feat, dtype=np.int32),
        thresh=np.asarray(thresh, dtype=np.float64),
        default_left=np.asarray(default_left, dtype=np.uint8),
        left=np.asarray(left, dtype=np.int32),
        right=np.asarray(right, dtype=np.int32),
        value=np.asarray(value, dtype=np.float64),
        cover=np.asarray(cover, dtype=np.float64),
        base_score=np.zeros(n_outputs, dtype=np.float64),
        tree_output=np.asarray(tree_output, dtype=np.int32),
        n_features=n_features,
        n_outputs=n_outputs,
        objective=objective,
        feature_names=fnames,
        monotone_cst=mcst,
    )


# ---------------------------------------------------------------------------
# Exact path-dependent TreeSHAP (Lundberg et al. 2020, Algorithm 2)
# ---------------------------------------------------------------------------

def _tree_expected_value(
    feat: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    value: np.ndarray,
    cover: np.ndarray,
) -> float:
    """Exact cover-weighted expected leaf value of a single tree."""
    def _rec(u: int) -> float:
        if feat[u] < 0:
            return float(value[u])
        lc = int(left[u])
        rc = int(right[u])
        c_tot = float(cover[lc] + cover[rc])
        if c_tot <= 0.0:
            return 0.5 * (_rec(lc) + _rec(rc))
        return (float(cover[lc]) * _rec(lc) + float(cover[rc]) * _rec(rc)) / c_tot
    return _rec(0)


def _extend_vec(
    m_d: list[int],
    m_z: list[float],
    m_o: list[np.ndarray],
    m_w: np.ndarray,
    pz: float,
    po: np.ndarray,
    pi: int,
) -> tuple[list[int], list[float], list[np.ndarray], np.ndarray]:
    """Batch-vectorized EXTEND routine of Lundberg et al. 2020 Algorithm 2."""
    L = len(m_d)
    n = po.shape[0]
    d = m_d + [pi]
    z = m_z + [pz]
    o = m_o + [po]
    w = np.zeros((L + 1, n), dtype=np.float64)
    if L == 0:
        w[0] = 1.0
        return d, z, o, w
    w[:L] = m_w
    inv = 1.0 / (L + 1)
    for i in range(L - 1, -1, -1):
        wi = w[i]
        w[i + 1] += po * wi * ((i + 1) * inv)
        w[i] = (pz * (L - i) * inv) * wi
    return d, z, o, w


def _unwind_vec(
    m_d: list[int],
    m_z: list[float],
    m_o: list[np.ndarray],
    m_w: np.ndarray,
    idx: int,
) -> tuple[list[int], list[float], list[np.ndarray], np.ndarray]:
    """Batch-vectorized UNWIND routine of Lundberg et al. 2020 Algorithm 2."""
    L = len(m_d) - 1
    n = m_w.shape[1]
    d = m_d[:idx] + m_d[idx + 1:]
    z = m_z[:idx] + m_z[idx + 1:]
    o = m_o[:idx] + m_o[idx + 1:]
    oi = m_o[idx]
    zi = m_z[idx]
    mask_nz = oi != 0.0
    any_nz = bool(np.any(mask_nz))
    any_z = bool(np.any(~mask_nz))

    if any_nz:
        safe_oi = np.where(mask_nz, oi, 1.0)
        cur = m_w[L].copy()
        w_nz = np.empty((L, n), dtype=np.float64)
        for j in range(L - 1, -1, -1):
            wj = cur * ((L + 1) / (j + 1)) / safe_oi
            w_nz[j] = wj
            cur = m_w[j] - wj * (zi * (L - j) / (L + 1))
        if not any_z:
            return d, z, o, w_nz
    if any_z:
        if zi != 0.0:
            scale = (L + 1) / (zi * np.arange(L, 0, -1, dtype=np.float64))
            w_z = m_w[:L] * scale[:, None]
        else:
            w_z = np.zeros((L, n), dtype=np.float64)
        if not any_nz:
            return d, z, o, w_z
    return d, z, o, np.where(mask_nz[None, :], w_nz, w_z)


def _unwound_path_sum_vec(
    m_z: list[float],
    m_o: list[np.ndarray],
    m_w: np.ndarray,
    idx: int,
) -> np.ndarray:
    """Batch-vectorized UNWOUND_PATH_SUM routine of Lundberg et al. 2020 Algorithm 2."""
    L = m_w.shape[0] - 1
    n = m_w.shape[1]
    oi = m_o[idx]
    zi = m_z[idx]
    mask_nz = oi != 0.0
    any_nz = bool(np.any(mask_nz))
    any_z = bool(np.any(~mask_nz))

    if any_nz:
        safe_oi = np.where(mask_nz, oi, 1.0)
        cur = m_w[L]
        tot_nz = np.zeros(n, dtype=np.float64)
        for j in range(L - 1, -1, -1):
            tmp = cur / ((j + 1) * safe_oi)
            tot_nz += tmp
            cur = m_w[j] - tmp * (zi * (L - j))
        tot_nz *= (L + 1)
        if not any_z:
            return tot_nz
    if any_z:
        if zi != 0.0:
            coeffs = (L + 1) / (zi * np.arange(L, 0, -1, dtype=np.float64))
            tot_z = coeffs @ m_w[:L]
        else:
            tot_z = np.zeros(n, dtype=np.float64)
        if not any_nz:
            return tot_z
    return np.where(mask_nz, tot_nz, tot_z)


def _treeshap_tree_batch(
    X: np.ndarray,
    feat: list[int],
    thresh: list[float],
    default_left: list[int],
    left: list[int],
    right: list[int],
    value: list[float],
    cover: list[float],
    phi_out: np.ndarray,
) -> None:
    """Accumulate exact TreeSHAP contributions of one tree for a batch ``X`` into ``phi_out``."""
    n = X.shape[0]
    ones = np.ones(n, dtype=np.float64)

    def _recurse(
        u: int,
        m_d: list[int],
        m_z: list[float],
        m_o: list[np.ndarray],
        m_w: np.ndarray,
        pz: float,
        po: np.ndarray,
        pi: int,
    ) -> None:
        d, z, o, w = _extend_vec(m_d, m_z, m_o, m_w, pz, po, pi)
        f = feat[u]
        if f < 0:
            v = value[u]
            for i in range(1, len(d)):
                s = _unwound_path_sum_vec(z, o, w, i)
                phi_out[:, d[i]] += s * (o[i] - z[i]) * v
            return

        xv = X[:, f]
        isnan = np.isnan(xv)
        go_left = np.where(isnan, default_left[u] != 0, xv <= thresh[u]).astype(np.float64)

        iz = 1.0
        io = ones
        k = -1
        for i in range(1, len(d)):
            if d[i] == f:
                k = i
                break
        if k >= 0:
            iz = z[k]
            io = o[k]
            d, z, o, w = _unwind_vec(d, z, o, w, k)

        lc = left[u]
        rc = right[u]
        c_l = cover[lc]
        c_r = cover[rc]
        c_u = c_l + c_r
        if c_u <= 0.0:
            f_l = 0.5
            f_r = 0.5
        else:
            f_l = c_l / c_u
            f_r = c_r / c_u

        _recurse(lc, d, z, o, w, iz * f_l, io * go_left, f)
        _recurse(rc, d, z, o, w, iz * f_r, io * (1.0 - go_left), f)

    _recurse(0, [], [], [], np.empty((0, n), dtype=np.float64), 1.0, ones, -1)
