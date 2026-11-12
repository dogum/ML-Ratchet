"""Export tensors for the in-browser instruments.

A *pack* is two files: ``<name>.json`` (a manifest) and ``<name>.bin`` (little-endian tensor data,
each tensor 8-byte aligned). The site's TypeScript runtime (site/src/lib/nn/pack.ts) reads it.

    from mlratchet.webexport import write_pack
    write_pack(out_dir / "weights", {k: v for k, v in npz.items()}, dtype="f16", meta=meta)

float16 halves the download; the parity test (site/tests) checks the browser forward pass against
Python on the same float16-rounded weights, so the comparison is exact in what it measures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DTYPES = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "u8": np.uint8}


def write_pack(prefix: str | Path, tensors: Mapping[str, np.ndarray], dtype: str = "f32",
               meta: Any = None, keep: Mapping[str, str] | None = None) -> dict:
    """Write ``prefix.json`` + ``prefix.bin``. ``keep`` overrides the dtype per tensor (e.g. labels as i32)."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"format": "mlratchet-pack/1", "tensors": {}, "meta": meta}
    chunks, offset = [], 0
    for name, arr in tensors.items():
        dt = (keep or {}).get(name, dtype)
        a = np.ascontiguousarray(np.asarray(arr), dtype=DTYPES[dt]).astype(np.dtype(DTYPES[dt]).newbyteorder("<"), copy=False)
        b = a.tobytes()
        pad = (-offset) % 8
        if pad:
            chunks.append(b"\0" * pad)
            offset += pad
        manifest["tensors"][name] = {"dtype": dt, "shape": list(a.shape), "offset": offset, "bytes": len(b)}
        chunks.append(b)
        offset += len(b)
    prefix.with_suffix(".bin").write_bytes(b"".join(chunks))
    prefix.with_suffix(".json").write_text(json.dumps(manifest, separators=(",", ":"), default=_json_default), encoding="utf-8")
    return manifest


def roundtrip(tensors: Mapping[str, np.ndarray], dtype: str = "f16") -> dict[str, np.ndarray]:
    """The values the browser will see, as float32 (use these for the reference outputs)."""
    return {k: np.asarray(v).astype(DTYPES[dtype]).astype(np.float32) for k, v in tensors.items()}


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, separators=(",", ":"), default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def instrument_dir(model_id: str) -> Path:
    """site/public/instruments/<id>/ — where a model's live-instrument data goes."""
    root = Path(__file__).resolve().parents[1]
    d = root / "site" / "public" / "instruments" / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d
