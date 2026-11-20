"""Research-only Python API for FFN activation capture and similarity.

Mirrors ``sycl_infer_research.h``.  Kept in a separate module so a normal
inference consumer never imports it; import explicitly:

    from sycl_infer.research import capture_ffn, FFNPattern, similarity

The three layers map to the C API:

* :func:`capture_ffn`  -> ``syi_model_capture_ffn`` (needs a :class:`Model`)
* :class:`FFNPattern`  -> ``syi_ffn_pattern_create`` / ``_top`` (no model needed)
* :func:`similarity`   -> ``syi_ffn_similarity`` (0-1 matrix, no model needed)

The method: per layer, z-score each FFN neuron across the batch of texts, take
each text's top-K neurons; pairwise overlap enriched over chance K/dim maps to
a [0,1] score averaged over layers.  See ``research/ffn_pattern.cpp``.
"""
from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from . import _capi
from ._capi import _FFNOptions, _FFNPatternOptions, _FFNSimilarityOptions

__all__ = [
    "FFNActivations",
    "FFNPattern",
    "capture_ffn",
    "concept_similarity",
    "similarity",
]


def _check(status: int) -> int:
    if status < 0:
        msg = _capi.last_error() or _capi.status_string(status)
        raise RuntimeError(msg)
    return status


@dataclass
class FFNActivations:
    """Flat FFN gate|up activations from capture_ffn.

    ``data`` is a flat list of ``n_texts * n_layers * 2 * n_ff`` floats,
    laid out ``[text][layer][gate_half || up_half]``.
    """

    data: list[float]
    n_texts: int
    n_layers: int
    n_ff: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.n_texts, self.n_layers, 2, self.n_ff)


class FFNPattern:
    """Top-K activated neurons per text/layer (layer 1 of the research API)."""

    def __init__(self, activations: FFNActivations, *, top_k: int = 0,
                 half: int = 0, zscore: bool = True):
        if half not in (0, 1, 2):
            raise ValueError("half must be 0 (gate), 1 (up), or 2 (both)")
        lib = _capi.load()
        o = _FFNPatternOptions()
        lib.syi_ffn_pattern_options_init(ctypes.byref(o))
        o.top_k = int(top_k)
        o.half = int(half)
        o.zscore = 1 if zscore else 0
        arr = (ctypes.c_float * len(activations.data))(*activations.data)
        h = ctypes.c_void_p()
        _check(lib.syi_ffn_pattern_create(arr, activations.n_texts, activations.n_layers,
                                          activations.n_ff, ctypes.byref(o), ctypes.byref(h)))
        self._h = h
        # keep the buffer alive for the duration of the C call (it copies).
        self._buf = arr

    def __enter__(self) -> "FFNPattern":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._h:
            _capi.load().syi_ffn_pattern_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def top(self):
        """Return ``(ids, z_values, n_texts, n_layers, top_k, dim)``.

        ``ids`` and ``z_values`` are nested ``[text][layer][k]`` lists.
        """
        lib = _capi.load()
        pi = ctypes.POINTER(ctypes.c_int32)()
        pf = ctypes.POINTER(ctypes.c_float)()
        nt = ctypes.c_size_t()
        nl = ctypes.c_size_t()
        k = ctypes.c_size_t()
        dim = ctypes.c_size_t()
        _check(lib.syi_ffn_pattern_top(self._h, ctypes.byref(pi), ctypes.byref(pf),
                                       ctypes.byref(nt), ctypes.byref(nl),
                                       ctypes.byref(k), ctypes.byref(dim)))
        try:
            ids = [[[int(pi[(t * nl.value + l) * k.value + i]) for i in range(k.value)]
                    for l in range(nl.value)] for t in range(nt.value)]
            z = [[[float(pf[(t * nl.value + l) * k.value + i]) for i in range(k.value)]
                  for l in range(nl.value)] for t in range(nt.value)]
            return ids, z, nt.value, nl.value, k.value, dim.value
        finally:
            if pi:
                lib.syi_free(pi)
            if pf:
                lib.syi_free(pf)


def capture_ffn(model, texts: Sequence[str], *, layer: int = -1, token_slot: int = -1) -> FFNActivations:
    """Capture FFN [gate|up] activations for a batch of texts (layer 0 of the API).

    ``layer`` is 0..n_layer-1 or -1 for all layers; ``token_slot`` is the token
    index or -1 for the last real token.  Each text must fit in one prefill
    chunk (<=32 tokens).  Do not submit generation requests on ``model`` during
    the capture.
    """
    lib = _capi.load()
    o = _FFNOptions()
    lib.syi_ffn_options_init(ctypes.byref(o))
    o.layer = int(layer)
    o.token_slot = int(token_slot)
    encoded = [t.encode("utf-8") for t in texts]
    arr = (ctypes.c_char_p * len(encoded))(*encoded)
    out = ctypes.POINTER(ctypes.c_float)()
    nt = ctypes.c_size_t()
    nl = ctypes.c_size_t()
    nf = ctypes.c_size_t()
    _check(lib.syi_model_capture_ffn(model._handle(), ctypes.byref(o), arr, len(encoded),
                                     ctypes.byref(out), ctypes.byref(nt),
                                     ctypes.byref(nl), ctypes.byref(nf)))
    try:
        n = nt.value * nl.value * 2 * nf.value
        data = [float(out[i]) for i in range(n)]
        return FFNActivations(data, nt.value, nl.value, nf.value)
    finally:
        if out:
            lib.syi_free(out)


def similarity(pattern: FFNPattern, *, layers: Optional[Sequence[int]] = None,
               chance_baseline: float = 0.0) -> list[list[float]]:
    """Compute the [0,1] similarity matrix from a pattern (layer 2 of the API).

    ``layers`` selects a subset of layer indices (None = all); ``chance_baseline``
    defaults to the theoretical K/dim.  Returns an ``n x n`` symmetric matrix
    with a diagonal of exactly 1.
    """
    lib = _capi.load()
    o = _FFNSimilarityOptions()
    lib.syi_ffn_similarity_options_init(ctypes.byref(o))
    layer_arr = None
    if layers is not None:
        layer_arr = (ctypes.c_int32 * len(layers))(*[int(l) for l in layers])
        o.layers = layer_arr
        o.n_layers = len(layers)
    o.chance_baseline = float(chance_baseline)
    out = ctypes.POINTER(ctypes.c_double)()
    n = ctypes.c_size_t()
    _check(lib.syi_ffn_similarity(pattern._h, ctypes.byref(o), ctypes.byref(out), ctypes.byref(n)))
    try:
        nv = n.value
        return [[float(out[i * nv + j]) for j in range(nv)] for i in range(nv)]
    finally:
        if out:
            lib.syi_free(out)


def concept_similarity(model, texts: Sequence[str], *, layer: int = 5, token_slot: int = -1,
                       top_k: int = 0, half: int = 0, zscore: bool = True,
                       layers: Optional[Sequence[int]] = None,
                       chance_baseline: float = 0.0):
    """Convenience: capture -> pattern -> similarity in one call.

    Returns ``(matrix, activations, pattern_ids)``.  Keep the pattern alive via
    the returned ids (the underlying C handle is closed on return).
    """
    activations = capture_ffn(model, texts, layer=layer, token_slot=token_slot)
    with FFNPattern(activations, top_k=top_k, half=half, zscore=zscore) as p:
        ids, _z, _nt, _nl, _k, _dim = p.top()
        mat = similarity(p, layers=layers, chance_baseline=chance_baseline)
    return mat, activations, ids
