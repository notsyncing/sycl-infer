"""ctypes bindings for the sycl-infer C ABI.

The shared library is loaded lazily.  The wheel ships the engine library and
its runtime dependencies in ``sycl_infer/_lib``; a source checkout can point
``SYCL_INFER_LIB`` at ``build-rel2/libsycl_infer.so.1.0.0``.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

_lib = None
_lib_path: Optional[str] = None


class _SString(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char_p), ("size", ctypes.c_size_t)]


class _ModelOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("abi_version", ctypes.c_uint32),
        ("model_path", ctypes.c_char_p),
        ("layer_map", ctypes.c_char_p),
        ("device", ctypes.c_int),
        ("max_seq", ctypes.c_int32),
        ("max_slots", ctypes.c_int32),
        ("n_blocks", ctypes.c_int32),
        ("kv_cap_mb", ctypes.c_int32),
        ("kv_type", ctypes.c_int),
        ("kv_type_v", ctypes.c_int),
        ("cpu_threads", ctypes.c_int32),
        ("n_splits", ctypes.c_int32),
        ("dec_splits", ctypes.c_int32),
        ("batch_window_us", ctypes.c_int32),
        ("max_batch_rows", ctypes.c_int32),
        ("max_queue_depth", ctypes.c_int32),
        ("max_queue_tokens", ctypes.c_int64),
        ("request_timeout_ms", ctypes.c_int32),
        ("prefix_cache", ctypes.c_int32),
        ("pc_dir", ctypes.c_char_p),
        ("pc_vram_mb", ctypes.c_int32),
        ("pc_ram_mb", ctypes.c_int32),
        ("pc_disk_mb", ctypes.c_int32),
        ("moe_resident", ctypes.c_int32),
        ("moe_vram_mb", ctypes.c_int64),
        ("moe_ram_mb", ctypes.c_int64),
        ("engram_ram_mb", ctypes.c_int64),
        ("yarn", ctypes.c_int32),
        ("yarn_orig_ctx", ctypes.c_int32),
        ("yarn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("head_split", ctypes.c_int32),
        ("flags", ctypes.c_uint64),
        ("log_fn", ctypes.c_void_p),
        ("log_user_data", ctypes.c_void_p),
        ("reserved", ctypes.c_void_p * 8),
    ]


class _ModelInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("model_name", _SString),
        ("architecture", _SString),
        ("device_name", _SString),
        ("n_vocab", ctypes.c_int32),
        ("n_ctx", ctypes.c_int32),
        ("n_embd", ctypes.c_int32),
        ("n_layer", ctypes.c_int32),
        ("n_head", ctypes.c_int32),
        ("n_head_kv", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("is_moe", ctypes.c_int32),
        ("is_split_gguf", ctypes.c_int32),
        ("max_seq", ctypes.c_int32),
        ("max_slots", ctypes.c_int32),
        ("kv_type", ctypes.c_int),
        ("kv_type_v", ctypes.c_int),
        ("device", ctypes.c_int),
    ]


class _ModelStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("submitted", ctypes.c_uint64),
        ("admitted", ctypes.c_uint64),
        ("completed", ctypes.c_uint64),
        ("cancelled", ctypes.c_uint64),
        ("failed", ctypes.c_uint64),
        ("generated_tokens", ctypes.c_uint64),
        ("prefill_steps", ctypes.c_uint64),
        ("decode_steps", ctypes.c_uint64),
        ("avg_decode_batch", ctypes.c_double),
        ("max_decode_batch", ctypes.c_uint32),
        ("queued", ctypes.c_int32),
        ("active", ctypes.c_int32),
        ("free_slots", ctypes.c_int32),
    ]


class _GenerationOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("abi_version", ctypes.c_uint32),
        ("max_tokens", ctypes.c_int32),
        ("ignore_eos", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("min_p", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("repeat_last_n", ctypes.c_int32),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("seed", ctypes.c_uint64),
        ("logprobs", ctypes.c_int32),
        ("top_logprobs", ctypes.c_int32),
        ("timeout_ms", ctypes.c_int32),
        ("logit_bias_ids", ctypes.POINTER(ctypes.c_int32)),
        ("logit_bias_values", ctypes.POINTER(ctypes.c_float)),
        ("n_logit_bias", ctypes.c_size_t),
        ("stop", ctypes.POINTER(ctypes.c_char_p)),
        ("n_stop", ctypes.c_size_t),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_void_p * 8),
    ]


class _ChatMessage(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("role", ctypes.c_char_p),
        ("content", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("tool_call_id", ctypes.c_char_p),
        ("reasoning_content", ctypes.c_char_p),
    ]


class _Logprob(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int32), ("logprob", ctypes.c_float), ("text", _SString)]


class _Token(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("id", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("logprob", ctypes.c_float),
        ("text", _SString),
        ("top", ctypes.POINTER(_Logprob)),
        ("n_top", ctypes.c_size_t),
        ("index", ctypes.c_uint64),
    ]


class _Result(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("text", _SString),
        ("token_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
        ("finish_reason", ctypes.c_int),
        ("prompt_tokens", ctypes.c_uint64),
        ("completion_tokens", ctypes.c_uint64),
        ("cached_tokens", ctypes.c_uint64),
        ("ttft_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("error", _SString),
    ]


def _lib_dir() -> Path:
    return Path(__file__).resolve().parent / "_lib"


def _candidate_paths() -> Iterable[str]:
    env = os.environ.get("SYCL_INFER_LIB")
    if env:
        yield env
    d = _lib_dir()
    for name in ("libsycl_infer.so.1.0.0", "libsycl_infer.so.1", "libsycl_infer.so",
                 "libsycl_infer.dylib", "sycl_infer.dll"):
        p = d / name
        if p.exists():
            yield str(p)
    found = ctypes.util.find_library("sycl_infer")
    if found:
        yield found


def _preload() -> None:
    """Load bundled dependencies with RTLD_GLOBAL.

    The wheel's ``_lib`` is self-contained, but the engine's DSOs do not carry
    an rpath that works for every loader, so loading the dependency closure
    explicitly is more robust than relying on $ORIGIN.  Failures are ignored:
    a system-installed oneAPI runtime can satisfy the same NEEDED entries.
    """
    d = _lib_dir()
    if not d.is_dir():
        return
    preferred = [
        "libsvml.so", "libimf.so", "libintlc.so.5", "libirng.so",
        "libtbb.so.12", "libOpenCL.so.1", "libze_loader.so.1",
        "libdnnl.so.3", "libsycl.so.9",
    ]
    seen = set()
    paths = []
    for name in preferred:
        p = d / name
        if p.exists():
            paths.append(p)
            seen.add(name)
    for p in sorted(d.glob("*.so*")):
        if p.name not in seen and not p.name.startswith("libsycl_infer"):
            paths.append(p)
    for p in paths:
        try:
            ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def load():
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    if sys.platform == "win32":
        mode = 0
    else:
        mode = ctypes.RTLD_GLOBAL
    last = None
    for path in _candidate_paths():
        try:
            _preload()
            _lib = ctypes.CDLL(path, mode=mode)
            _lib_path = path
            break
        except OSError as exc:
            last = exc
    if _lib is None:
        raise ImportError(
            "cannot load libsycl_infer; set SYCL_INFER_LIB or install the wheel "
            f"with its bundled runtime ({last})"
        )
    _declare(_lib)
    return _lib


def lib_path() -> str:
    load()
    return _lib_path or ""


def _declare(lib) -> None:
    P = ctypes.POINTER
    lib.syi_abi_version.restype = ctypes.c_uint32
    lib.syi_version_string.restype = ctypes.c_char_p
    lib.syi_status_string.argtypes = [ctypes.c_int]
    lib.syi_status_string.restype = ctypes.c_char_p
    lib.syi_last_error.restype = ctypes.c_char_p
    lib.syi_clear_error.argtypes = []
    lib.syi_free.argtypes = [ctypes.c_void_p]

    lib.syi_model_options_init.argtypes = [P(_ModelOptions)]
    lib.syi_model_create.argtypes = [P(_ModelOptions), P(ctypes.c_void_p)]
    lib.syi_model_create.restype = ctypes.c_int
    lib.syi_model_destroy.argtypes = [ctypes.c_void_p]
    lib.syi_model_shutdown.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.syi_model_shutdown.restype = ctypes.c_int
    lib.syi_model_info_init.argtypes = [P(_ModelInfo)]
    lib.syi_model_get_info.argtypes = [ctypes.c_void_p, P(_ModelInfo)]
    lib.syi_model_get_info.restype = ctypes.c_int
    lib.syi_model_stats_init.argtypes = [P(_ModelStats)]
    lib.syi_model_get_stats.argtypes = [ctypes.c_void_p, P(_ModelStats)]
    lib.syi_model_get_stats.restype = ctypes.c_int
    lib.syi_model_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                       ctypes.c_int32, P(P(ctypes.c_int32)), P(ctypes.c_size_t)]
    lib.syi_model_tokenize.restype = ctypes.c_int
    lib.syi_model_detokenize.argtypes = [ctypes.c_void_p, P(ctypes.c_int32), ctypes.c_size_t,
                                         P(ctypes.c_void_p), P(ctypes.c_size_t)]
    lib.syi_model_detokenize.restype = ctypes.c_int
    lib.syi_chat_message_init.argtypes = [P(_ChatMessage)]
    lib.syi_model_render_chat.argtypes = [ctypes.c_void_p, P(_ChatMessage), ctypes.c_size_t,
                                          ctypes.c_int32, ctypes.c_char_p,
                                          P(ctypes.c_void_p), P(ctypes.c_size_t)]
    lib.syi_model_render_chat.restype = ctypes.c_int

    lib.syi_generation_options_init.argtypes = [P(_GenerationOptions)]
    lib.syi_request_create.argtypes = [ctypes.c_void_p, P(_GenerationOptions), P(ctypes.c_void_p)]
    lib.syi_request_create.restype = ctypes.c_int
    lib.syi_request_destroy.argtypes = [ctypes.c_void_p]
    lib.syi_request_set_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.syi_request_set_text.restype = ctypes.c_int
    lib.syi_request_set_tokens.argtypes = [ctypes.c_void_p, P(ctypes.c_int32), ctypes.c_size_t]
    lib.syi_request_set_tokens.restype = ctypes.c_int
    lib.syi_request_set_chat.argtypes = [ctypes.c_void_p, P(_ChatMessage), ctypes.c_size_t,
                                         ctypes.c_int32, ctypes.c_char_p]
    lib.syi_request_set_chat.restype = ctypes.c_int
    lib.syi_request_submit.argtypes = [ctypes.c_void_p]
    lib.syi_request_submit.restype = ctypes.c_int
    lib.syi_submit_batch.argtypes = [ctypes.c_void_p, P(ctypes.c_void_p), ctypes.c_size_t]
    lib.syi_submit_batch.restype = ctypes.c_int
    lib.syi_wait_batch.argtypes = [P(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint32]
    lib.syi_wait_batch.restype = ctypes.c_int
    lib.syi_request_get_state.argtypes = [ctypes.c_void_p, P(ctypes.c_int)]
    lib.syi_request_get_state.restype = ctypes.c_int
    lib.syi_token_init.argtypes = [P(_Token)]
    lib.syi_request_next.argtypes = [ctypes.c_void_p, P(_Token), ctypes.c_uint32]
    lib.syi_request_next.restype = ctypes.c_int
    lib.syi_request_wait.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.syi_request_wait.restype = ctypes.c_int
    lib.syi_request_cancel.argtypes = [ctypes.c_void_p]
    lib.syi_request_cancel.restype = ctypes.c_int
    lib.syi_result_init.argtypes = [P(_Result)]
    lib.syi_request_get_result.argtypes = [ctypes.c_void_p, P(_Result)]
    lib.syi_request_get_result.restype = ctypes.c_int
    lib.syi_request_error.argtypes = [ctypes.c_void_p]
    lib.syi_request_error.restype = ctypes.c_char_p


def status_string(status: int) -> str:
    return load().syi_status_string(status).decode("utf-8", "replace")


def last_error() -> str:
    p = load().syi_last_error()
    return p.decode("utf-8", "replace") if p else ""


def sstr(s: _SString) -> str:
    if not s.data or s.size == 0:
        return ""
    return ctypes.string_at(s.data, s.size).decode("utf-8", "replace")
