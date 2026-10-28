"""Python binding for the sycl-infer C ABI.

The public surface mirrors docs/design/20-python-binding.md.  The v1
implementation uses ctypes (no build-time pybind11 dependency): the C ABI is
the stable boundary, and the wheel carries the engine library plus its
runtime closure.
"""
from __future__ import annotations

import ctypes
from dataclasses import asdict
from typing import Iterable, Optional, Sequence

from . import _capi
from ._capi import _ChatMessage, _GenerationOptions, _Logprob, _ModelInfo, _ModelOptions, _ModelStats, _Result, _Token
from ._types import (
    ChatMessage,
    FinishReason,
    Generation,
    GenerationOptions,
    Logprob,
    ModelInfo,
    ModelStats,
    RequestState,
    Token,
)

__all__ = [
    "Model",
    "Request",
    "ChatMessage",
    "GenerationOptions",
    "Generation",
    "Token",
    "Logprob",
    "ModelInfo",
    "ModelStats",
    "RequestState",
    "FinishReason",
    "SyclInferError",
    "InvalidArgumentError",
    "ModelLoadError",
    "BackendError",
    "OutOfMemoryError",
    "QueueFullError",
    "SyclInferTimeout",
    "CancelledError",
    "UnsupportedError",
    "ClosedError",
    "doctor",
    "__version__",
    "__abi_version__",
]

_INF = 0xFFFFFFFF


class SyclInferError(RuntimeError):
    pass


class InvalidArgumentError(SyclInferError, ValueError):
    pass


class ModelLoadError(SyclInferError):
    pass


class BackendError(SyclInferError):
    pass


class OutOfMemoryError(SyclInferError, MemoryError):
    pass


class QueueFullError(SyclInferError):
    pass


class SyclInferTimeout(SyclInferError, TimeoutError):
    pass


class CancelledError(SyclInferError):
    pass


class UnsupportedError(SyclInferError):
    pass


class ClosedError(SyclInferError):
    pass


_STATUS_ERRORS = {
    -1: InvalidArgumentError,
    -2: SyclInferError,
    -3: UnsupportedError,
    -4: OutOfMemoryError,
    -5: BackendError,
    -6: ModelLoadError,
    -7: SyclInferError,
    -8: SyclInferError,
    -9: QueueFullError,
    -10: ClosedError,
    -11: CancelledError,
}


def _fail(status: int) -> None:
    msg = _capi.last_error() or _capi.status_string(status)
    cls = _STATUS_ERRORS.get(status, SyclInferError)
    raise cls(msg)


def _check(status: int) -> int:
    if status < 0:
        _fail(status)
    return status


def _timeout_ms(timeout: Optional[float]) -> int:
    if timeout is None:
        return _INF
    if timeout < 0:
        return _INF
    return max(0, min(_INF - 1, int(timeout * 1000.0)))


def _kv_code(name: str) -> int:
    return {
        "auto": 0,
        "i8": 1,
        "int8": 1,
        "i4": 2,
        "int4": 2,
        "f16": 3,
        "fp16": 3,
        "bf16": 4,
        "f32": 5,
        "fp32": 5,
    }[name.lower()]


def _kv_pair(spec: str) -> tuple[int, int]:
    if ":" not in spec:
        v = _kv_code(spec)
        return v, v
    a, b = spec.split(":", 1)
    return _kv_code(a), _kv_code(b)


def _kv_name(code: int) -> str:
    return {0: "auto", 1: "i8", 2: "i4", 3: "f16", 4: "bf16", 5: "f32"}.get(code, "auto")


def _device_code(name: str) -> int:
    n = name.lower()
    if n in ("auto", ""):
        return 0
    if n in ("gpu", "gpu.0", "gpu.1"):
        return 1
    if n in ("cpu", "host"):
        return 2
    raise ValueError(f"unknown device {name!r}")


def _device_name(code: int) -> str:
    return {0: "auto", 1: "gpu", 2: "cpu"}.get(code, "auto")


def __getattr__(name: str):
    if name == "__version__":
        return _capi.load().syi_version_string().decode()
    if name == "__abi_version__":
        return int(_capi.load().syi_abi_version())
    raise AttributeError(name)


class Request:
    """One generation.  Single-consumer: next()/wait() must not run concurrently."""

    def __init__(self, model: "Model", handle) -> None:
        self._model = model
        self._h = handle
        self._destroyed = False

    def _handle(self):
        if self._destroyed or not self._h:
            raise ClosedError("request is closed")
        return self._h

    def next(self, timeout: Optional[float] = None) -> Optional[Token]:
        t = _Token()
        _capi.load().syi_token_init(ctypes.byref(t))
        st = _capi.load().syi_request_next(self._handle(), ctypes.byref(t), _timeout_ms(timeout))
        if st == 1:  # SYI_END
            return None
        if st == 2:  # SYI_AGAIN
            raise SyclInferTimeout(_capi.last_error() or "request is still running")
        _check(st)
        text = _capi.sstr(t.text)
        top = []
        if t.n_top and t.top:
            for i in range(t.n_top):
                lp = t.top[i]
                top.append(Logprob(lp.id, lp.logprob, _capi.sstr(lp.text)))
        return Token(id=t.id, text=text, logprob=t.logprob, top=top, index=t.index)

    def wait(self, timeout: Optional[float] = None) -> "Request":
        st = _capi.load().syi_request_wait(self._handle(), _timeout_ms(timeout))
        if st == 2:
            raise SyclInferTimeout(_capi.last_error() or "request is still running")
        _check(st)
        return self

    def result(self) -> Generation:
        r = _Result()
        _capi.load().syi_result_init(ctypes.byref(r))
        _check(_capi.load().syi_request_get_result(self._handle(), ctypes.byref(r)))
        ids = []
        if r.token_ids and r.n_tokens:
            ids = [int(r.token_ids[i]) for i in range(r.n_tokens)]
        fr = {
            0: "none",
            1: "stop",
            2: "length",
            3: "cancelled",
            4: "timeout",
            5: "error",
        }.get(int(r.finish_reason), "unknown")
        return Generation(
            text=_capi.sstr(r.text),
            token_ids=ids,
            finish_reason=fr,
            prompt_tokens=int(r.prompt_tokens),
            completion_tokens=int(r.completion_tokens),
            cached_tokens=int(r.cached_tokens),
            ttft_ms=float(r.ttft_ms),
            total_ms=float(r.total_ms),
            error=_capi.sstr(r.error),
        )

    def cancel(self) -> None:
        if not self._destroyed and self._h:
            _check(_capi.load().syi_request_cancel(self._h))

    @property
    def state(self) -> RequestState:
        s = ctypes.c_int(0)
        _check(_capi.load().syi_request_get_state(self._handle(), ctypes.byref(s)))
        try:
            return RequestState(s.value)
        except ValueError:
            return RequestState.FAILED

    @property
    def error(self) -> str:
        if self._destroyed or not self._h:
            return ""
        p = _capi.load().syi_request_error(self._h)
        return p.decode("utf-8", "replace") if p else ""

    def __iter__(self):
        while True:
            tok = self.next(None)
            if tok is None:
                return
            yield tok

    def close(self) -> None:
        if not self._destroyed and self._h:
            _capi.load().syi_request_destroy(self._h)
            self._h = None
            self._destroyed = True

    def __enter__(self) -> "Request":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Model:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        layer_map: Optional[str] = None,
        max_seq: int = 0,
        max_slots: int = 16,
        n_blocks: int = 0,
        kv_cap_mb: int = 0,
        kv_type: str = "i8",
        cpu_threads: int = 0,
        n_splits: int = 0,
        dec_splits: int = 0,
        batch_window_ms: float = 2.0,
        max_batch_rows: int = 0,
        max_queue_depth: int = 1024,
        max_queue_tokens: int = 4_194_304,
        request_timeout_s: Optional[float] = None,
        prefix_cache: Optional[bool] = None,
        pc_dir: Optional[str] = None,
        pc_vram_mb: int = -1,
        pc_ram_mb: int = -1,
        pc_disk_mb: int = -1,
        moe_resident: int = 0,
        moe_vram_mb: int = 0,
        moe_ram_mb: int = 0,
        engram_ram_mb: int = -1,
        yarn: bool = False,
        yarn_factor: float = 0.0,
        yarn_orig_ctx: int = 0,
        head_split: bool = False,
    ) -> None:
        lib = _capi.load()
        o = _ModelOptions()
        lib.syi_model_options_init(ctypes.byref(o))
        self._refs: list = []
        self._path = str(model_path)
        self._model_path = self._path.encode()
        self._refs.append(self._model_path)
        o.model_path = self._model_path
        if layer_map:
            self._layer_map = layer_map.encode()
            self._refs.append(self._layer_map)
            o.layer_map = self._layer_map
        o.device = _device_code(device)
        o.max_seq = max_seq
        o.max_slots = max_slots
        o.n_blocks = n_blocks
        o.kv_cap_mb = kv_cap_mb
        o.kv_type, o.kv_type_v = _kv_pair(kv_type)
        o.cpu_threads = cpu_threads
        o.n_splits = n_splits
        o.dec_splits = dec_splits
        o.batch_window_us = int(batch_window_ms * 1000)
        o.max_batch_rows = max_batch_rows
        o.max_queue_depth = max_queue_depth
        o.max_queue_tokens = max_queue_tokens
        o.request_timeout_ms = 0 if request_timeout_s is None else int(request_timeout_s * 1000)
        o.prefix_cache = -1 if prefix_cache is None else (1 if prefix_cache else 0)
        if pc_dir:
            self._pc_dir = pc_dir.encode()
            self._refs.append(self._pc_dir)
            o.pc_dir = self._pc_dir
        o.pc_vram_mb = pc_vram_mb
        o.pc_ram_mb = pc_ram_mb
        o.pc_disk_mb = pc_disk_mb
        o.moe_resident = moe_resident
        o.moe_vram_mb = moe_vram_mb
        o.moe_ram_mb = moe_ram_mb
        o.engram_ram_mb = engram_ram_mb
        o.yarn = 1 if yarn else 0
        o.yarn_factor = yarn_factor
        o.yarn_orig_ctx = yarn_orig_ctx
        o.head_split = 1 if head_split else 0
        h = ctypes.c_void_p()
        st = lib.syi_model_create(ctypes.byref(o), ctypes.byref(h))
        _check(st)
        self._h = h
        self._closed = False

    def _handle(self):
        if self._closed or not self._h:
            raise ClosedError("model is closed")
        return self._h

    @property
    def info(self) -> ModelInfo:
        i = _ModelInfo()
        _capi.load().syi_model_info_init(ctypes.byref(i))
        _check(_capi.load().syi_model_get_info(self._handle(), ctypes.byref(i)))
        return ModelInfo(
            model_name=_capi.sstr(i.model_name),
            architecture=_capi.sstr(i.architecture),
            device_name=_capi.sstr(i.device_name),
            n_vocab=i.n_vocab,
            n_ctx=i.n_ctx,
            n_embd=i.n_embd,
            n_layer=i.n_layer,
            n_head=i.n_head,
            n_head_kv=i.n_head_kv,
            head_dim=i.head_dim,
            is_moe=bool(i.is_moe),
            is_split_gguf=bool(i.is_split_gguf),
            max_seq=i.max_seq,
            max_slots=i.max_slots,
            kv_type=_kv_name(i.kv_type),
            kv_type_v=_kv_name(i.kv_type_v),
            device=_device_name(i.device),
        )

    @property
    def stats(self) -> ModelStats:
        s = _ModelStats()
        _capi.load().syi_model_stats_init(ctypes.byref(s))
        _check(_capi.load().syi_model_get_stats(self._handle(), ctypes.byref(s)))
        return ModelStats(
            submitted=s.submitted,
            admitted=s.admitted,
            completed=s.completed,
            cancelled=s.cancelled,
            failed=s.failed,
            generated_tokens=s.generated_tokens,
            prefill_steps=s.prefill_steps,
            decode_steps=s.decode_steps,
            avg_decode_batch=s.avg_decode_batch,
            max_decode_batch=s.max_decode_batch,
            queued=s.queued,
            active=s.active,
            free_slots=s.free_slots,
        )

    def tokenize(self, text: str | bytes, *, parse_special: bool = True) -> list[int]:
        data = text.encode() if isinstance(text, str) else text
        p = ctypes.POINTER(ctypes.c_int32)()
        n = ctypes.c_size_t()
        _check(_capi.load().syi_model_tokenize(self._handle(), data, len(data), 1 if parse_special else 0,
                                               ctypes.byref(p), ctypes.byref(n)))
        if not n.value:
            return []
        try:
            return [int(p[i]) for i in range(n.value)]
        finally:
            _capi.load().syi_free(p)

    def detokenize(self, ids: Sequence[int]) -> str:
        arr = (ctypes.c_int32 * len(ids))(*ids)
        p = ctypes.c_void_p()
        n = ctypes.c_size_t()
        _check(_capi.load().syi_model_detokenize(self._handle(), arr, len(ids), ctypes.byref(p), ctypes.byref(n)))
        try:
            return ctypes.string_at(p, n.value).decode("utf-8", "replace") if n.value else ""
        finally:
            if p:
                _capi.load().syi_free(p)

    def render_chat(
        self,
        messages: Sequence[ChatMessage] | Sequence[dict],
        *,
        enable_thinking: bool = False,
        tools_json: Optional[str] = None,
    ) -> str:
        arr, refs = _chat_messages(messages)
        p = ctypes.c_void_p()
        n = ctypes.c_size_t()
        _check(_capi.load().syi_model_render_chat(
            self._handle(), arr, len(arr), 1 if enable_thinking else 0,
            tools_json.encode() if tools_json else None, ctypes.byref(p), ctypes.byref(n)))
        try:
            return ctypes.string_at(p, n.value).decode("utf-8", "replace") if n.value else ""
        finally:
            if p:
                _capi.load().syi_free(p)

    apply_chat_template = render_chat

    def submit(
        self,
        prompt: str | bytes | None = None,
        *,
        tokens: Optional[Sequence[int]] = None,
        messages: Optional[Sequence[ChatMessage] | Sequence[dict]] = None,
        enable_thinking: bool = False,
        tools_json: Optional[str] = None,
        options: Optional[GenerationOptions | dict] = None,
        **kwargs,
    ) -> Request:
        gen = _generation_options(options, kwargs)
        cgen, refs = _gen_to_c(gen)
        h = ctypes.c_void_p()
        _check(_capi.load().syi_request_create(self._handle(), ctypes.byref(cgen), ctypes.byref(h)))
        req = Request(self, h)
        try:
            if messages is not None:
                arr, mrefs = _chat_messages(messages)
                _check(_capi.load().syi_request_set_chat(
                    h, arr, len(arr), 1 if enable_thinking else 0,
                    tools_json.encode() if tools_json else None))
            elif tokens is not None:
                arr = (ctypes.c_int32 * len(tokens))(*tokens)
                _check(_capi.load().syi_request_set_tokens(h, arr, len(tokens)))
            elif prompt is not None:
                data = prompt.encode() if isinstance(prompt, str) else prompt
                _check(_capi.load().syi_request_set_text(h, data, len(data)))
            else:
                raise InvalidArgumentError("prompt, tokens or messages is required")
            _check(_capi.load().syi_request_submit(h))
        except Exception:
            req.close()
            raise
        # Keep the generation-option arrays alive until the C call copied them.
        req._refs = refs
        return req

    def submit_batch(self, prompts: Sequence[str | bytes], *, options: Optional[GenerationOptions | dict] = None,
                     **kwargs) -> list[Request]:
        return [self.submit(p, options=options, **kwargs) for p in prompts]

    def generate(self, prompt: str | bytes, *, options: Optional[GenerationOptions | dict] = None,
                 **kwargs) -> Generation:
        req = self.submit(prompt, options=options, **kwargs)
        try:
            req.wait()
            return req.result()
        finally:
            req.close()

    def generate_batch(self, prompts: Sequence[str | bytes], *,
                       options: Optional[GenerationOptions | dict] = None, **kwargs) -> list[Generation]:
        reqs = self.submit_batch(prompts, options=options, **kwargs)
        out = []
        try:
            for r in reqs:
                r.wait()
                out.append(r.result())
            return out
        finally:
            for r in reqs:
                r.close()

    def chat(self, messages: Sequence[ChatMessage] | Sequence[dict], *,
             options: Optional[GenerationOptions | dict] = None, **kwargs) -> Generation:
        req = self.submit(messages=messages, options=options, **kwargs)
        try:
            req.wait()
            return req.result()
        finally:
            req.close()

    def close(self, timeout: float = 5.0) -> None:
        if self._closed or not self._h:
            return
        lib = _capi.load()
        # Ask the scheduler to stop and cancel what is still running; if the
        # current forward outlives the grace period, destroy() cancels it and
        # joins.  Either way the handle is not usable afterwards.
        st = lib.syi_model_shutdown(self._h, _timeout_ms(timeout))
        if st not in (0, 2):
            lib.syi_model_destroy(self._h)
            self._h = None
            self._closed = True
            _check(st)
        lib.syi_model_destroy(self._h)
        self._h = None
        self._closed = True

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _chat_messages(messages) -> tuple[ctypes.Array, list]:
    refs: list = []
    arr = (_ChatMessage * len(messages))()
    for i, m in enumerate(messages):
        if isinstance(m, dict):
            m = ChatMessage(
                role=str(m.get("role", "user")),
                content=str(m.get("content", "") or ""),
                name=m.get("name"),
                tool_call_id=m.get("tool_call_id"),
                reasoning_content=m.get("reasoning_content"),
            )
        if not isinstance(m, ChatMessage):
            raise TypeError("messages must be ChatMessage or mapping")
        _capi.load().syi_chat_message_init(ctypes.byref(arr[i]))
        vals = {}
        for key in ("role", "content", "name", "tool_call_id", "reasoning_content"):
            v = getattr(m, key)
            if v is None:
                continue
            b = str(v).encode()
            refs.append(b)
            vals[key] = b
        for key, val in vals.items():
            setattr(arr[i], key, val)
    return arr, refs


def _generation_options(options: Optional[GenerationOptions | dict], kwargs: dict) -> GenerationOptions:
    if options is None:
        g = GenerationOptions()
    elif isinstance(options, GenerationOptions):
        g = GenerationOptions(**asdict(options))
    elif isinstance(options, dict):
        g = GenerationOptions(**options)
    else:
        raise TypeError("options must be GenerationOptions or dict")
    for k, v in kwargs.items():
        if not hasattr(g, k):
            raise TypeError(f"unknown generation option {k!r}")
        setattr(g, k, v)
    return g


def _gen_to_c(g: GenerationOptions):
    c = _GenerationOptions()
    _capi.load().syi_generation_options_init(ctypes.byref(c))
    c.max_tokens = int(g.max_tokens)
    c.ignore_eos = 1 if g.ignore_eos else 0
    c.temperature = float(g.temperature)
    c.top_p = float(g.top_p)
    c.top_k = int(g.top_k)
    c.min_p = float(g.min_p)
    c.repeat_penalty = float(g.repeat_penalty)
    c.repeat_last_n = int(g.repeat_last_n)
    c.presence_penalty = float(g.presence_penalty)
    c.frequency_penalty = float(g.frequency_penalty)
    c.seed = int(g.seed)
    c.logprobs = 1 if g.logprobs else 0
    c.top_logprobs = int(g.top_logprobs)
    c.timeout_ms = 0 if g.timeout_s is None else int(g.timeout_s * 1000)
    refs = []
    if g.stop:
        arr = (ctypes.c_char_p * len(g.stop))()
        for i, s in enumerate(g.stop):
            b = s.encode()
            refs.append(b)
            arr[i] = b
        c.stop = arr
        c.n_stop = len(g.stop)
        refs.append(arr)
    if g.logit_bias:
        ids = (ctypes.c_int32 * len(g.logit_bias))(*[int(k) for k in g.logit_bias])
        vals = (ctypes.c_float * len(g.logit_bias))(*[float(v) for v in g.logit_bias.values()])
        c.logit_bias_ids = ids
        c.logit_bias_values = vals
        c.n_logit_bias = len(g.logit_bias)
        refs.extend([ids, vals])
    return c, refs


def doctor() -> int:
    from .doctor import main
    return main()
