from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class RequestState(IntEnum):
    CREATED = 0
    QUEUED = 1
    PREFILL = 2
    DECODING = 3
    FINISHED = 4
    FAILED = 5
    CANCELLED = 6


class FinishReason(IntEnum):
    NONE = 0
    STOP = 1
    LENGTH = 2
    CANCELLED = 3
    TIMEOUT = 4
    ERROR = 5


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None


@dataclass
class GenerationOptions:
    max_tokens: int = 256
    ignore_eos: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 0
    logprobs: bool = False
    top_logprobs: int = 0
    timeout_s: Optional[float] = None
    logit_bias: dict[int, float] = field(default_factory=dict)
    stop: list[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    model_name: str
    architecture: str
    device_name: str
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_layer: int
    n_head: int
    n_head_kv: int
    head_dim: int
    is_moe: bool
    is_split_gguf: bool
    max_seq: int
    max_slots: int
    kv_type: str
    kv_type_v: str
    device: str


@dataclass
class ModelStats:
    submitted: int
    admitted: int
    completed: int
    cancelled: int
    failed: int
    generated_tokens: int
    prefill_steps: int
    decode_steps: int
    avg_decode_batch: float
    max_decode_batch: int
    queued: int
    active: int
    free_slots: int


@dataclass
class Logprob:
    id: int
    logprob: float
    text: str = ""


@dataclass
class Token:
    id: int
    text: str
    logprob: float = 0.0
    top: list[Logprob] = field(default_factory=list)
    index: int = 0


@dataclass
class Generation:
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    error: str = ""
