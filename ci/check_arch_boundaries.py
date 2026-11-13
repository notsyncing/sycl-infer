#!/usr/bin/env python3
"""Architecture/device-boundary gate.

The generic engine and model files must not dispatch on, name, or hold
architecture-specific state: that is what the arch_ops hooks and the per-model
pimpls are for.  This scans the small set of files that are supposed to stay
architecture-agnostic and fails if one of them names an architecture in *code*
(comments and string literals are stripped first, so a comment or a registry
key string is not a violation).

It also enforces the model/device boundary: ``src/model/model.h`` describes the
file (GGUF, hparams, tensor bindings, the ``w8t`` data slots) and must not
include SYCL or expose the device API - the whole-mmap blob, the SIn copies,
upload/pointer translation and their frees live in
``src/backend/weight_store.{h,cpp}``.  ``src/model/model_w8.cpp`` is gone; its
existence is a violation.

The same store is the single device weight storage for the vision mmproj, the
audio mmproj and the DFlash drafter: those models may own derived f32 side
caches (``d_w32`` / ``d_qkv_b32``) and scratch, but not a raw weight blob of
their own, so ``dev_weights`` / ``dev_weights_size`` / ``dev_reserved`` must not
reappear in their headers.

Exit 0 with a one-line summary, or 1 with one `file:line: snippet` per hit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Only these files are supposed to stay architecture-agnostic.
GENERIC_FILES = [
    "src/engine/engine.cpp",
    "src/engine/graph.cpp",
    "src/engine/engine_prefix_cache.cpp",
    "src/engine/engine_kvpool.cpp",
    "src/server/server.cpp",
    "src/model/model.h",
    "src/model/model.cpp",
    # The public model seam: `struct arch::loader` + find/registry, and the
    # shared loader helpers.  The per-arch loader/profile declarations and the
    # `kLoaders[]` table moved to arch/<arch>/load.h and the internal
    # arch/registry.cpp hub, so these headers must keep naming no architecture.
    "src/model/arch/registry.h",
    "src/model/arch_util.h",
    # The public arch seam: `struct arch_ops` + `arch_ops_for` only.  The
    # per-arch hook declarations moved to registry_internal.h, so this header
    # must keep naming no architecture.
    "src/engine/arch/arch_ops.h",
]

# Rules applied to the comment-and-string-stripped text.
STRIPPED_RULES = [
    re.compile(r"\b(hc_fwd|ds41_fwd)\s*\("),
    re.compile(r"\b(qwen35|qwen35moe|qwen4exp|qwen2vl|deepseek41)\b"),
    re.compile(r"\bq4_[a-zA-Z0-9_]+"),
    re.compile(r"\bds41_[a-zA-Z0-9_]+"),
    # The per-model state members themselves: a generic file must reach them
    # only through an arch_ops hook, never name the composed state.
    re.compile(r"\b(q4x_|ds41x_)\b"),
    # The forward-kind enum is a dispatch decision, so naming a model-owned
    # kind is a branch the arch_ops row should own.
    re.compile(r"\bmodel_profile::fwd_kind::(hc_ple_qsa|deepseek41)\b"),
    re.compile(r"\barch_profile\b"),
]

# The include path lives inside a string literal, so this rule runs on the
# comment-stripped text (strings preserved).  `arch/registry.h` is the public
# seam these generic files are allowed to include; every `arch/<name>/...` and
# `engine/arch/...` include is an implementation leak.
INCLUDE_RULE = re.compile(r'#\s*include\s*"(?:arch/(?!registry\.h")|engine/arch/)')

# The internal half of the seam (the per-arch hook declarations) is included by
# exactly the registration and implementation TUs: the table in registry.cpp and
# one runtime.cpp per model-owned forward.  No other file may reach it directly.
INTERNAL_INCLUDE_RULE = re.compile(r'#\s*include\s*"arch/registry_internal\.h"')
INTERNAL_INCLUDE_ALLOWED = {
    "src/engine/arch/registry.cpp",
    "src/engine/arch/qwen4exp/runtime.cpp",
    "src/engine/arch/deepseek41/runtime.cpp",
}

# model.h must not hold architecture-specific fields either.
MODEL_H_FIELD_RULE = re.compile(
    r"\b(hc_attn|hc_ffn|ple_table|engram_token_map|ds41_hparams|q4exp|deepseek41|down_i8[A-Za-z0-9_]*|down_sc)\b"
)

# model.h is a description of the file, not of the device representation: no
# SYCL type, no whole-blob upload, no SIn build/free/pointer API.  Those are
# weight_store's.  Checked on the comment/string-stripped text so a prose
# mention of `upload` in a comment is not a violation.
MODEL_H_DEVICE_RULES = [
    re.compile(r"\bsycl\s*::"),
    re.compile(r"\bdev_weights(_size)?\b"),
    re.compile(r"\bbuild_w8(_one)?\b"),
    re.compile(r"\bfree_w8\b"),
    re.compile(r"\bdev_ptr\b"),
    re.compile(r"\bdev_f32\b"),
    re.compile(r"\bupload\s*\("),
]
MODEL_H_SYCL_INCLUDE_RULE = re.compile(r"#\s*include\s*<sycl")
# The SIn build/free implementation moved to src/backend/weight_store.cpp; a
# resurrected model-side file means the device representation is leaking back
# into `model`.
MODEL_W8_PATH = "src/model/model_w8.cpp"

# The encoder/drafter models that use weight_store directly: their headers may
# name the derived f32 side caches and scratch, never the old per-model raw
# weight blob (a second device weight store).
SHARED_BLOB_FILES = {
    "src/mm/vision.h",
    "src/mm/audio_model.h",
    "src/model/dflash.h",
}
SHARED_BLOB_RULE = re.compile(r"\b(dev_weights|dev_weights_size|dev_reserved)\b")


def strip_comments_and_strings(text: str) -> tuple[str, str]:
    """Return (comments_and_strings_stripped, comments_only_stripped).

    Both preserve length and line breaks, so a regex match maps back to the
    original line number and snippet.  The first removes comments *and* string/
    char literals; the second removes comments only (so `#include "arch/..."`
    can still be seen).
    """
    no_comments: list[str] = []
    stripped: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                no_comments.append(" ")
                stripped.append(" ")
                i += 1
            continue
        if c == "/" and nxt == "*":
            no_comments.append("  ")
            stripped.append("  ")
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                ch = text[i]
                no_comments.append("\n" if ch == "\n" else " ")
                stripped.append("\n" if ch == "\n" else " ")
                i += 1
            if i < n:
                no_comments.append("  ")
                stripped.append("  ")
                i += 2
            continue
        if c in ('"', "'"):
            quote = c
            no_comments.append(c)
            stripped.append(" ")
            i += 1
            while i < n:
                ch = text[i]
                if ch == "\\" and i + 1 < n:
                    no_comments.append(ch)
                    no_comments.append(text[i + 1])
                    stripped.append("  ")
                    i += 2
                    continue
                if ch == quote:
                    no_comments.append(ch)
                    stripped.append(" ")
                    i += 1
                    break
                no_comments.append(ch)
                stripped.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        no_comments.append(c)
        stripped.append(c)
        i += 1
    return "".join(stripped), "".join(no_comments)


def check_file(rel: str) -> list[str]:
    path = ROOT / rel
    if not path.is_file():
        return [f"{rel}:0: missing file"]
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped, no_comments = strip_comments_and_strings(text)
    original_lines = text.splitlines()
    hits: list[str] = []

    def report(lineno: int, needle: str) -> None:
        snippet = original_lines[lineno - 1].strip() if 0 < lineno <= len(original_lines) else ""
        hits.append(f"{rel}:{lineno}: {snippet}")

    def scan(haystack: str, pattern: re.Pattern[str]) -> None:
        for m in pattern.finditer(haystack):
            report(haystack.count("\n", 0, m.start()) + 1, m.group(0))

    for rule in STRIPPED_RULES:
        scan(stripped, rule)
    scan(no_comments, INCLUDE_RULE)
    if rel == "src/model/model.h":
        scan(stripped, MODEL_H_FIELD_RULE)
        for rule in MODEL_H_DEVICE_RULES:
            scan(stripped, rule)
        scan(no_comments, MODEL_H_SYCL_INCLUDE_RULE)
    if rel in SHARED_BLOB_FILES:
        scan(stripped, SHARED_BLOB_RULE)

    # De-duplicate identical (line, snippet) pairs from overlapping rules.
    seen = set()
    unique = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique


def check_internal_includes() -> list[str]:
    """Only registry.cpp and the per-arch runtime.cpp may include the internal
    registry header; any other includer is a leak of the non-public seam."""
    hits: list[str] = []
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix not in (".h", ".cpp"):
            continue
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        _stripped, no_comments = strip_comments_and_strings(text)
        original_lines = text.splitlines()
        for m in INTERNAL_INCLUDE_RULE.finditer(no_comments):
            if rel not in INTERNAL_INCLUDE_ALLOWED:
                line = no_comments.count("\n", 0, m.start()) + 1
                snippet = original_lines[line - 1].strip() if 0 < line <= len(original_lines) else ""
                hits.append(f"{rel}:{line}: {snippet}")
    return hits


def main() -> int:
    violations: list[str] = []
    for rel in GENERIC_FILES:
        violations.extend(check_file(rel))
    for rel in sorted(SHARED_BLOB_FILES - set(GENERIC_FILES)):
        violations.extend(check_file(rel))
    violations.extend(check_internal_includes())
    if (ROOT / MODEL_W8_PATH).is_file():
        violations.append(f"{MODEL_W8_PATH}:0: device storage must live in src/backend/weight_store.cpp")
    if violations:
        for v in violations:
            print(v)
        print(
            f"arch_boundaries: {len(violations)} violation(s) across {len(GENERIC_FILES)} generic "
            f"and {len(SHARED_BLOB_FILES)} shared-model files"
        )
        return 1
    print(
        f"arch_boundaries: OK ({len(GENERIC_FILES)} generic files, no architecture dispatch/state; "
        f"model.h has no SYCL/device API; {len(SHARED_BLOB_FILES)} shared models use weight_store)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
