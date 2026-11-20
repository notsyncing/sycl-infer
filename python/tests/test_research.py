"""End-to-end test for the research Python API (model-backed, CPU).

    SYCL_INFER_LIB=/path/to/build/libsycl_infer.so.1.0.0 \
        python3 python/tests/test_research.py /path/to/Qwen3.5-0.8B-Q4_K_M.gguf

The wheel's bundled library predates the research API, so SYCL_INFER_LIB must
point at a freshly built libsycl_infer.  Verifies all three layers: capture,
top-neuron pattern export, and the 0-1 similarity matrix.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sycl_infer  # noqa: E402
from sycl_infer.research import FFNPattern, capture_ffn, concept_similarity, similarity  # noqa: E402

WORDS = [
    "苹果", "梨", "香蕉", "葡萄", "橘子",
    "猫", "狗", "老虎", "兔子", "盒子",
    "汽车", "桌子", "椅子", "上", "下",
    "思想", "语言", "数字", "剪刀", "白菜",
    "石头", "茶", "咖啡", "牛奶",
]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: test_research.py <model.gguf>")
        return 2
    model = sycl_infer.Model(sys.argv[1], device="cpu", max_seq=128, max_slots=1,
                             n_blocks=32, kv_cap_mb=128, cpu_threads=4, prefix_cache=False)
    info = model.info
    print(f"model: {info.architecture} layers={info.n_layer}")

    # layer 1: capture
    acts = capture_ffn(model, WORDS, layer=5, token_slot=-1)
    assert acts.n_texts == len(WORDS) and acts.n_layers == 1 and acts.n_ff > 0
    assert acts.shape == (len(WORDS), 1, 2, acts.n_ff)
    assert len(acts.data) == acts.n_texts * acts.n_layers * 2 * acts.n_ff
    assert all(x == x and abs(x) < 1e30 for x in acts.data)  # finite
    print(f"capture shape={acts.shape} n_ff={acts.n_ff}")

    # layer 2: pattern -> top neurons
    with FFNPattern(acts, top_k=128, half=0, zscore=True) as pat:
        ids, z, nt, nl, k, dim = pat.top()
        assert nt == len(WORDS) and nl == 1 and k == 128 and dim == acts.n_ff
        assert all(0 <= v < dim for row in ids for lr in row for v in lr)
        assert all(x == x for row in z for lr in row for x in lr)
        print(f"pattern: {nt} texts x {nl} layers, top-k={k}, dim={dim}")

        # layer 3: similarity
        mat = similarity(pat)
        n = len(WORDS)
        assert len(mat) == n and all(len(r) == n for r in mat)
        assert mat[0][0] == 1.0
        for i in range(n):
            for j in range(n):
                assert 0.0 <= mat[i][j] <= 1.0
                assert abs(mat[i][j] - mat[j][i]) < 1e-12
        near = mat[WORDS.index("苹果")][WORDS.index("梨")]
        far = mat[WORDS.index("剪刀")][WORDS.index("白菜")]
        print(f"similarity: 苹果/梨={near:.4f} 猫/狗={mat[WORDS.index('猫')][WORDS.index('狗')]:.4f} "
              f"剪刀/白菜={far:.4f} 苹果/石头={mat[WORDS.index('苹果')][WORDS.index('石头')]:.4f}")
        assert near > far, (near, far)

    # convenience one-shot
    mat2, acts2, ids2 = concept_similarity(model, WORDS, layer=5, top_k=128)
    assert mat2[0][0] == 1.0 and len(ids2) == len(WORDS)
    print("concept_similarity OK")

    model.close()
    print("test_research OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
