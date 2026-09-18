# 设计 12：构建、测试与验证

覆盖 `CMakeLists.txt`、`tests/` 与验证流程。约定与快速验证清单见 [AGENTS.md](../../AGENTS.md)。

---

## 1. 构建系统（`CMakeLists.txt`）

### 1.1 工具链

* C++17，CMake ≥ 3.20，编译器必须是 `icpx`（CMake 显式设置 `CMAKE_CXX_COMPILER`），链接 `-fsycl`。
* Release 默认 `-O3 -DNDEBUG`；全局 `-Wall -Wextra -Wno-unused-parameter`。
* **Debug 构建强制设备代码 `-O2`**（`CMakeLists.txt:16-23`）：icpx 默认 `-O0` 会让 IGC 生成的设备镜像
  膨胀约 20 倍、翻译耗时数分钟（看起来像挂起）。不要删除这条。
* `-DSYCL_INFER_AOT=ON` 用 `spir64_gen -device adl-p` AOT 编译设备镜像。
* oneDNN 期望在 `/opt/intel/oneapi/dnnl/2026.0`（`-DDNNL_ROOT=` 覆盖），缺失直接 fatal。
* `CMAKE_EXPORT_COMPILE_COMMANDS=ON`；`.clangd` 加了 SYCL include 并去掉 `-fsycl`。

### 1.2 源文件列表

`sycl_infer_core` 静态库**显式列出**所有 `.cpp`（无 globbing）。新增源文件必须手动加入。目录划分：
`third_party/unicode*`、`src/kernels/*`（每 kernel 一个 TU）、`src/model/*`、`src/mm/*`、
`src/engine/*`、`src/server/*`。

包含目录：`src/common`、`src/backend`、`src/kernels`、`src/model`、`src/mm`、`src/engine`、`src/server`、
`third_party`。

二进制：`sycl-infer`（`src/main.cpp`）。

### 1.3 测试目标

`si_add_test(name, sources...)` 帮助函数链接 `sycl_infer_core` 并把 `tests/common` 加入 include。
目标列表见 §2。`test_gpu_stages` 由 runner + 每个 stage 一个 `.cpp` 组成；新增 stage 要同时加入
`CMakeLists.txt`。

---

## 2. 测试矩阵

测试默认模型路径是 `/path/to/Qwen3.5-0.8B-Q4_K_M.gguf`，多模态还默认
`Qwen3.5-0.8B-mmproj-BF16.gguf`。每个测试接受可选模型路径作为第一个参数。

| 目标 | 类型 | 覆盖 |
|---|---|---|
| `test_tokenizer` | CPU | 词表加载、特殊 id、ASCII/CJK/空白/emoji/特殊 token 往返 |
| `test_chat_template` | CPU | GGUF `tokenizer.chat_template` 经 minja vs 参考 Jinja2 输出，严格计数 mismatch |
| `test_compare` | CPU | CPU 参考各阶段 vs llama.cpp tensor dump |
| `test_cpuref` | CPU | CPU 参考 forward head on token ids（默认 `{9419}`） |
| `test_pc_disk` | CPU | 磁盘层记录格式往返、token 校验、LRU 预算、重开持久化、损坏/未知记录 |
| `test_pc_ram` | CPU | RAM 层 LRU 记录存储 |
| `test_multimodal` | CPU+GPU | 图像预处理几何、host/device 视觉编码器、位置与 prompt 布局、逐 kernel 对照 |
| `test_gpu_stages` | GPU | 每个 kernel vs CPU 参考（强制 `PF_DP4A=0`）；调用 rmsnorm/embed/qk_norm_rope/attn/conv/gdn/gated_norm |
| `test_gemv` | GPU | 每个真实 GGUF 张量的 GEMV vs CPU 反量化参考，多 TB |
| `test_dp4a` | GPU | SIn 重排 + DP4A GEMM vs 同量化输入的 CPU 参考，报告相对 fp32 的量化误差 |
| `test_gpu_vs_ref` | GPU | 端到端 GPU logits vs `cpu_ref`（强制 `PF_DP4A=0`） |
| `test_forward` | GPU | 端到端 logits/top-k 与校验和（强制 `PF_DP4A=0`） |
| `test_pc_gpu` | GPU | 磁盘 spill + promote 往返，第二次相同 prompt 从磁盘恢复需与首次 logits 一致 |
| `test_pc_ram_gpu` | GPU | VRAM→RAM→VRAM 往返；新引擎不得看到陈旧 RAM 条目 |

### 2.1 stage 测试脚手架

* `tests/kernels/stage_tests.h` 声明七个 stage 入口。
* 每个 `*_stage.cpp` 定义 `void stage_<kernel>(si::stage_env & env)`，用 `env.get(...)` 取快照、
  `env.cmp(...)` 比较（见 `tests/common/stage_test.h`）。
* `tests/common/cpu_ref.h` 提供主机参考前向，被 `test_cpuref`、`test_compare`、`test_gpu_vs_ref` 使用。

### 新增 stage 测试

1. 新建 `tests/kernels/<kernel>_stage.cpp`，定义 `void stage_<kernel>(si::stage_env & env)`。
2. 在 `stage_tests.h` 声明。
3. 在 `tests/kernels/test_gpu_stages.cpp` 调用。
4. 把文件加入 `CMakeLists.txt` 的 `test_gpu_stages` 源列表。

---

## 3. 验证流程

### 3.1 构建（必须无警告，链路器关于 `libsvml.so`/`libimf.so`/`libintlc.so.5` 的说明是良性的）

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build
cmake --build build -j$(nproc)
```

### 3.2 改动 kernel/引擎后至少跑

```bash
./build/test_gpu_stages     # all stages OK
./build/test_gpu_vs_ref     # argmax ... SAME
./build/test_forward        # stable last_id
```

### 3.3 include-cleaner lint（`Diagnostics.UnusedIncludes: Strict` 必须干净）

```bash
clang-tidy -p build -checks='-*,misc-include-cleaner' <changed files>
```

---

## 4. 运行期前置

二进制/测试启动需要 oneAPI 运行库。`source /opt/intel/oneapi/setvars.sh` 即可；最小覆盖：

```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:$LD_LIBRARY_PATH
```

CLI/服务用法见 [../README.md](../../README.md) 与 [01-model-loading.md](01-model-loading.md)。

---

## 5. 仓库目录与文档对应

```
src/common/     → docs/design/02-quantization.md
src/backend/    → docs/design/02-quantization.md
src/kernels/    → docs/design/03-kernels.md
src/model/      → docs/design/01-model-loading.md, 08-tokenizer.md, 11-qwen35-model.md
src/mm/         → docs/design/10-multimodal.md
src/engine/     → docs/design/04-engine.md, 05-kv-cache.md, 06-prefix-cache.md, 07-sampler.md
src/server/     → docs/design/09-server.md
tests/          → 本文档
```

---

## 6. 测量报告索引（`reports/`）

`reports/` 保存一次性的性能/正确性报告，与稳定设计文档互补：

| 报告 | 主题 |
|---|---|
| `attn_blockread_int8kv.md` | int8 KV 的 attention block-read |
| `kv_bf16_splits.md` | bf16 KV split 调优 |
| `longctx_16k.md` / `longctx_decode.md` | 长上下文 prefill / decode |
| `prefix_cache.md` | 前缀缓存设计与测量 |
| `vtune_profile.md` | VTune profile |
| `bugfix_hang.md` | 挂起问题修复记录 |

性能数字必须注明所用的环境变量，因为不同开关会选择不同 kernel 变体。
