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
`third_party/unicode*`、`src/backend/gpu/kernels/*` 与 `src/backend/cpu/kernels/*`（每 kernel 一个 TU）、
`src/model/*`、`src/mm/*`、`src/engine/*`、`src/server/*`。

包含目录：`src/common`、`src/backend`、`src/backend/cpu`、`src/backend/gpu/kernels`、`src/model`、`src/mm`、
`src/engine`、`src/server`、`third_party`。

CPU 内核目录（`src/common/cpu_isa.cpp` 与 `src/backend/cpu/kernels/*`）通过 `SI_CPU_SOURCES` 列表
设为 **`-fno-sycl`**：它们是纯主机 TU，带 AVX2 / AVX-VNNI / AVX-512 target attribute，必须留在
SYCL device pass 之外。新增 CPU 内核要同时加入该列表。

二进制：`sycl-infer`（`src/main.cpp`）。测试按后端分树：`tests/backend/cpu/`（含 `kernels/`）与
`tests/backend/gpu/`（含 `kernels/`），共享 harness 在 `tests/common/`。

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
| `test_response_parser` | CPU | `reasoning_content`/`tool_calls` 流式拆分（整段与逐字节喂入一致） |
| `test_compare` | CPU | CPU 参考各阶段 vs llama.cpp tensor dump |
| `test_cpuref` | CPU | CPU 参考 forward head on token ids（默认 `{9419}`） |
| `test_sampler` | CPU | `logit_bias` 强制/封禁 token、logprob log-softmax 与归一化、best_of 打分路径 |
| `test_cpu_gemv` | CPU | CPU 融合 fp32 + 整数 int8 GEMV/RMSNorm vs `quant.h` 主机反量化参考，多类型多 TB；`PF_CPU_ISA` 可锁变体 |
| `test_cpu_gdn` | CPU | `cpu_gdn` 用**非对称 head 数**（`n_k_heads=2, n_heads=6`）对照测试内标量参考，覆盖 `head % n_k_heads` 配对与状态写回（0.8B 的真实维度是恒等映射，测不出该 bug） |
| `test_w4` | CPU | w4 原生宽度打包：unpack(pack(Q4_K)) 往返（~0.077% rel L2，int8 转换是 ~0.98%） |
| `test_pc_cpu` | CPU | 主机后端的 paged 注意力 + 前缀缓存磁盘 spill/promote 往返（日志逐位一致） |
| `test_pc_disk` | CPU | 磁盘层记录格式往返、token 校验、LRU 预算、重开持久化、损坏/未知记录 |
| `test_pc_ram` | CPU | RAM 层 LRU 记录存储 |
| `test_multimodal` | CPU+GPU | 图像预处理几何、host/device 视觉编码器、位置与 prompt 布局、逐 kernel 对照 |
| `test_gpu_stages` | GPU | 每个 kernel vs CPU 参考（强制 `PF_DP4A=0`）；调用 rmsnorm/embed/qk_norm_rope/attn/conv/gdn/gated_norm |
| `test_gemv` | GPU | 每个真实 GGUF 张量的 GEMV vs CPU 反量化参考，多 TB |
| `test_dp4a` | GPU | SIn 重排 + DP4A GEMM vs 同量化输入的 CPU 参考，报告相对 fp32 的量化误差 |
| `test_w4_gemm` | GPU | u4 prefill GEMM（`dnnl_gemm::gemm_w4`）vs 真实反量化权重的 fp32 参考（残差应只含权重表示误差 ~0.08%） |
| `test_w4_vs_i8` | GPU | 同一批量化激活下 u4 GEMM vs int8 GEMM 的逐张量比较，用于定位 u4 打包/元数据的偏差 |
| `test_gemv_stride` | GPU | `gemv_group` 在引擎真实参数组合下的探针：`x_stride == K` vs `2*K`（gate/up 交错）、`residual == nullptr` vs `out`（原地残差） |
| `test_iq_dequant` | GPU | IQ*/Q3_K 的 GPU 反量化/GEMV vs `quant.h` 主机参考（用 27B 混合量化模型；缺失张量跳过） |
| `test_quant_audit` | GPU | 每种 ggml 类型取一张张量，比较 int8(oneDNN) GEMM 与"真实反量化权重"的 fp32 参考，量化 int8 转换对每种类型的损失 |
| `test_gpu_vs_ref` | GPU | 端到端 GPU logits vs `cpu_ref`（强制 `PF_DP4A=0`） |
| `test_forward` | GPU | 端到端 logits/top-k 与校验和（强制 `PF_DP4A=0`） |
| `test_w4_vs_cpuref` | GPU | u4 权重路径的端到端 logits vs fp32 CPU 参考；配 `TEST_LAYER_MAP` 可跑 27B 分卡 |
| `test_w4_topk` | GPU | 打印 chat prompt 的 top-12 prefill logits（u4 vs int8 对比）；`PF_DUMP_LOGITS=<path>` 落盘整条 logit 向量 |
| `test_27b_prefill` | GPU | 27B 分卡 prefill 的接线/阈值排查（依次加长 prompt），带 `TEST_KERNEL_BENCH`/`TEST_SKIP_DECODE`/`STOP_AFTER_LAYER` |
| `test_decode_vs_prefill` | GPU | **单 token decode 与同序列重新 prefill 必须给出同一预测**；纯 prefill 的测试对解码路径是盲区，这个测试是唯一覆盖 |
| `test_pc_gpu` | GPU | 磁盘 spill + promote 往返，第二次相同 prompt 从磁盘恢复需与首次 logits 一致 |
| `test_pc_ram_gpu` | GPU | VRAM→RAM→VRAM 往返；新引擎不得看到陈旧 RAM 条目 |

### 2.1 stage 测试脚手架

* `tests/backend/gpu/kernels/stage_tests.h` 声明七个 stage 入口。
* 每个 `*_stage.cpp` 定义 `void stage_<kernel>(si::stage_env & env)`，用 `env.get(...)` 取快照、
  `env.cmp(...)` 比较（见 `tests/common/stage_test.h`）。
* `tests/common/cpu_ref.h` 提供主机参考前向，被 `test_cpuref`、`test_compare`、`test_gpu_vs_ref` 使用。

### 新增 stage 测试

1. 新建 `tests/backend/gpu/kernels/<kernel>_stage.cpp`，定义 `void stage_<kernel>(si::stage_env & env)`。
2. 在 `stage_tests.h` 声明。
3. 在 `tests/backend/gpu/kernels/test_gpu_stages.cpp` 调用。
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
src/backend/gpu/kernels/    → docs/design/03-kernels.md §1-13
src/backend/cpu/kernels/    → docs/design/03-kernels.md §14, docs/architecture.md §11
src/backend/cpu/    → docs/design/03-kernels.md §14, docs/architecture.md §11（设备选择与多设备执行）
src/model/      → docs/design/01-model-loading.md, 08-tokenizer.md, 11-qwen35-model.md
src/mm/         → docs/design/10-multimodal.md
src/engine/     → docs/design/04-engine.md, 05-kv-cache.md, 06-prefix-cache.md, 07-sampler.md
src/server/     → docs/design/09-server.md
tests/backend/gpu/  → docs/design/03-kernels.md, 06-prefix-cache.md
tests/backend/cpu/  → docs/architecture.md §11, docs/design/06-prefix-cache.md
tests/common/       → 本文档（共享 harness）
tests/model/, tests/mm/ → 本文档
tests/server/, tests/engine/ → docs/design/09-server.md, 07-sampler.md
```

---

## 6. 测量报告索引（`reports/`）

`reports/` 保存一次性的性能/正确性报告，与稳定设计文档互补：

| 报告 | 主题 |
|---|---|
| `attn_blockread_int8kv.md` | int8 KV 的 attention block-read |
| `int4_kv.md` | int4 KV 存储、精度与 GPU 性能 |
| `kv_bf16_splits.md` | bf16 KV split 调优 |
| `longctx_16k.md` / `longctx_decode.md` | 长上下文 prefill / decode |
| `prefix_cache.md` | 前缀缓存设计与测量 |
| `vtune_profile.md` | VTune profile |
| `bugfix_hang.md` | 挂起问题修复记录 |

性能数字必须注明所用的环境变量，因为不同开关会选择不同 kernel 变体。

---

## 7. 调试手册：27B 输出错乱是怎么查出来的

2026-09 的 Qwen3.8-27B 接入过程中，模型在两种后端、所有量化路径下都只输出重复/一两个 token。
最终查出**三个独立的 bug**（全在未提交的 multi-device 改造里），下面是可复用的方法论与踩过的坑。

### 7.1 方法论（按顺序做，别跳）

1. **先拿独立参照，再谈对错**。用 llama.cpp 跑同一个 GGUF（`llama-completion -no-cnv -p ...`）
   证明"GGUF 是好的、引擎是坏的"——否则一切都是猜测。
2. **端到端数值对比**，而不是只看文本。`ll_logits`（见 §7.3）与引擎的 logits 做
   corr / top-20 / argmax 比较：27B prefill 达到 **corr 0.998–0.999、top-20 20/20** 才算对上。
3. **逐层指纹找首个分歧层**（`PF_DUMP_LAYERS`），再从该层往上查。
4. **逐元素对比**（`PF_DUMP_RAW=<prefix>`）：指纹类的指标（sum/sumsq/max@idx）**看不出方向变化**。
   实测中 hidden state 的 `sumsq` 只差 0.15%、`max@idx` 恒定，但 logits 已经全错——只有逐元素
   或"把 head 的输入直接 dump 出来比"才能看到。
5. **二分定位**：换内核变体、换环境开关、换运行模式，看结果是否变化（但要先读 §7.2 的坑 2）。
6. **最后用最小复现固化成测试**（并做 §7.2 第 8 条的反向验证）。

### 7.2 踩过的坑（每一条都真的浪费了时间）

1. **"尺子"本身可能是坏的**。`tests/common/cpu_ref.h` 有三个 bug：`head()` 用 `m.tok_embd` 而不是
   `m.output`（0.8B 共享权重看不出，27B 有独立 LM head → 参考值全错）、用 `3*d_inner` 当 qkv/conv 宽度
   （27B 实际 `qkv_dim()=10240`，越界读）、同样的 GDN 头映射错误。**引擎和参考错得一样时测试会"全绿"**。
   接入新模型时，先手工核对参考实现（例如与 llama.cpp dump 比、或至少检查张量形状）。
2. **"消融结果完全没变"有两种解释，不要急着下结论**。当时连换 5 个 GDN 内核变体（`PF_GDN_VEC`/
   `PF_GDN_COLS`/`PF_GDN_WG`…，它们数学等价、只有浮点累加顺序不同）+ `PF_KV_TYPE=f32` +
   `PF_DP4A=0`，采样结果**逐位相同**，于是被读成"与这些子系统无关"。其实更可能是**被更大的结构性
   bug 掩盖**：head 段读到陈旧缓冲时，logits 本身就与权重路径无关地错着，任何精度级开关都改变不了
   采到的 token。（确实也存在"开关无效"的情形：multi-device 下 `use_w8` 恒为真，`PF_DP4A=0`/`PF_W4=0`
   不会关掉 w8 路径。）**要区分两者，得单独确认开关是否真的生效**——用 `PF_PROF`、启动日志或
   `PF_DUMP_SEGS` 观察路径/数据是否变化，而不是只看最终 token。
3. **`n_group == dt_rank` 的退化模型会隐藏头映射 bug**。0.8B 上取模与分块映射**都是恒等**，所以
   "参考模型一切正常"不能证明这类代码是对的。看到"能把相同结构的两种写法互换"的代码就要警惕。
4. **注意相似但不同的约定**。attention 的 GQA 展开是**分块**（HF `repeat_kv`），而 GDN 的 q/k↔value
   配对是**取模**（`ggml_repeat_4d`）。两者都在同一个模型里，不能想当然统一。
5. **plan 是"设备指针快照"**。`bind_acts(dev)` 只改引擎成员，不改已构建的 plan；因此固定在 primary
   执行的 head 段必须在 `bind_acts(0)` 之后构建。症状极具迷惑性：**prefill 正常、decode 全错**，
   而且对所有内核开关免疫（属于 plan 布线）。见 [04-engine.md](04-engine.md) §4.2。
6. **纯 prefill 的测试对 decode 是盲区**。`test_forward`/`test_gpu_vs_ref` 都只做 `eval()`，所以
   decode 路径的 bug 一路绿灯。`test_decode_vs_prefill` 是唯一覆盖（"同一序列 decode 与重新 prefill
   必须给出同一预测"）。**任何只影响单 token 路径的改动都要跑它。**
7. **诊断代码必须与被测代码在同一条队列/设备上**。用 `dev_queue(cur_dev)`（最后一层所在设备）去
   `memcpy` 一个由 primary 队列写的缓冲，会读到**上一个 step**的值（wait() 只保证该队列自己有序），
   于是得到"晚一拍"的假数据并误导推理。诊断也要遵守 [architecture.md](../architecture.md) §9 的不变量。
8. **回归测试必须反向验证**。新增 `test_cpu_gdn` 后，把内核改回错误映射确认它报 `MISMATCH`/退出码 1，
   才算这个测试有效——否则它可能什么都没测。
9. **张量形状是"宽度假设"最快的事实来源**。`blk.0.attn_qkv.weight = [5120, 10240]` 一眼就能否掉
   `3*d_inner=18432`。怀疑维度假设时先 dump GGUF 张量形状。
10. **`rsync -a` 会保留源文件 mtime**，改动时间早于远端目标文件时 `make` 不会重编——排查"改了没效果"
    时先 `find src tests -name '*.cpp' -o -name '*.h' | xargs touch` 再构建。
11. **怀疑第三方库"版本旧"之前先核对**。`minja` 的 `is undefined` 缺口在**上游 HEAD 同样存在**：
    用 `git hash-object third_party/minja/minja.hpp` 与上游 blob 比对，确认我们的副本逐字节等于上游
    HEAD，升级无解，只能本地打补丁（见 [AGENTS.md](../../AGENTS.md) 的 third_party 例外）。

### 7.3 可复用的工具

| 工具 | 用途 |
|---|---|
| `ll_logits`（一次性小程序，链接 llama.cpp 的 `libllama`） | 对显式 token id 序列 dump llama.cpp 的 logits（逐位置），作为独立参照 |
| `cmp_logits.py` 风格比较 | max\|diff\| / mean\|diff\| / corr / top-20 overlap / argmax |
| `PF_DUMP_LAYERS` / `PF_DUMP_RAW` | 逐层指纹与整条激活向量；对比 decode 与 prefill 时用"最后真实 token slot"与"单个 decode 行" |
| `PF_DUMP_DEC_LOGITS` / `PF_DUMP_GEN` | decode 每一步采样器看到的 logits、采样结果与停止原因 |
| `PF_DUMP_PROMPT` | 打印模型实际被喂进去的 id（chat 模板渲染结果）——区分"模板坏"与"前向坏" |
| GGUF 张量形状 dump | 核对所有宽度假设 |
