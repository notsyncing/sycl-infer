# 设计 12：构建、测试与验证

覆盖 `CMakeLists.txt`、`tests/` 与验证流程。约定与快速验证清单见 [AGENTS.md](../../AGENTS.md)。

---

## 1. 构建系统（`CMakeLists.txt`）

### 1.1 工具链

* C++17，CMake ≥ 3.20，编译器必须是 `icpx`（CMake 显式设置 `CMAKE_CXX_COMPILER`），链接 `-fsycl`。
* Release 默认 `-O3 -DNDEBUG`；全局 `-Wall -Wextra -Wno-unused-parameter`。
* **Debug 构建强制设备代码 `-O2`**（`CMakeLists.txt:16-23`）：icpx 默认 `-O0` 会让 IGC 生成的设备镜像
  膨胀约 20 倍、翻译耗时数分钟（看起来像挂起）。不要删除这条。
* oneDNN 期望在 `/opt/intel/oneapi/dnnl/2026.0`（`-DDNNL_ROOT=` 覆盖），缺失直接 fatal。
* OpenSSL 是**可选**的：`find_package(OpenSSL QUIET)` 找到才定义 `CPPHTTPLIB_OPENSSL_SUPPORT` 并链接
  `OpenSSL::SSL`/`OpenSSL::Crypto`（`CMakeLists.txt:207-216`）。没有它 http:// 的 `image_url` 仍可用，
  https:// 会显式报错。
* `CMAKE_EXPORT_COMPILE_COMMANDS=ON`；`.clangd` 加了 SYCL include、去掉 `-fsycl`，并设
  `Diagnostics: UnusedIncludes: Strict`（lint 见 §3.5）。

#### CMake 选项一览

| 选项 | 默认 | 作用 |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Release` 给 `-O3 -DNDEBUG`（`CMakeLists.txt:13`） |
| `DNNL_ROOT` | `/opt/intel/oneapi/dnnl/2026.0` | oneDNN 前缀，找不到 `libdnnl` 直接 fatal |
| `SYCL_INFER_AOT` | `OFF` | AOT 编译设备镜像，避免运行期 JIT（见 §1.4） |
| `SYCL_INFER_AOT_DEVICE` | `adl-p` | ocloc target。`adl-p` = Iris Xe，`acm-g10` = A770/DG2 |
| `SYCL_INFER_AOT_PROFILE` | 空 | 把 device profile key 烤进二进制（`SI_FORCE_DEVICE_PROFILE`）；空 = 运行时按设备名自选 |
| `SYCL_INFER_AOT_JOBS` | `nproc` | AOT 设备链接阶段的并行 ocloc/IGC 进程数（RAM 随此数线性增长） |
| `SI_DEV_BENCH` | `OFF` | 构建 `dev/` 测量 harness（见 §1.5） |

`SYCL_INFER_AOT_DEVICE` 与 `SYCL_INFER_AOT_PROFILE` 必须与二进制运行时选中的 profile 一致——前者是
profile 值的**构建侧镜像**，所以是显式选项而不是自动推导（`CMakeLists.txt:35-41`）。其它 SKU 用
`ocloc query` 查，不要继承任一默认值。

### 1.2 源文件列表

`sycl_infer_core` 静态库**显式列出**所有 `.cpp`（无 globbing）。新增源文件必须手动加入。目录划分：
`third_party/unicode*`、`src/device/`（`device_registry.cpp` + `profiles/<card>.cpp`，每张卡一个文件）、
`src/backend/gpu/kernels/*` 与 `src/backend/cpu/kernels/*`（每 kernel 一个 TU）、
`src/model/*`、`src/mm/*`、`src/engine/*`、`src/server/*`。

包含目录：`src`（让 `device/device_profile.h` 在任何 TU 里都能解析）、`src/common`、`src/backend`、
`src/backend/cpu`、`src/backend/gpu/kernels`、`src/model`、`src/mm`、`src/engine`、`src/server`、
`third_party`。

CPU 内核目录（`src/common/cpu_isa.cpp` 与 `src/backend/cpu/kernels/*`）通过 `SI_CPU_SOURCES` 列表
设为 **`-fno-sycl`**：它们是纯主机 TU，带 AVX2 / AVX-VNNI / AVX-512 target attribute，必须留在
SYCL device pass 之外。新增 CPU 内核要同时加入 `SI_CPU_SOURCES`（`CMakeLists.txt:170-185`）。
注意设置这层属性的 `if(NOT SYCL_INFER_AOT)` 条件（`CMakeLists.txt:186-188`）意味着 **AOT 构建下这些 TU
会带着 `-fsycl` 编译**——两条路径都没被测过，别默认它无害。

`third_party/` 不改，但可以用 `set_source_files_properties(... COMPILE_OPTIONS ...)` 抑制告警：
vendored 的 llama.cpp Unicode 表留了一个故意未用的 helper，编译单元上加 `-Wno-unused-function`
（`CMakeLists.txt:190-193`）。

二进制：`sycl-infer`（`src/main.cpp`）。测试按后端分树：`tests/backend/cpu/`（含 `kernels/`）与
`tests/backend/gpu/`（含 `kernels/`），共享 harness 在 `tests/common/`。

### 1.3 测试目标

`si_add_test(name, sources...)` 帮助函数链接 `sycl_infer_core` 并把 `tests/common` 加入 include。
目标列表见 §2。`test_gpu_stages` 由 runner + 每个 stage 一个 `.cpp` 组成；新增 stage 要同时加入
`CMakeLists.txt`。

`-DSI_DEV_BENCH=ON` 会额外构建 `bench_mtp`（源文件 `dev/bench_mtp.cpp`）——`dev/` 里**只有这一个**
harness 挂在 CMake 上（`CMakeLists.txt:233-238`）。它默认关闭，因为这些是测量工具而不是交付物，
而且每个都会在链接期重新 lower 一遍设备镜像。

### 1.4 AOT 构建（`SYCL_INFER_AOT`）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSYCL_INFER_AOT=ON -DSYCL_INFER_AOT_JOBS=6
cmake --build build --target sycl-infer -j$(nproc)   # 只构建二进制，不构建全部测试
```

* **lowering 是最贵的一步，也是最容易被误判的一步。** `icpx` 为每个 kernel 产出一个设备镜像，然后
  `llvm-foreach` 对每个镜像跑 `ocloc`（IGC）。它发生在**每个可执行文件的最终链接**，所以那 30 个测试
  二进制会各自重新 lower 它们引入的设备代码——除非确实需要 AOT，否则只构建 `--target sycl-infer`。
* **默认的链接后设备步骤是串行的**（单个 `ocloc` 进程），这**不是** `make -j` 的问题。
  `CMakeLists.txt` 传 `-fsycl-max-parallel-link-jobs=N`，所以 `-DSYCL_INFER_AOT_JOBS=N` 会同时跑 N 个
  IGC 翻译（默认 `nproc`）。每个 job 是一次独立的 IGC 翻译，**RAM 随 N 线性增长**，内存紧张的机器要
  调小。
* `CMakeLists.txt` 还传 `-fsycl-device-code-split=per_kernel`：默认的 `auto` 会把整个 module 合成一个
  镜像从而串行化 lowering，per-kernel 镜像可以并行 lower，也能逐个跳过。
* **用 `-DCMAKE_BUILD_TYPE=Release`。** Debug 构建的 `-g` 会给设备镜像加上 debug info，让每个镜像大若干倍、
  lower 更慢；强制的 `-O2` 并不会去掉它。
* **ocloc 编译器缓存在这里不可用**：`-allow_caching` / `-cache_dir`（以及 `NEO_CACHE_DIR`）会创建目录但
  不写入任何条目，在发行版的 `intel-ocloc` 26.27.1 上没有任何加速。**可用的替代是运行期 JIT 缓存**：

  ```bash
  export SYCL_CACHE_PERSISTENT=1
  export SYCL_CACHE_DIR=/path/to/cache    # 条目落在 ~/.cache/neo_compiler_cache/*.l0_cache
  ```

  它跨运行缓存 JIT 编译后的 kernel，可以完全绕开 AOT 构建。**`ccache` 没有用**：设备 lowering 是链接
  步骤，不经过编译器的 cc 缓存路径。

  ⚠️ **在 2x A770 那台机器上这个缓存是个陷阱**（2026-10-03 实测，随盒子的驱动版本）：设了
  `SYCL_CACHE_PERSISTENT=1` 之后**每个** GPU 测试程序都在几秒内 `SIGSEGV`——正好在 KV 池 init 那一行
  之后，也就是第一个 JIT kernel 里面；而同一个二进制不设缓存就通过（`test_gpu_stages`：
  "all stages OK"）。换一个全新的 `SYCL_CACHE_DIR` 同样崩，所以不是目录被污染：**缓存本身在那台驱动上
  会崩**。那台机器请用 AOT 构建（或干脆不设缓存）。

### 1.5 `dev/` 压测工具（本地未跟踪）

`dev/` 在 `.gitignore` 里、`git ls-files dev` 为空，即**这些工具不属于本仓库**，仓库内文档只能按名字
引用，不能把它们当交付物。`-DSI_DEV_BENCH=ON` 只构建 `dev/bench_mtp.cpp`（一进程一配置的 MTP sweep），
其余都是手工单文件编译——每个工具都会在链接期重新 lower 设备镜像：

```bash
icpx -fsycl -std=c++17 -O2 dev/<tool>.cpp -o dev/<tool> \
  -Isrc -Isrc/common -Isrc/backend -Isrc/backend/cpu -Isrc/backend/gpu/kernels \
  -Isrc/model -Isrc/mm -Isrc/engine -Isrc/server -Ithird_party \
  -I/opt/intel/oneapi/dnnl/2026.0/include -Lbuild -lsycl_infer_core -ldnnl
```

`build/libsycl_infer_core.a` 必须是最新的，否则链接会在**该工具最后改动过的那个 launcher** 上失败
（静态库里还没有新符号）。常被引用的几个（名字以磁盘上的为准，`dev/` 下还有几十个同类工具）：

| 工具 | 用途 |
|---|---|
| `dev/bench_mtp.cpp` | MTP sweep，**一进程一配置**（同一进程内换 k 会因前缀缓存命中改变数值，配置之间不可比） |
| `dev/bench_wpass.cpp` + `dev/wpass_list.py` | 用生产 launcher 重放真实的逐 token 调用清单，给出一个 weight pass 的字节数与时间 |
| `dev/bench_dp4a_peak.cpp` | dp4a 与 DPAS 的寄存器常驻吞吐上限（注意它**低估**真实 kernel 1.4-2 倍） |
| `dev/bench_natgemm.cpp` | `nat_gemm_launch` 对标量参考与 M=1 GEMV |
| `dev/bench_dpas_gemm.cpp` | 真实 GEMM 里的 DPAS kernel（配 `dpas_plumb.cpp` / `dpas_data_probe.cpp` 三个探针） |
| `dev/test_w2.cpp` / `dev/test_fused_i8.cpp` | 2-bit 打包、fused int8 call group 的正确性对拍 |
| `dev/alu_probe2.cpp` | fp32 FMA 峰值（`dev/alu_probe.cpp` 已被弃用：编译器会把它的循环折叠掉，报出 1.5 PFLOP/s 的假数） |
| `dev/cmp_pfb.cpp` | mode-2 分批 prefill 的 hidden-state 余弦比对 |

---

## 2. 测试矩阵

三档默认模型，逐个测试自己写死（不是全局约定）：

| 记号 | 默认路径 | 参数 |
|---|---|---|
| **0.8B** | `/path/to/Qwen3.5-0.8B-Q4_K_M.gguf` | `argv[1]` 可覆盖 |
| **27B** | `/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` | `argv[1]` 可覆盖 |
| **无** | 不读 GGUF，纯合成数据 | 不接受模型参数 |

27B 档是**唯一**的混合量化模型，所以审计权重表示的测试（IQ*/Q3_K/Q5_K、每种 ggml 类型的 int8 转换损失）
必须用它；0.8B 只有一个量化类型，测不出格式差异。`test_multimodal` 是唯一多参数的：
`argv[1]` 文本 GGUF、`argv[2]` mmproj GGUF、`argv[3]` **音频 mmproj** GGUF（默认空，即跳过音频编码器测试）。

| 目标 | 默认 | 类型 | 覆盖 |
|---|---|---|---|
| `test_tokenizer` | 0.8B | CPU | 词表加载、特殊 id、ASCII/CJK/空白/emoji/特殊 token 往返 |
| `test_chat_template` | 0.8B | CPU | GGUF `tokenizer.chat_template` 经 minja vs 参考 Jinja2 输出，严格计数 mismatch |
| `test_response_parser` | 无 | CPU | `reasoning_content`/`tool_calls` 流式拆分（整段与逐字节喂入一致） |
| `test_compare` | 0.8B | CPU | CPU 参考各阶段 vs llama.cpp tensor dump |
| `test_cpuref` | 0.8B | CPU | CPU 参考 forward head on token ids（默认 `{9419}`） |
| `test_sampler` | 无 | CPU | `logit_bias` 强制/封禁 token、logprob log-softmax 与归一化、best_of 打分路径 |
| `test_cpu_gemv` | 0.8B | CPU | CPU 融合 fp32 + 整数 int8 GEMV/RMSNorm vs `quant.h` 主机反量化参考，多类型多 TB；`PF_CPU_ISA=scalar\|avx2\|avx512` 钉住变体 |
| `test_cpu_gdn` | 无 | CPU | `cpu_gdn` 用**非对称 head 数**（`n_k_heads=2, n_heads=6`）对照测试内标量参考，覆盖 `head % n_k_heads` 配对与状态写回（0.8B 的真实维度是恒等映射，测不出该 bug） |
| `test_w4` | 0.8B | CPU | w4 原生宽度打包：unpack(pack(Q4_K)) 往返（~0.077% rel L2，int8 转换是 ~0.98%）+ prefill 的 grouped-scale 恒等式 |
| `test_pc_cpu` | 0.8B | CPU | 主机后端的 paged 注意力 + 前缀缓存磁盘 spill/promote 往返（日志逐位一致） |
| `test_pc_disk` | 无 | CPU | 磁盘层记录格式往返、token 校验、LRU 预算、重开持久化、损坏/未知记录 |
| `test_pc_ram` | 无 | CPU | RAM 层 LRU 记录存储 |
| `test_multimodal` | 0.8B + mmproj | CPU+GPU | 图像预处理几何、host/device 视觉编码器、位置与 prompt 布局、逐 kernel 对照；音频解码/mel、AuT 塔 |
| `test_gpu_stages` | 0.8B | GPU | 每个 kernel vs CPU 参考（强制 `PF_DP4A=0`）；调用 rmsnorm/embed/qk_norm_rope/attn/conv/gdn/gated_norm |
| `test_gemv` | 0.8B | GPU | 每个真实 GGUF 张量的 GEMV vs CPU 反量化参考，多 TB（强制 `PF_DP4A=0`） |
| `test_dp4a` | 0.8B | GPU | SIn 重排 + DP4A GEMM vs 同量化输入的 CPU 参考，报告相对 fp32 的量化误差 |
| `test_w4_gemm` | **27B** | GPU | u4 prefill GEMM（`dnnl_gemm::gemm_w4`）vs 真实反量化权重的 fp32 参考（残差应只含权重表示误差 ~0.08%）；`PF_W4_RATE=1` 在跑慢参考之前先做 kernel 速率探针（`N >= 4096`、200 次迭代）并退出，让 kernel 调优迭代在约 20 s 内 |
| `test_k5_gemv` | **27B** | GPU | 原生 5-bit（Q5_K）decode GEMV vs `dequantize_block_q5_K` 的 native q5 参考（应精确到 f32 舍入，实测 max rel 5.7e-08） |
| `test_w4_vs_i8` | **27B** | GPU | 同一批量化激活下 u4 GEMM vs int8 GEMM 的逐张量比较，用于定位 u4 打包/元数据的偏差 |
| `test_gemv_stride` | **27B** | GPU | `gemv_group` 在引擎真实参数组合下的探针：`x_stride == K` vs `2*K`（gate/up 交错）、`residual == nullptr` vs `out`（原地残差） |
| `test_iq_dequant` | **27B** | GPU | IQ*/Q3_K 的 GPU 反量化/GEMV vs `quant.h` 主机参考（缺失张量跳过） |
| `test_quant_audit` | **27B** | GPU | 每种 ggml 类型取一张张量，比较 int8(oneDNN) GEMM 与"真实反量化权重"的 fp32 参考，量化 int8 转换对每种类型的损失 |
| `test_gpu_vs_ref` | 0.8B | GPU | 端到端 GPU logits vs `cpu_ref`（强制 `PF_DP4A=0`） |
| `test_forward` | 0.8B | GPU | 端到端 logits/top-k 与校验和（强制 `PF_DP4A=0`） |
| `test_w4_vs_cpuref` | **27B** | GPU | u4 权重路径的端到端 logits vs fp32 CPU 参考；`TEST_LAYER_MAP` 跑分卡 |
| `test_w4_topk` | **27B** | GPU | 打印 chat prompt 的 top-12 prefill logits（u4 vs int8 对比）；`PF_DUMP_LOGITS=<path>` 落盘整条 logit 向量 |
| `test_27b_prefill` | **27B** | GPU | 27B 分卡 prefill 的接线/阈值排查（依次加长 prompt），带 `TEST_LAYER_MAP`/`TEST_DEVICE`/`TEST_FP32`/`TEST_KERNEL_BENCH`/`TEST_SKIP_DECODE`/`STOP_AFTER_LAYER` |
| `test_decode_vs_prefill` | 0.8B | GPU | **单 token decode 与同序列重新 prefill 必须给出同一预测**；纯 prefill 的测试对解码路径是盲区，这个测试是唯一覆盖 |
| `test_pc_gpu` | 0.8B | GPU | 磁盘 spill + promote 往返，第二次相同 prompt 从磁盘恢复需与首次 logits 一致 |
| `test_pc_ram_gpu` | 0.8B | GPU | VRAM→RAM→VRAM 往返；新引擎不得看到陈旧 RAM 条目 |
| `test_dflash_kernels` | 无 | GPU | DFlash top-k 与卷积的合成输入对照 |
| `test_spec` | 27B + draft GGUF | 双 GPU | MTP/DFlash2 对**显式绕过投机**的 plain greedy oracle；`TEST_SPEC_LONG=1` 另测跨 512-token prefill 和缓存复用 |

`TEST_LAYER_MAP` 是多设备 split 的通用开关（`test_w4_vs_cpuref`、`test_w4_topk`、`test_27b_prefill` 读它），
`TEST_DEVICE` 只有 `test_27b_prefill` 读。`test_gpu_stages` 除了 `argv[1]` 的模型，还从 `argv[2..]`
读要跑 stage 的 token id 列表（不传则用 `stage_test.h` 的默认集合）——它也是唯一一个默认模型路径不写
在 `main()` 里而藏在 `stage_arg_model()` 的测试。

### 2.1 CTest 分层注册

默认 `ctest --test-dir build -L hermetic --output-on-failure` 仅运行不依赖
GGUF 的五个 CPU 测试及 `test_multimodal --video-only` 的合成视频/视觉宽度
测试（ffmpeg 可缺席，AVI 测试仍运行）；它们必须已构建。模型测试必须显式配置
`-DSI_REGISTER_GPU_TESTS=ON -DSI_TEST_08B_MODEL=/path/to/0.8b.gguf`；
另外给 `SI_TEST_27B_MODEL`（可再给 `SI_TEST_DFLASH_MODEL`）才注册双卡 27B
测试，自动设置 `TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1`。用
`ctest --test-dir build -L model-0.8b` / `-L model-27b` 分档运行，
不得把没注册或没运行的模型档称为通过。`test_pc_cpu` 是已知失败，
修复前没有纳入 hermetic 默认档。

### 2.2 stage 测试脚手架

* `tests/backend/gpu/kernels/stage_tests.h` 声明七个 stage 入口。
* 每个 `*_stage.cpp` 定义 `void stage_<kernel>(si::stage_env & env)`，用 `env.get(...)` 取快照、
  `env.cmp(...)` 比较（见 `tests/common/stage_test.h`）。
* `tests/common/cpu_ref.h` 提供主机参考前向，被 `test_cpuref`、`test_compare`、`test_gpu_vs_ref`、
  `test_w4_vs_cpuref` 以及 `stage_test.h` 使用。

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
./build/test_forward        # default prompt predicted_argmax=248068 (input_last_id is not a prediction)
./build/test_decode_vs_prefill   # OK
```

**四个都要跑。** 前三个是 prefill-only，对 decode-only 的 bug 完全盲（§7.2 第 6 条就是这个 bug 的藏身处），
所以**任何影响单 token 路径的改动都必须跑 `test_decode_vs_prefill`**。

### 3.2.1 已知失败（2026-10-03，两台机器都复现，与当次改动无关）

`test_pc_cpu`（CPU 后端的前缀缓存往返）**当前是失败的**，在本地 Iris Xe-LP 与 2x A770 上都复现：
最后一行是 `pc_cpu: 3 check(s) failed` /
`matched=96 logits max|diff|=3.840658 argmax 248046/198 DIFFERENT`——命中了 96 个 token 的前缀，但
从缓存恢复后的 logits 与参考不一致（`max|diff|` 在两台机器上分别约 3.37 与 3.84，说明这条 CPU 路径
本身还有随线程数/ISA 变化的数值成分）。改动前（`6d77f95` 的 AOT 二进制）同样失败，所以它是一个
**尚未跟踪的既存缺陷**，不是回归；CPU 分区的 `pc_restore_state` / 主机反量化路径值得单独查。
其余测试（0.8B 全套 + 27B 的 `test_27b_prefill` / `test_w4_topk` / `test_w4_gemm` /
`test_w4_vs_cpuref` / `test_decode_vs_prefill`）在 2x A770 上全部通过。

另一个环境陷阱：同一台 A770 上连续跑多个 27B 测试时，后面的运行可能以
`level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` 失败（第一个失败的测试之后
卡片状态就坏了）。**单独跑每一个**就都通过（`test_w4_vs_cpuref` 单独跑：
`max|diff|=0.4448 mean|diff|=0.0528 argmax gpu=248045 ref=248045 SAME`），所以遇到 DEVICE_LOST
先单独重跑该测试，别急着改代码。

### 3.3 27B（需要分卡，单卡放不下）

```bash
M=/data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1 ./build/test_w4_vs_cpuref      # argmax SAME
TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1 ./build/test_decode_vs_prefill "$M"  # OK
```

`test_w4_vs_cpuref` 默认就是 27B（可以不带路径），而 `test_decode_vs_prefill` 默认是 0.8B，**必须显式传
`"$M"`**——否则第二个命令测的是 0.8B 而不是 27B。

### 3.4 0.8B 的 tied embedding 与 hybrid map

0.8B 是**共享 embedding**（GGUF 里没有 `output.weight`）的参考模型，也是唯一能触发"多设备 tied LM head"
这条路径的模型，两种 split 都要跑：

```bash
TEST_LAYER_MAP=0-11:gpu.0,12-23:gpu.1 ./build/test_decode_vs_prefill  # OK
TEST_LAYER_MAP=0-11:gpu.0,12-23:cpu   ./build/test_decode_vs_prefill  # OK
```

### 3.5 include-cleaner lint（`Diagnostics.UnusedIncludes: Strict` 必须干净）

```bash
clang-tidy -p build --extra-arg=-I/opt/intel/oneapi/compiler/2026.1/include \
  -checks='-*,misc-include-cleaner' <changed files>
```

### 3.6 构建"看起来没生效"时

`rsync -a`（以及任何保留 mtime 的拷贝）会让源文件的时间戳早于目标文件，`make` 因此不重编。怀疑自己的改动
被忽略时，先统一刷新时间戳：

```bash
find src tests -name '*.cpp' -o -name '*.h' | xargs touch
```

---

## 4. 运行期前置

二进制/测试启动需要 oneAPI 运行库在 `LD_LIBRARY_PATH` 上。`source /opt/intel/oneapi/setvars.sh` 即可；
CI 式的最小覆盖是：

```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:$LD_LIBRARY_PATH
```

`ld.bfd` 关于 `libsvml.so` / `libimf.so` / `libintlc.so.5`（`libdnnl.so` 需要）的告警是**良性**的：它们
在运行期从 oneAPI runtime 路径解析。

CLI/服务用法见 [../../README.md](../../README.md) 与 [01-model-loading.md](01-model-loading.md)；
AOT/JIT 缓存在 §1.4。

---

## 5. 仓库目录与文档对应

```
src/device/     → docs/design/03-kernels.md（kernel 调优常量按卡分文件）, AGENTS.md 的 Device profiles
src/common/     → docs/design/02-quantization.md
src/backend/    → docs/design/02-quantization.md
src/backend/gpu/kernels/    → docs/design/03-kernels.md §1-13
src/backend/cpu/kernels/    → docs/design/03-kernels.md §14, docs/architecture.md §11
src/backend/cpu/    → docs/design/03-kernels.md §14, docs/architecture.md §11（设备选择与多设备执行）
src/model/      → docs/design/01-model-loading.md, 08-tokenizer.md, 11-qwen35-model.md
src/mm/         → docs/design/10-multimodal.md, 13-audio-video.md
src/engine/     → docs/design/04-engine.md, 05-kv-cache.md, 06-prefix-cache.md, 07-sampler.md
src/server/     → docs/design/09-server.md
dev/            → 本文档 §1.5（本地未跟踪，不属于本仓库）
tests/backend/gpu/  → docs/design/03-kernels.md, 06-prefix-cache.md
tests/backend/cpu/  → docs/architecture.md §11, docs/design/06-prefix-cache.md
tests/common/       → 本文档（共享 harness）
tests/model/, tests/mm/ → 本文档
tests/server/, tests/engine/ → docs/design/09-server.md, 07-sampler.md
```

---

## 6. 测量报告不进仓库，数字的归档位置

一次性的性能/正确性报告写在 `reports/`，该目录在 `.gitignore` 里，**内容永远不会被提交**，所以
仓库内任何文档都不引用它。需要保留的数字必须连同测量条件（硬件、开关、环境变量）写进本文档或
`AGENTS.md`；性能数字尤其必须注明所用的环境变量，因为不同开关会选择不同 kernel 变体。

### 6.1 实测数字归档在哪

不建指向 `reports/` 的索引（那必然是死链），而是记录每个主题的数字**最终落在哪个已提交的文档里**：

| 主题 | 归档位置 |
|---|---|
| 构建/AOT/测试矩阵、快速验证清单 | 本文档 §1-§3 |
| decode 的 attention 占用率、`PF_DEC_SPLIT` 悬崖、8 warps/EU | `AGENTS.md` 的 Environment variables |
| dp4a vs DPAS 的 issue 率、算术强度、M\* 的推导 | `AGENTS.md` 的 MTP 章节 |
| GEMV/native store 的实测带宽、per-format B/weight | `AGENTS.md` 的 Weight representation |
| 逐 kernel 的布局与 launch geometry | [03-kernels.md](03-kernels.md) |
| 逐调优常数的出处（哪个数在哪张卡上测的） | `AGENTS.md` 的 Device profiles |

一条测量只有在"数字 + 硬件 + 环境变量 + 怎么复现的命令"四件齐全时才可以被引用；只写数字不写条件
的数字在下次复现时会变成噪音。

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
    时先 `find src tests -name '*.cpp' -o -name '*.h' | xargs touch` 再构建（见 §3.6）。
11. **怀疑第三方库"版本旧"之前先核对**。`minja` 的 `is undefined` 缺口在**上游 HEAD 同样存在**：
    用 `git hash-object third_party/minja/minja.hpp` 与上游 blob 比对，确认我们的副本逐字节等于上游
    HEAD，升级无解，只能本地打补丁（见 [AGENTS.md](../../AGENTS.md) 的 third_party 例外）。

### 7.3 可复用的工具

下表除环境变量外的两个工具都不在本仓库里（`ll_logits` 链接 llama.cpp 的 `libllama`，数值比较是一次性
Python 脚本），按描述重建即可。

| 工具 | 用途 |
|---|---|
| `ll_logits`（仓库外，一次性小程序，链接 llama.cpp 的 `libllama`） | 对显式 token id 序列 dump llama.cpp 的 logits（逐位置），作为独立参照 |
| max\|diff\| / mean\|diff\| / corr / top-20 overlap / argmax 比较 | 数值口径；27B prefill 的判据见 §7.1 第 2 条 |
| `PF_DUMP_LAYERS` / `PF_DUMP_RAW` | 逐层指纹与整条激活向量；对比 decode 与 prefill 时用"最后真实 token slot"（`layerlast`）与"单个 decode 行" |
| `PF_DUMP_SEGS=<layer>` | 单层内逐 GEMV segment 的 hidden 指纹（`-1` = 全部）；确认某个开关是否真的换了路径 |
| `PF_DUMP_DEC_LOGITS` / `PF_DUMP_GEN` | decode 每一步采样器看到的 logits、采样结果与停止原因 |
| `PF_DUMP_PROMPT` | 打印模型实际被喂进去的 id（chat 模板渲染结果）——区分"模板坏"与"前向坏" |
| `PF_PROF` / `PF_HOSTPROF` | 调用组耗时分解；`PF_HOSTPROF` 不带 per-stamp 设备等待，量的是**提交**开销——区分"kernel 慢"与"launch 慢"的唯一办法 |
| GGUF 张量形状 dump | 核对所有宽度假设 |
