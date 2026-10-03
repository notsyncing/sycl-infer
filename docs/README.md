# sycl-infer 设计文档

本目录是 `sycl-infer` 的架构设计与各功能设计文档。文档以源码为准，所有关键结论都标注了对应的
源文件与行号；遇到与注释冲突时，以代码实现为准（个别历史注释已过时，文中会指出）。

## 阅读顺序

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [architecture.md](architecture.md) | 总体架构、模块划分、运行时对象模型、数据流、线程模型、设备内存布局、命令行 flag、MTP 路径、全局不变量 |
| 2 | [design/01-model-loading.md](design/01-model-loading.md) | GGUF 解析、内存映射、架构注册表、张量绑定（含 MTP/NextN 头）、权重一次性上传与主机页回收、CLI 参数 |
| 3 | [design/02-quantization.md](design/02-quantization.md) | ggml K-quant 布局、SIn int8 权重格式、原生位宽存储（u4 / 5-bit / codebook 4-bit / 2-bit）、DP4A 数学、激活量化（`do_split`）、oneDNN 路径 |
| 4 | [design/03-kernels.md](design/03-kernels.md) | 各 SYCL kernel 的设计与数据布局：norm/embed/gemv/nat_gemm/attn（含 oneDNN int8 prefill attention）/qk-rope/conv/gdn/gated-norm/vit/at/MTP，以及设备 profile 如何提供形状 |
| 5 | [design/04-engine.md](design/04-engine.md) | `engine` 编排、`seg_plan`、`record_forward`、SYCL command graph、prefill/decode、连续批处理、多设备分层交接、MTP/NextN 投机解码 |
| 6 | [design/05-kv-cache.md](design/05-kv-cache.md) | 分页 KV cache、动态虚拟内存块池、KV 存储类型与 K/V 独立定尺、块分配器、池的定尺 |
| 7 | [design/06-prefix-cache.md](design/06-prefix-cache.md) | 跨请求前缀缓存、链式哈希、递归状态检查点、VRAM/RAM/磁盘三层 LRU |
| 8 | [design/07-sampler.md](design/07-sampler.md) | 采样算法：penalty、temperature、top-k/top-p/min-p、`logit_bias`、logprobs、随机数与可复现性 |
| 9 | [design/08-tokenizer.md](design/08-tokenizer.md) | GPT-2 byte-level BPE、特殊 token、chat template |
| 10 | [design/09-server.md](design/09-server.md) | OpenAI 兼容 HTTP 服务（chat/completions/models、reasoning_content、工具调用、logprobs、usage 缓存统计）、连续批处理调度器、SSE 流式、CORS、字段级支持矩阵 |
| 11 | [design/10-multimodal.md](design/10-multimodal.md) | 图像预处理、视觉编码器（host/device）、M-RoPE 组装 |
| 12 | [design/11-qwen35-model.md](design/11-qwen35-model.md) | Qwen3.5 混合架构（Gated DeltaNet + full attention）的加载与执行 |
| 13 | [design/12-build-and-testing.md](design/12-build-and-testing.md) | 构建系统（含 AOT）、测试矩阵、验证流程、调试手册（27B 案例分析）与经验教训、实测数字的归档位置 |
| 14 | [design/13-audio-video.md](design/13-audio-video.md) | 音频与视频输入：解码、音频编码塔、视频/音频 M-RoPE、混合 prompt 装配 |

## 快速定位

* 想新增一个模型架构 → [design/01-model-loading.md](design/01-model-loading.md)、[design/11-qwen35-model.md](design/11-qwen35-model.md)
* 想新增一个 kernel（GPU 或 CPU）→ [design/03-kernels.md](design/03-kernels.md)、[design/12-build-and-testing.md](design/12-build-and-testing.md)
* 想新增一张 GPU 卡的调优，或改 launch 形状 → [architecture.md §11](architecture.md)、[design/03-kernels.md §1.3](design/03-kernels.md)
* 想加图像/视频/音频输入 → [design/10-multimodal.md](design/10-multimodal.md)、[design/13-audio-video.md](design/13-audio-video.md)
* 想理解图捕获与模式 0/1/2 → [design/04-engine.md](design/04-engine.md)
* 想看设备选择 / CPU 后端 / 多设备（`--device`、`--layer-map`）→ [architecture.md §11](architecture.md)、[design/04-engine.md §11](design/04-engine.md)
* 想动权重存储（u4 / k5 / cb4 / 2-bit）或多设备 oneDNN 权重路径 → [design/02-quantization.md](design/02-quantization.md)、[design/03-kernels.md §5](design/03-kernels.md)
* 想动 MTP / 投机解码 → [design/04-engine.md §12](design/04-engine.md)、[design/03-kernels.md §15](design/03-kernels.md)
* 想让 prefill 更快（batch 形状、oneDNN prefill attention）→ [design/04-engine.md §7](design/04-engine.md)、[design/03-kernels.md §5.8](design/03-kernels.md)、[design/03-kernels.md §7.5](design/03-kernels.md)
* 想调优内存/前缀缓存 → [design/05-kv-cache.md](design/05-kv-cache.md)、[design/06-prefix-cache.md](design/06-prefix-cache.md)
* 环境变量总表 → [AGENTS.md](../AGENTS.md#environment-variables)
* 排查"模型输出不正常"（decode/prefill 不一致、量化路径差异、参考实现不可信等）→ [design/12-build-and-testing.md §7](design/12-build-and-testing.md)

## 与其它文档的关系

* [`../README.md`](../README.md)：面向用户的功能、构建、使用说明。
* [`../AGENTS.md`](../AGENTS.md)：面向开发者的约定、不变量、环境变量、扩展步骤。
* `docs/`（本目录）：稳定的系统级设计文档。

设计文档描述“为什么这样设计”和“各模块如何协作”；测量数据、一次性调试结论以代码注释中的实测数字为准。

**一次性测量报告不进仓库。**`reports/` 在 `.gitignore` 里，其中内容永远不会被提交，所以仓库内
任何文档都不引用它（含本文件）。有价值的数字必须连同它的测量条件一起写进本文档或 `AGENTS.md`。
