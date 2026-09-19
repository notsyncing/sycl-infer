# sycl-infer 设计文档

本目录是 `sycl-infer` 的架构设计与各功能设计文档。文档以源码为准，所有关键结论都标注了对应的
源文件与行号；遇到与注释冲突时，以代码实现为准（个别历史注释已过时，文中会指出）。

## 阅读顺序

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [architecture.md](architecture.md) | 总体架构、模块划分、运行时对象模型、数据流、线程模型、设备内存布局、全局不变量 |
| 2 | [design/01-model-loading.md](design/01-model-loading.md) | GGUF 解析、内存映射、架构注册表、张量绑定、权重一次性上传、CLI 参数 |
| 3 | [design/02-quantization.md](design/02-quantization.md) | ggml K-quant 布局、SIn int8 权重格式、DP4A 数学、激活量化、oneDNN 路径 |
| 4 | [design/03-kernels.md](design/03-kernels.md) | 各 SYCL kernel 的设计与数据布局：norm/embed/gemv/attn/qk-rope/conv/gdn/gated-norm/vit |
| 5 | [design/04-engine.md](design/04-engine.md) | `engine` 编排、`seg_plan`、`record_forward`、SYCL command graph、prefill/decode、连续批处理 |
| 6 | [design/05-kv-cache.md](design/05-kv-cache.md) | 分页 KV cache、动态虚拟内存块池、KV 存储类型、块分配器 |
| 7 | [design/06-prefix-cache.md](design/06-prefix-cache.md) | 跨请求前缀缓存、链式哈希、递归状态检查点、VRAM/RAM/磁盘三层 LRU |
| 8 | [design/07-sampler.md](design/07-sampler.md) | 采样算法：penalty、temperature、top-k/top-p/min-p、`logit_bias`、logprobs |
| 9 | [design/08-tokenizer.md](design/08-tokenizer.md) | GPT-2 byte-level BPE、特殊 token、chat template |
| 10 | [design/09-server.md](design/09-server.md) | OpenAI 兼容 HTTP 服务（chat/completions/models、reasoning_content、工具调用、logprobs、usage 缓存统计）、连续批处理调度器、SSE 流式、CORS |
| 11 | [design/10-multimodal.md](design/10-multimodal.md) | 图像预处理、视觉编码器（host/device）、M-RoPE 组装 |
| 12 | [design/11-qwen35-model.md](design/11-qwen35-model.md) | Qwen3.5 混合架构（Gated DeltaNet + full attention）的加载与执行 |
| 13 | [design/12-build-and-testing.md](design/12-build-and-testing.md) | 构建系统、测试矩阵、验证流程、测量报告索引 |
| 14 | [design/13-audio-video.md](design/13-audio-video.md) | 音频与视频输入：解码、音频编码塔、视频/音频 M-RoPE、混合 prompt 装配 |

## 快速定位

* 想新增一个模型架构 → [design/01-model-loading.md](design/01-model-loading.md)、[design/11-qwen35-model.md](design/11-qwen35-model.md)
* 想新增一个 kernel（GPU 或 CPU）→ [design/03-kernels.md](design/03-kernels.md)、[design/12-build-and-testing.md](design/12-build-and-testing.md)
* 想加图像/视频/音频输入 → [design/10-multimodal.md](design/10-multimodal.md)、[design/13-audio-video.md](design/13-audio-video.md)
* 想理解图捕获与模式 0/1/2 → [design/04-engine.md](design/04-engine.md)
* 想看设备选择 / CPU 后端 / 多设备（`--device`、`--layer-map`）→ [architecture.md §11](architecture.md)、[design/04-engine.md §11](design/04-engine.md)
* 想调优内存/前缀缓存 → [design/05-kv-cache.md](design/05-kv-cache.md)、[design/06-prefix-cache.md](design/06-prefix-cache.md)
* 环境变量总表 → [AGENTS.md](../AGENTS.md#environment-variables)

## 与其它文档的关系

* [`../README.md`](../README.md)：面向用户的功能、构建、使用说明。
* [`../AGENTS.md`](../AGENTS.md)：面向开发者的约定、不变量、环境变量、扩展步骤。
* [`../reports/`](../reports/)：一次性的性能/正确性测量报告（前缀缓存、长上下文、VTune 等）。
* `docs/`（本目录）：稳定的系统级设计文档。

设计文档描述“为什么这样设计”和“各模块如何协作”；测量数据、一次性调试结论仍以 `reports/` 与
代码注释中的实测数字为准。
