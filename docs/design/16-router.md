# 设计 16：多模型路由进程（router）

覆盖 `src/router/`（新增）、`src/server/control.{h,cpp}`（新增）、`src/main.cpp`（新增 flag）、
`CMakeLists.txt`（新增 target 与测试）。相关既有设计见
[09-server.md](09-server.md)（OpenAI 兼容 HTTP 与 SSE）、[05-kv-cache.md](05-kv-cache.md)（KV 池定尺）、
[06-prefix-cache.md](06-prefix-cache.md)（三层前缀缓存）。

---

## 1. 目标与非目标

### 1.1 目标

1. **一个对外端点**：客户端只跟路由器说话，请求里的 `model` 字段决定落到哪个模型。
2. **一模型一进程**：每个模型跑在独立的 worker 进程里，**进程即显存预算的边界**。
3. **声明式配置**：一个 JSON 文件描述所有模型（GGUF 路径、每模型的显存/上下文/前缀缓存配置、
   启动参数、环境变量）。
4. **生命周期由路由器管**：拉起、健康检查、崩溃重启、优雅停机、状态汇总。
5. **可容器化**：路由器与 worker 都能进 Docker，模型文件与缓存目录用卷挂载。
6. **worker 可在远程机器**：协议与客户端从第一天就是跨网的，路由器只负责"不 spawn、只连接"。

### 1.2 非目标（明确不做，并说明为什么）

| 不做 | 原因 |
|---|---|
| 跨模型批处理 | 权重不同，同一次前向无法同时算两个模型。连续批处理的收益全部来自"同一份权重下的多行并行"（`src/server/scheduler.h:132` 的 `engine & e` 就是 per-engine 的），跨模型无处安放 |
| GPU 显存硬分区 | Level Zero 没有显存分区/隔离能力（见 §9.3）。路由器能做的是**准入 + 预测 + 快速失败**，不是隔离 |
| 在路由器里重做 OpenAI 响应整形 | `reasoning_content` / `tool_calls` 的流式拆分、logprobs、`usage` 已经在 worker 侧实现并被 `test_response_parser` 覆盖（`src/server/response_parser.cpp`）。复制一份会引入"两套实现漂移"，且路由器成为所有流量的单点 |
| 一个进程内多个 engine | 已知会踩雷：`kd::gemm_ws()` 是**进程级单例且没有按 queue 索引**（`src/backend/gpu/kernels/dp4a_common.cpp:10`），两个 engine 同卡会共用同一块 K-split workspace；AGENTS.md 也记录了"同进程第二个 engine 继承第一个的状态"。见 §2.3 |

---

## 2. 为什么是"一模型一进程 + 路由器"

### 2.1 现有结构本来就是单模型

* `serve(engine & e, const server_config & cfg)`（`src/server/server.cpp:565`）——一个进程一个 engine。
* `/v1/models` 只返回一条（`src/server/server.cpp:672`），`model` 字段只回显不路由
  （[09-server.md](09-server.md) §8 的字段矩阵："仅回显/用于选择（单模型）"）。
* `engine` 独占：GGUF 映射、device 权重、KV 池、递归状态、三层前缀缓存、oneDNN primitive 缓存。
  这些全是"每模型一份"，没有共享成分（除非两个模型共享 base 权重 + LoRA，本项目不支持）。

### 2.2 多模型拿不到吞吐收益

单模型 decode 已经是带宽瓶颈：27B 每 token 读 ~16.3 GB 权重、67 ms/token ≈ 243 GB/s，
而实测卡上限 ~376 GB/s（AGENTS.md "The arithmetic-intensity ceiling"）。同卡放第二个模型不是并行，
是分时；Level Zero 是设备级调度，不同 context 之间是时间片而非并发——**同进程两个 queue 也一样**。

所以"多模型"的收益全部在**运维面**（隔离、路由、健康、预算），不在吞吐面。进程边界正好是运维面
需要的边界。

### 2.3 同进程多模型的代价（不做它的理由）

| 项 | 现状 |
|---|---|
| `kd::gemm_ws()` | 进程级单例，未按 queue 索引（`src/backend/gpu/kernels/dp4a_common.cpp:10`），被 `dp4a_gemv.cpp` / `dp4a_gemm.cpp` 的 decode 路径使用。同卡两个 engine → 静默数据竞争 |
| 同进程第二个 engine | AGENTS.md 已记录："an MTP-enabled engine followed by a DFlash one fails at construction"；`test_spec` 因此被设计成"一配置一进程" |
| `engine_dflash.cpp` 的 `static sycl::event` | 进程级 |
| oneDNN primitive 缓存 | 每个 `dnnl_gemm` 实例一份 → 每 engine 一份（**跨进程也一样，省不了**） |

### 2.4 显存预算已有现成旋钮

"每个模型显式指定显存占用"这件事，项目里已经是一等公民，不需要新机制：

| 成分 | 控制项 | 27B / 2×A770 实测 |
|---|---|---|
| 权重（固定） | `--layer-map` 分卡 | 9 072 MiB/卡（含 1 248 MiB 的 int8 LM head） |
| 激活 + 递归状态 | `--max-slots` | 1 484 MiB/卡（其中 1 125 是 `gdn_state`，16 slot；单流用 1） |
| oneDNN + attn_xmx scratch | 固定 | 318 MiB |
| KV 池 | `--kv-cap-mb` / `--blocks` | 剩余 4 667 MiB；虚拟 USM 只预留地址、按需提交物理页（`src/engine/engine_kvpool.cpp:342`、`:411`） |
| 前缀缓存 VRAM 层 | `--pc-vram-mb` | 见 [06-prefix-cache.md](06-prefix-cache.md) |
| MoE 专家库 | `--moe-vram-mb` | 见 [15-moe-ncmoe.md](15-moe-ncmoe.md) |

这些全是 per-process 的命令行参数。路由器只是把它们从配置文件映射过去（§4.3）。

---

## 3. 总体架构

> **现状**：M1/M2 已实现的是"数据面 + 生命周期"这一半；控制面（§5 与 §7）是 M3。
> 下图是目标形态。

```
                     ┌──────────────────────────────────────────────────┐
   OpenAI 客户端 ───► │  sycl-infer-router   (C++17，无 SYCL/无 oneDNN) │
   HTTP + SSE        │  · 配置装载 + 模型注册表                          │
                     │  · supervisor：spawn / 监控 / 重启 / 优雅停机      │
                     │  · 控制面 RPC 客户端（JSON-RPC 2.0 / NDJSON / TCP）│
                     │  · 数据面 HTTP 代理（SSE 逐块透传）                │
                     │  · 显存预算记账 + 准入                            │
                     └───────────┬───────────────────────────┬──────────┘
                                 │ 控制面 RPC（TCP）          │ 数据面 HTTP
                                 ▼                            ▼
                  ┌────────────────────────┐      ┌────────────────────────┐
                  │ worker（本地）          │      │ worker（远程，预留）    │
                  │ sycl-infer serve       │      │ sycl-infer serve       │
                  │ http  127.0.0.1:18081  │      │ http  10.0.0.x:18081   │
                  │ rpc   127.0.0.1:18082  │      │ rpc   10.0.0.x:18082   │
                  └────────────────────────┘      └────────────────────────┘
```

### 3.1 两个平面

| 平面 | 协议 | 用途 | 远程就绪 |
|---|---|---|---|
| **控制面** | JSON-RPC 2.0 over TCP（NDJSON 分帧） | 生命周期、健康、显存/运行时统计、模型元数据、停机 | 是（TCP） |
| **数据面** | HTTP（worker 已有的 OpenAI 接口） | 生成请求与响应（含 SSE） | 是（HTTP） |

### 3.2 为什么数据面不也走 RPC

worker 已经实现并测试了完整的 OpenAI 表面：SSE 分帧、`reasoning_content` / `tool_calls` 的流式拆分
（`src/server/response_parser.cpp`，被 `test_response_parser` 覆盖）、logprobs、`usage` 的
`cached_tokens` 统计。把这些搬进 RPC 意味着路由器要重做一遍响应整形（§1.2 的非目标）。

控制面则相反：它承载的是**现在还不存在**的能力（生命周期、显存上报、模型注册表），而且必须是
跨网可用的。两者的边界因此是"新的走 RPC，已有的走 HTTP"。

**预留**：RPC 的帧格式与传输（TCP + NDJSON + JSON-RPC 2.0）允许未来把数据面也搬上去——帧类型预留
`generate` 事件。但那需要路由器实现 OpenAI 整形，属于可选项而非路线图（§13）。

---

## 4. 配置文件

### 4.1 格式

JSON，**允许注释**：`//` 行注释与 `/* */` 块注释都会被剥离。实现上直接用 nlohmann 的
`json::parse(text, nullptr, true, /*ignore_comments=*/true)`（`src/router/config.cpp`），
不额外写预处理，所以注释的语义就是 JSON 的词法：字符串里的 `//` 是数据，不是注释
（`test_router_config` 有一条用例专门钉住这一点）。

为什么允许注释：这是一个人要改的配置文件，而 JSON 的语法里没有注释的位置。代价是
"这不是严格的 JSON"——别的工具读不了。所以格式选择被限制在这个仓库自己的解析器上，
将来若要支持 YAML，那是另一个决定。

### 4.2 示例

```jsonc
{
  "router": {
    "listen": "0.0.0.0:8080",
    "access_log": true,
    "auth": { "bearer_tokens": ["sk-local-only"] },
    "request_timeout_s": 3600,
    "startup_timeout_s": 900,
    "log_dir": "logs",
    "device_vram_mb": 15541
  },
  "defaults": {
    "binary": "/usr/local/bin/sycl-infer",
    "env": { "PF_PREFIX_CACHE": "1" }
  },
  "models": [
    {
      "id": "qwen35-0.8b",
      "path": "/models/Qwen3.5-0.8B-Q4_K_M.gguf",
      "http": "127.0.0.1:18081",
      "args": ["--ctx", "20480"],
      "device": { "group": "gpu.0", "filter": "level_zero:0" },
      "memory": {
        "vram_budget_mb": 4096,
        "kv_cap_mb": 2048,
        "max_slots": 8,
        "prefix_cache_vram_mb": 512,
        "prefix_cache_dir": "/var/lib/sycl-infer/pc/qwen35-0.8b"
      }
    },
    {
      "id": "qwen38-27b",
      "path": "/models/Qwen3.8-27B-UD-Q4_K_M.gguf",
      "http": "127.0.0.1:18083",
      "args": ["--ctx", "131072"],
      "device": { "group": "gpu.0,gpu.1", "layer_map": "0-31:gpu.0,32-63:gpu.1" },
      "memory": {
        "vram_budget_mb": 30720,
        "kv_cap_mb": 4096,
        "max_slots": 1,
        "prefix_cache_vram_mb": 0
      }
    },
    {
      "id": "remote-35b",
      "remote": { "http": "http://gpu-host-b:19081" }
    }
  ]
}
```

### 4.3 字段语义

| 字段 | 必填 | 含义 |
|---|---|---|
| `router.listen` | 否 | 路由器的 HTTP 监听 `host:port`（默认 `127.0.0.1:8080`）。**对外暴露要放在反向代理后面**（worker 的 HTTP 本身无鉴权，见 [09-server.md](09-server.md)） |
| `router.auth.bearer_tokens` | 否 | 路由器层的 Bearer 鉴权（也接受顶层 `router.bearer_tokens`）。`/health` 始终免鉴权——探针不该带 token |
| `router.access_log` | 否 | 每个请求一行日志（方法、路径、状态、模型、耗时） |
| `router.request_timeout_s` | 否 | 数据面代理的上游超时（流式请求要 >= 生成时长） |
| `router.startup_timeout_s` | 否 | 等一个 worker 就绪的上限。27B 加载 ~80 s，默认给足 |
| `router.health_interval_s` / `router.health_failures` | 否 | 存活探测的周期与连续失败次数（超过即判定 worker 失联并重启） |
| `router.max_restarts` / `router.restart_backoff_s` | 否 | 重启次数上限与首次退避（之后翻倍） |
| `router.log_dir` | 否 | worker 日志目录，每个模型一个 `<id>.log` |
| `router.binary` | 否 | LLM worker 可执行文件的默认路径（默认 `sycl-infer`，按 PATH 找，找不到则找路由器自己旁边的） |
| `router.binary_onnx` | 否 | ONNX worker 可执行文件的默认路径（默认 `sycl-infer-onnx`） |
| `router.device_vram_mb` | 否 | **声明**的单卡可用显存。给了它，路由器就在配置阶段检查同组模型的预算之和（§9） |
| `defaults.env` | 否 | 传给所有 worker 的环境变量（`"K=V"` 形式） |
| `defaults.binary` | 否 | worker 可执行文件的默认路径 |
| `models[].id` | 是 | 客户端 `model` 字段用的名字，路由键，必须唯一且不含空白 |
| `models[].backend` | 否 | `llm`（默认，GGUF 引擎）或 `onnx`（文档流水线运行时） |
| `models[].path` | 本地必填 | GGUF/ONNX 路径（在 worker 的挂载命名空间里）。**必须在路由器所在的主机上也存在可读**——路由器在启动前校验它（早期发现挂载漏了，比等到 worker 启动失败好） |
| `models[].binary` | 否 | 该模型 worker 的可执行文件 |
| `models[].http` | 本地必填 | worker 的 OpenAI HTTP 地址 `host:port` |
| `models[].args` | 否 | 追加到派生命令行之后的额外参数。**`--model` / `--host` / `--port` 由路由器生成，不允许覆盖**（冲突即配置错误） |
| `models[].env` | 否 | 传给该 worker 的额外环境变量（与 `defaults.env` 合并，子进程优先） |
| `models[].device` | 否 | 要么是一个字符串（就是 `--device` 的取值：`cpu` / `gpu` / `auto`），要么是下面这些字段组成的对象。字符串形式是常用写法 |
| `models[].device.group` | 否 | 预算分组键：同组的模型视为共用一组卡，它们的预算必须相加。**命名沿用设备语法**：单卡 `gpu.0`，多卡 `gpu.0,gpu.1`（与 `--layer-map` 里的 device token 一致）；它只是分组标签，路由器不解析它 |
| `models[].device.kind` | 否 | 传给 worker 的 `--device`：`cpu` / `gpu` / `auto`（与 LLM 主程序相同）。`filter` 仍然负责钉卡；ONNX 默认 `gpu` |
| `models[].device.layer_map` | 否 | 传给 worker 的 `--layer-map`，**语法与命令行完全相同**：`begin-end:device,...`，device 为 `gpu` / `gpu.N` / `cpu`。路由器在启动时校验范围连续；`count` 未给时按其中不同的 GPU 数推导 |
| `models[].device.filter` | 否 | 传给子进程的 `SYCL_DEVICE_FILTER`，用来把 worker 钉到某张卡（**这是多进程方案白拿的好处**：单进程方案要改 `make_queue` 才能选卡） |
| `models[].device.count` | 否 | 这个模型跑在几张卡上（用 `--layer-map` 切分的模型是多张）。预算是**跨这些卡的总和**，所以静态检查用 `budget / count` 与单卡容量比较；省略时由 `layer_map` 推导；ONNX 没有 layer map，必须为 1 |
| `models[].onnx.drop_outputs` | 否 | ONNX 模型不读回的输出（→ `--drop-outputs`）；layout 的 48 MB mask 就是这里丢掉的 |
| `models[].onnx.dnnl` | 否 | ONNX 的 oneDNN f32 卷积/matmul 开关（→ `--dnnl 0/1`，默认 1） |
| `models[].control` | 否 | worker 的控制面地址 `host:port`（JSON-RPC over TCP）。给了它，路由器就用它拿模型身份与显存、并优雅停机；不给就退回 HTTP + 信号 |
| `models[].control_token` | 否 | 控制面的共享密钥（路由器在握手里出示）。**本地模型**：路由器会把它以 `--control-token` 传给 worker；**远程模型**：worker 自己起，令牌要一致 |
| `models[].control_token_file` | 否 | 从文件读密钥。容器里推荐这个：两端读同一个文件，令牌从不出现在命令行上 |
| `models[].memory.vram_budget_mb` | 否 | **路由器侧**的准入预算，不是 CLI 参数（§9） |
| `models[].memory.kv_cap_mb` | 否 | → `--kv-cap-mb N` |
| `models[].memory.max_slots` | 否 | → `--max-slots N` |
| `models[].memory.prefix_cache_vram_mb` | 否 | → `--pc-vram-mb N` |
| `models[].memory.prefix_cache_dir` | 否 | → `--pc-dir <dir>`（磁盘层目录；记录按模型指纹分子目录，多 worker 不冲突） |
| `models[].remote` | 否 | 非空表示**远程 worker**：路由器不 spawn，只连接（§10）。`remote.http` 可带 `http://` 前缀 |

### 4.4 派生的 worker 命令行

```
sycl-infer --model <path> --host <http.host> --port <http.port> \
           [--device K] [--layer-map L] \
           [--control-port N] [--control-token-file P] \
           [--kv-cap-mb N] [--max-slots N] [--pc-vram-mb N] [--pc-dir D] \
           [args...] serve
```

`K` 和 `L` 就是 worker 自己的 `--device` / `--layer-map` 取值，路由器不做语法转换。
`--layer-map` 也可以直接写在 `args` 里（同样的命令行语法），解析时会并入
`device.layer_map`；两处同时给是配置错误。

`--control-port` 由路由器派生（它要连的面就是 worker 要开的面）；令牌用**文件**形式传给 worker
（`--control-token-file`），这样令牌从不出现在命令行上——`ps` 是能看见命令行的。

ONNX 的派生命令行没有 `serve`，并且它的模型名就是路由键：

```
sycl-infer-onnx --model <id>=<path> --device <kind> --host <http.host> --port <http.port> \
                [--control-port N] [--control-token-file P] \
                [--drop-outputs NAME]... [--dnnl 0] [args...]
```

`id` 里不能出现 `=`（它是 `id=path` 的分隔符）；`device.count` 必须是 1。

`main.cpp` 的参数解析是**顺序无关**的（`src/main.cpp:231` 起的循环把任何非 `-` 开头的参数当作命令），
所以 `serve` 放最后即可。注意 `serve` 是唯一的非选项参数。

### 4.5 校验规则（配置错误必须在启动时失败，不是在第一次请求时）

1. `id` 唯一、非空、不含空白。
2. HTTP 端口不重复，且不与 `router.listen` 冲突。
3. 本地模型的 `path` 存在且可读。
4. `args` 里不出现 `--model` / `--host` / `--port` / `--control-port` / `--control-host` / `--device` / `--drop-outputs` / `--dnnl` / `--config`。`--layer-map` 允许出现在 `args`，并被并入 `device.layer_map`。
5. `kv_cap_mb` / `max_slots` 不为 0；`device.layer_map` 的范围从 0 开始且连续，device 只接受 `gpu` / `gpu.N` / `cpu`；ONNX 模型不接受 `layer_map`。
6. 同一 `device.group` 的 `vram_budget_mb` 之和 ≤ `router.device_vram_mb`（给了后者才检查）。
7. 远程模型不要求 `path`。
8. 所有超时与重试计数为正。

校验失败抛异常，`main` 打印 `error: ...` 并退出 1；`--check` 只校验不启动。

---

## 5. 控制面协议

> **已实现（M3）**：`src/common/jsonrpc.h`（协议，两端共用）+ `src/server/control.cpp`（worker 侧）
> + `src/router/rpc_client.cpp`（路由器侧）。

### 5.1 传输与分帧

* **TCP**，长连接，一条连接上多路复用请求（请求/响应按顺序匹配 id）。
* **NDJSON**：每行一个 JSON 值，行尾 `\n`。nlohmann 的 `dump()` 会把字符串内的换行转义成
  `\n`，因此一行一个 JSON 值是无歧义的，可以用 `nc` 直接调试。
* **JSON-RPC 2.0 语义**：`{"jsonrpc":"2.0","id":N,"method":"...","params":{...}}` →
  `{"jsonrpc":"2.0","id":N,"result":{...}}` 或 `{"jsonrpc":"2.0","id":N,"error":{"code":-32601,"message":"..."}}`。
  用标准协议而不是自定义格式，是为了让未来用别的语言写的远程 worker 能直接对接。
* **协议版本**：`hello` 带 `params.protocol`，版本不符回 `-32602`（路由器宁可拒绝，也不误读报告）。
  当前版本 **2**：版本 2 起，握手带**令牌**（`params.token`），握手之后的请求都算已认证。

### 5.2 认证

控制面能**停掉 worker**、能读它的设备账目，所以它不是可以随便暴露的面。做法：

* worker 侧：`--control-token <tok>` 或 `--control-token-file <path>`。**不配就是开放**——
  这是环回默认值下的既有行为，启动那行会明说（`[ctl] control plane on 127.0.0.1:18082 (token
  required)` / `(NO token: loopback only)`）。
* 路由器侧：`models[].control_token` / `models[].control_token_file`。
* 协议：**令牌只在握手里出现一次**，握手之后这条连接就认证过了。别的请求不带令牌，令牌也不
  在每条请求里反复上路。
* **文件形式是给容器用的**：`control_token_file` 让两端读**同一个文件**，令牌因此从不出现在
  命令行上（`ps` 能看到命令行）。路由器派生命令行时传的是 `--control-token-file <同一个文件>`。
* 比较是**常数时间**的（`rpc::secret_equal`）：令牌是共享密钥，而能连上控制面的对端恰恰是
  不该一字节一字节把它猜出来的那种。
* 失败一律回 `-32001`（JSON-RPC 的实现自定义区间 -32000..-32099），未认证的连接上**任何**方法
  （包括 `stats`）都被拒——握手是唯一的入口。
* **降级是优雅的**：令牌不对时，路由器的控制面不可用，它会退回 HTTP（就绪看 `/health`，
  身份看 `/v1/models`），模型照常服务，只是没有设备账本（于是没有实测准入，退回静态预算）。

### 5.3 方法表

| 方法 | 参数 | 结果 | 用途 |
|---|---|---|---|
| `hello` | `{"protocol":2,"token":"..."}` | `{"protocol":2,"worker":<report>}` | 握手 + 模型元数据 + 设备清单（§7.3） |
| `health` | — | `{"ready":true,"uptime_s":N,"inflight":N}` | 存活与就绪 |
| `stats` | — | `<report>` | 显存/池/缓存口径（§7.3） |
| `shutdown` | `{"timeout_s":30}` | `{"ok":true}` | 优雅停机：**走的是 worker 内部 SIGTERM 的同一条路**，所以前缀缓存会被 flush |
| `reload` | — | 错误 | 预留：热换权重（未实现） |

worker 可以在任何时刻主动推送**事件**（无 `id`，`method:"event"`）。协议里留了位置，但
v1 没有实现任何事件：worker 的日志走 stderr（路由器收进文件），不需要事件通道。

### 5.4 超时与心跳

* 路由器每 `health_interval_s`（默认 10 s）发一次 `stats`（它同时是存活检查和显存刷新）；
  连续 `health_failures`（默认 3）次失败 → 判定 worker 失联 → 重启。
* RPC 调用的超时独立于数据面：控制面默认 5 s，`shutdown` 用它自己的 `timeout_s`。
* **数据面请求绝不重试**（§6.3）。

### 5.5 安全

* worker 的 RPC 端口**默认只听 127.0.0.1**（`--control-port` 不给就没有这个面）。
* 远程 worker 需要显式 `--control-host 0.0.0.0`，且 worker 会打一行警告
  （对齐 `src/server/server.cpp:1056` 的既有做法）；配了令牌时那行警告就不打了。
* **令牌是 v1 的全部**：没有 mTLS，没有 per-peer 身份。令牌泄露 = 任何人都能停掉这个 worker
  并读它的设备账目，所以它应当像密码一样被对待（文件形式 + 0600）。

---

## 6. 数据面：HTTP 代理

### 6.1 路由器的 HTTP 表面

| 方法 | 路径 | 行为 |
|---|---|---|
| `POST` | `/v1/chat/completions` | 按 `model` 路由，代理到 worker |
| `POST` | `/v1/completions` | 同上 |
| `GET` | `/v1/models` | 汇总所有 worker 的模型条目（每个 worker 一条，用路由器的 `id`；ONNX 条目带 `backend: onnx`） |
| `POST` | `/v1/onnx/run` | ONNX 二进制协议：从请求体读出 `model`，原样转发到对应 worker（无重试，§6.3） |
| `GET` | `/v1/onnx/models` | 汇总各 ONNX worker 的模型表 |
| `GET` | `/v1/models/{id}` | 单条；未知 id → 404 |
| `GET` | `/health` | 路由器自身 + 各模型状态（`up` / `starting` / `down` / `degraded`） |
| `GET` | `/router/status` | 监督视图：重启次数、显存记账、每个模型的最近错误（运维用，不在 OpenAI 规范内） |
| `OPTIONS` | 任意 | CORS 预检，返回 204（对齐 worker 的 `cors()` 行为） |

### 6.2 代理语义

* **请求**：原样转发方法、路径、查询串、请求体；剥掉逐跳头（`Connection`、`Keep-Alive`、
  `Transfer-Encoding`、`TE`、`Upgrade`），其余（含 `Authorization`）转发。请求体上限为 512 MiB
  （ONNX 一次可以带多张大图；OpenAI 的 base64 图片远小于它）。
* **`model` 字段**：路由器是路由的权威。它按 `model` 找到 worker，然后**把请求体里的 `model` 改写成
  worker 自报的 id**（`general.name`，见 `src/server/server.cpp:615`），避免 worker 侧的
  `/v1/models/{id}` 语义与路由器的别名不一致。
* **响应**：非流式直接透传状态码与响应体。流式（`stream: true`）走 SSE 透传（§6.4）。
* **错误映射**：
  | 情况 | 状态码 | 响应体 |
  |---|---|---|
  | 未知 `model` | 404 | `{"error":{"message":"model 'x' not found","type":"invalid_request_error"}}`（与 worker 自身的形状一致，`src/server/server.cpp:679`） |
  | 缺 `model` 且只有一个模型 | — | 用那一个 |
  | 缺 `model` 且有多个模型 | 400 | `invalid_request_error` |
  | worker 不在 `up` | 503 | `{"error":{"message":"model 'x' is unavailable (state: down)","type":"server_error"}}` |
  | 上游超时 | 504 | `server_error` |

### 6.2.1 ONNX 二进制路由

`/v1/onnx/run` 的请求体是 `src/common/onnx_wire.h` 的二进制帧。路由器只读帧头的
`(version, model name)`（`peek_request_model`），不复制张量载荷；随后把整个 body 原样转给
`models[].id == model` 且 `backend == onnx` 的 worker。响应（包括 worker 的 wire error）也原样
返回。未知模型/worker 不在 `up` 时，路由器生成一个 `SIOR` 错误帧，客户端看到的仍是同一种协议。
这一条不是 OpenAI 路由，因此不做 `model` 字段改写、SSE 或重试。

### 6.3 不重试

生成请求**非幂等**：worker 可能已经产出了 token、甚至已经写了 KV。重试会造成重复生成与重复计费。
因此路由器只对**控制面**调用（`hello` / `health` / `stats`）重试，数据面一律不重试。

### 6.4 SSE 透传与取消传播

这是路由器最容易写错的一段。规则如下：

1. **不缓冲**：worker 的 SSE 块到达即转发给客户端。worker 的块被推进一个队列，路由器的
   chunked provider 逐块取走写出。
2. **不重排**：SSE 是有序的，路由器不改写帧内容。
3. **取消传播**（关键）：客户端断开 → 路由器中止上游请求 → 与 worker 的连接关闭 →
   **worker 侧的写失败** → worker 的 `sse_session` 取消 → 引擎停止生成。

**"路由器怎么知道客户端断开了"这件事，第一版想错了，测试把它纠正过来了。**
直觉的答案是"chunked provider 的 `sink.write()` 返回 false"——**那是错的**：httplib 在调用
provider **之前**先做一次 `strm.is_peer_alive()` 检查，客户端若是恰好在两次产出之间消失，
provider 根本不会被再调用一次，写失败永远不会发生。

可靠信号是 `ContentProviderResourceReleaser`：它在响应对象析构时被调用，参数是
`content_provider_success_`，即"这次响应体有没有写成功"。客户端消失 ⇒ 写失败 ⇒
`success == false`。这正是 `src/server/sse.h` 顶部注释为**worker 侧**记录的那条事实
（"httplib signals a client disconnect in exactly one place"）——多一跳之后，路由器必须用
同一个信号，只是方向反过来。

所以路由器的两个出口都要接：

* provider 里 `sink.write()` 失败 → 立刻中止上游；
* releaser 收到 `success == false` → 中止上游（这一条覆盖 provider 没被再调用的情形）。

中止上游用 `httplib::Client::stop()`：它是 httplib 唯一线程安全的中断手段
（`ClientImpl::stop()` 的实现只做 `shutdown_socket`，因为关 socket 不是线程安全的）。
`shutdown` 让 worker 侧看到 EOF，worker 的下一次写失败 → worker 取消。

4. **超时**：上游读超时按 `router.request_timeout_s` 设（默认 3600 s），不能沿用 httplib 的 5 s 默认。

`test_sse_cancel`（`tests/server/test_sse_cancel.cpp`）是这条链在 worker 侧的现成覆盖；
`tests/router/test_router_proxy.cpp` 的 `test_cancel_propagates` 是多一跳之后的同构覆盖：
它用一个裸 socket + `SO_LINGER 0`（RST）模拟客户端消失，然后断言**假 worker 观察到了断开**。

---

## 7. worker 侧改动

> **已实现（M3）**：`src/server/control.{h,cpp}`、`src/main.cpp` 的两个 flag、
> `engine::device_memory()`。

### 7.1 控制面服务

`src/server/control.{h,cpp}`：一个独立的 TCP 监听线程 + 每连接一个线程。

* 协议在 `src/common/jsonrpc.h`（**与路由器共用**，所以方法名和帧格式不会漂移）。
  它是 header-only 的：路由器不能链接 SYCL 核心，worker 也不该链接路由器。
* 报告由**回调**提供（`std::function<control_report()>`），所以控制面本身不依赖 engine，
  可以脱离模型测试（`tests/server/test_control.cpp` 就是这么做的）。
* `shutdown` 调用**同一个**终止回调，即 worker 内部的 SIGTERM 路径
  （`src/server/server.cpp` 的 `g_term_requested` → watchdog → `srv.stop()` →
  调度器停机 → 引擎析构时把前缀缓存 flush 到磁盘）。**这条路不能另写一条**：
  另写一条就会跳过前缀缓存的落盘。

### 7.2 CLI 新增

| flag | 默认 | 说明 |
|---|---|---|
| `--control-port N` | 关闭 | 启用 RPC 控制面并监听该端口（`0` = 端口随机，仅测试用） |
| `--control-host H` | `127.0.0.1` | RPC 监听地址；非环回且**没有令牌**时打印与 HTTP 同款警告 |
| `--control-token T` | 空 | 控制面的共享密钥：路由器必须在握手里出示它 |
| `--control-token-file P` | 空 | 从文件读密钥（容器里挂一个 secret 的做法）；文件内容去掉尾部换行 |

路由器派生 worker 命令行时，**自动带上 `--control-port`**（见 §4.4）——worker 必须自己
打开路由器要连的那个面。

### 7.3 报告的口径

worker 能报什么，取决于它手上有什么：

| 字段 | 来源 |
|---|---|
| `devices[].name` / `.total_vram_bytes` | `sycl::info::device::name` / `global_mem_size` |
| `devices[].free_vram_bytes` | `sycl::ext::intel::info::device::free_memory`，**best effort**：不是每个后端都实现这个 aspect，取不到就报 0（= 未知） |
| `devices[].index` | 分区序号（多卡时每个分区一项） |
| `devices[].kv_bytes` | 该设备上的 KV 池字节（多卡时按各分区负责的注意力层切分，与 `kv_setup()` 的分配一致） |
| `kv.used_bytes` / `.cap_bytes` / `.pool_blocks` / `.pool_cap` / `.used_blocks` | `engine::kv_bytes_total()` / `kv_bytes_cap()` / `pool_blocks` / `pool_cap` / `pool_used_blocks()` |
| `prefix_cache.vram_bytes` / `.ram_records` / `.disk_records` | `engine::pc_vram_bytes` / `pc_ram_records()` / `pc_disk_records()` |
| `host.rss_bytes` | `/proc/self/statm` |
| `inflight` | 调度器在跑的序列数（`scheduler::inflight()`，在同一把锁下读） |
| `uptime_s` | 进程启动至今 |
| `model.id` / `.n_ctx` / `.n_vocab` / `.n_layer` | GGUF 的 `general.name` + 引擎的 hparams |

**路由器优先用控制面拿模型的身份**（`model.id`），而不是去解析 worker 的 `/v1/models`：
一次往返就同时拿到身份、设备清单和显存，省掉一次 HTTP。

### 7.4 "ready" 的定义

**worker 的 HTTP 端口开始监听 = 模型已加载完成**：`main.cpp` 先构造 `engine`（加载 GGUF、上传权重、
定尺 KV 池、记录命令图），再调 `serve()` 监听端口（`src/server/server.cpp:1071`）。所以"端口通"就等于
"模型在显存里、可以服务"。

路由器的就绪判定**以 HTTP 为准**（`/health`），控制面随后用来补身份与显存。理由：控制面是在
`serve()` 里、HTTP 监听之前启动的，用它判定就绪会有一个"控制面已通、HTTP 还没开"的窗口，
那时候路由过去会失败。用 HTTP 判定就没有这个窗口。

控制面的存在**不是必需的**：没有 `--control-port` 的 worker 照样能被路由器管理（就绪走 HTTP，
优雅停机走信号），只是路由器拿不到它的身份与显存。

---

## 8. 生命周期与故障处理

### 8.1 状态机

```
        ┌──────────┐  spawn/fork+exec   ┌──────────┐  hello ok   ┌───────┐
   ───► │ starting │ ──────────────────► │ loading  │ ──────────► │  up   │
        └──────────┘                     └──────────┘             └───────┘
             │                                 │                      │
             │ exit≠0                          │ exit≠0               │ exit / health 连续失败
             ▼                                 ▼                      ▼
        ┌──────────┐                     ┌──────────┐          ┌────────────┐
        │  failed  │◄────────────────────│  failed  │          │ restarting │
        └──────────┘   不可重试的错误      └──────────┘          └────────────┘
                                                                    │ backoff
                                                                    └──► starting
```

### 8.2 错误分类（决定重启还是放弃）

分类的依据是**死在哪一段**，不是日志里的错误文本：

| 现象 | 分类 | 动作 |
|---|---|---|
| **起不来**（就绪之前退出：模型打不开、架构不支持、显存不够、flag 非法） | **不可重试** | 标 `failed`，不重启（重试只会再失败一次，且会刷日志） |
| 服务中崩溃（就绪之后退出，退出码非 0） | 可重试 | 指数退避重启（默认最多 3 次，间隔 2 s / 4 s / 8 s） |
| 进程活着但健康检查连续失败 | 可重试 | 先 `shutdown`，超时则 `SIGTERM`，再 `SIGKILL`；然后重启 |
| 正常退出（收到 `shutdown` 或 SIGTERM） | — | 不重启（这是有意的停机） |

**为什么按阶段而不是按错误文本**：`device alloc failed` 这类启动失败在同一个环境下每次都会
失败，重试只是刷日志；而服务中的崩溃可能是偶发的，值得重试。日志文本只用来**生成错误信息**
（"the worker exited during startup: ..."），不参与判断。

**worker 自己也会说**：启动失败（模型加载不了、显存不够、参数非法）时 worker 退出码是 **78**
（`sysexits.h` 的 `EX_CONFIG`，见 `src/common/exit_codes.h`）：

```
error: cannot open /path/to/nonexistent.gguf
(exit 78: a startup failure - do not retry, the same attempt would fail the same way)
```

路由器看到 78 就**确定**不重试，不必再依赖"它是在就绪之前死的吗"这个阶段假设；而阶段规则
仍然留着，作为没有这个退出码的 worker（被 OOM killer 杀掉之类）的兜底。78 也是给**外部**
supervisor（systemd、docker、k8s）看的：重启策略可以按退出码区分"配置错了，别重试"和
"崩了，可以重试"。

worker 的日志：stdout/stderr 重定向到 `<log_dir>/<id>.log`，`/router/status` 里有它的尾巴
（`last_error`）——**否则"模型起不来"只有一行 `device alloc failed`，没有上下文**。

### 8.3 优雅停机

路由器收到 SIGINT/SIGTERM 时：

1. 停止接受新请求（返回 503）。
2. 对每个 worker 发 RPC `shutdown`（带超时）；拿不到就退化成 SIGTERM，再超时 SIGKILL。
3. 等所有 worker 退出后退出。

worker 侧的 `shutdown` 走的是它自己的 SIGTERM 路径（§7.1），因此**前缀缓存会被 flush 到磁盘**
（`src/engine/engine_prefix_cache.cpp:283` 的 graceful-shutdown flush）。这条不能省：省了就是一次
静默的前缀缓存丢失。

**每个子进程只能有一个"主人"。** supervisor 的线程负责 waitpid 自己的子进程，路由器的 stop 路径
**不许**再 waitpid 同一个 pid。两个 waitpid 抢一个 pid 是竞态：输的那个拿到 ECHILD，把它读成
"子进程已经没了"，于是对一个正在优雅退出的 worker 发 SIGKILL——前缀缓存的 flush 就这样被静默
丢掉了。这个 bug 是实测出来的（worker 的日志停在 "server listening"，没有 "[srv] shutting down"），
回归用例在 `tests/router/test_router_control.cpp` 的 `test_graceful_shutdown`：假 worker 在收到
shutdown 后**故意多活 2 秒**（模拟 flush），测试断言它走完了（标记文件里有 "-done"）。

---

## 9. 显存预算与准入

这是本设计要解决的核心问题：**多个模型同时部署，各自的显存占用要显式、可控**。

### 9.1 三层机制

| 层 | 机制 | 谁执行 | 失败表现 |
|---|---|---|---|
| ① 静态预算 | 配置里的 `memory.vram_budget_mb`，按设备组求和 ≤ `router.device_vram_mb` | 路由器（配置校验，§4.5） | 路由器拒绝启动，配置错误 |
| ② 经验准入 | worker 加载后通过控制面回报**加载后**的 `free_vram_bytes`；路由器据此决定同组的下一个模型能否启动 | 路由器（supervisor 的账本） | 标 `failed`，附原因（"the device group 'gpu.0' has 600 MB free, and this model needs 900 MB"） |
| ③ 池内退让 | worker 自己的 `--kv-cap-mb` / `--max-slots` / `--pc-vram-mb` 决定实际峰值；KV 池按需增长、到 cap 停止 | worker | KV 池停在上限，**不是崩溃**（`src/engine/engine_kvpool.cpp:411`） |

**第 ② 层是这套设计里最值钱的部分**：它把显存准入从"预测"变成"实测"。27B 加载完才占满 9 072 MiB，
静态表算不准；让 worker 自己报，比路由器猜准得多。

**账本的口径**（实现细节，值得写清楚）：

* 账本是**按设备组**记的：组里的模型共用一组卡，所以组的空闲显存 = 该组模型上报的设备空闲之和。
* 一个模型的 `vram_budget_mb` 是它**跨所有卡的总和**，所以静态检查用 `budget / device.count`
  与单卡容量比较（`device.count` 是配置里声明的卡数）。
* 实测检查用**总预算**与组的空闲总和比较：卡上的空闲显存**已经扣掉**了已驻留的模型，
  再加一次"已承诺"就是把它们算了两遍。
* 账本必须在**放行下一个模型之前**更新。所以控制面的握手放在"就绪"之后、"释放组门"之前——
  顺序反了，下一个模型就会对着一个还没有账本的组做准入（第一版就是这个顺序，测试抓到了）。

**现状（M1/M2）**：① 和 ③ 已经生效，② 要等 M3 的控制面。在 ② 到位之前，路由器只能做**静态**
预算检查，且同卡模型的启动是**串行**的（一个就绪再起下一个），所以不会有两个模型同时抢同一张卡
的初始化显存——这已经是 ② 的一半效果。

### 9.2 启动顺序

同一设备组（`models[].device.group`）上的模型**串行启动**：起一个 → 等它就绪 → 下一个。
跨组的模型并行。

这同时解决了"两个模型同时抢同一张卡的初始化显存"的竞态。

**顺序必须是确定的，不能是线程之间的赛跑。** 实现是一个**有序队列**：组里的模型按配置的
顺序编号，门只放行当前编号的那个；一个模型的"首次启动"要等 `next == 自己的编号`，而
**重启**只要求组空闲（它已经占过号了）。第一版只做了"等到组空闲"，于是两个同组模型的
先后取决于哪个线程先抢到锁——测试立刻抓到了这个不确定性（`test_router_supervisor` 的
`test_same_group_starts_serially` 间歇性失败），修法是把"空闲"升级成"有序"。

### 9.3 明确不承诺硬隔离

**Level Zero 没有显存分区**：进程之间不隔离显存（§1.2）。因此：

* `vram_budget_mb` 是**软预算**——路由器用它做准入与告警，不是内核级的限额。
* 真正的上限由 worker 的池参数决定，超出表现为**启动期 fail-fast**
  （`dev_alloc_on` 抛 `device alloc failed`，`src/engine/engine.cpp:992`），而不是运行期神秘 OOM。
* 如果多个 worker 的预算之和超过卡的容量，**后启动的那个会分配失败**——所以 ① 层必须做，
  路由器要在启动前就把这种配置挡掉，而不是让它在部署时炸。

### 9.4 记账视图

`/router/status` 给出每张设备的账：

| 项 | 来源 |
|---|---|
| 设备总容量 | 任一 worker 的 `hello.devices[].total_vram_bytes` |
| 已承诺（预算） | Σ 该设备上各模型的 `vram_budget_mb` |
| 实测已用 | Σ 该设备上各 worker 的 `stats`（或 `hello`）里的 `free_vram_bytes` 之差 |
| KV 池上限 | 各 worker 的 `stats.kv.cap_bytes` |

"实测已用"与"已承诺"的差值就是**预算与实际的偏差**，也是运维调整 `vram_budget_mb` 的依据。

---

## 10. 远程 worker

### 10.1 写法

```jsonc
{
  "id": "remote-35b",
  "remote": { "http": "http://gpu-host-b:18081" },
  "control": "gpu-host-b:18082",
  "control_token_file": "/run/secrets/control-token"
}
```

路由器的行为差异只有一处：**不 spawn，只连接**。协议、代理、健康检查、重启策略（改成"重连"）
全都一样。这不是"预留"，是**第一天就是远程可用的**，只是 spawn 只支持本地。

注意 `control` 与 `control_token` 是**模型级**的字段（不在 `remote` 里面）：`remote` 只说"这个
worker 在别处"，控制面怎么连、令牌是什么，是模型自己的事。令牌用文件形式：远程那台机器上的
worker 用同一个文件（它是自己起的，带 `--control-token-file`）。

### 10.2 worker 侧

```
sycl-infer --model ... --host 0.0.0.0 --port 18081 \
           --control-host 0.0.0.0 --control-port 19082 \
           --control-token-file /run/secrets/control-token serve
```

### 10.3 限制

* **控制面只有共享密钥**（§5.2、§5.5）→ 令牌泄露就等于 worker 被别人控制，所以远程 worker
  的控制面端口必须靠网络隔离兜底。
* **数据面无鉴权** → 远程 worker 的 HTTP 端口同理。
* 远程 worker 的失败处理是"重连 + 退避"，不是"重启进程"。

---

## 11. Docker 部署

> **已实现（M4）**：`docker/Dockerfile.router`、`docker/Dockerfile.worker`、
> `docker/Dockerfile.all`、`docker/compose.yaml`、`docker/router.example.json`，
> 以及 CMake 的 `SI_ROUTER_ONLY` 选项。

### 11.1 三个镜像

| 文件 | 内容 | 用途 |
|---|---|---|
| `docker/Dockerfile.router` | 只有路由器，用系统编译器构建（`-DSI_ROUTER_ONLY=ON`） | 纯路由的容器 |
| `docker/Dockerfile.worker` | 只有 worker（SYCL 二进制），oneAPI basekit 构建 | 远程 worker 的容器 |
| `docker/Dockerfile.all` | 两个二进制都在一个镜像里 | **推荐拓扑**：路由器在自己进程里拉起 worker |

**`SI_ROUTER_ONLY=ON`** 是这套部署的关键：它让 CMake **完全不碰 SYCL**——不找 `icpx`、不加
`-fsycl`、不构建设备镜像、不链接 oneDNN。于是路由器的镜像：

* 构建阶段只需要一个 Debian + g++（不用拉 10 GB 的 oneAPI basekit）；
* 运行阶段只有 libc/libstdc++；
* **它自己的测试也在这个镜像里跑**（`ctest`），仍然是 hermetic 的。

路由器与 worker 之间的镜像内容是分开的，但**同一台机器上的"路由器拉起 worker"拓扑要求两者在
同一个进程命名空间里**——也就是同一个容器，所以用 `Dockerfile.all`。

### 11.2 镜像构成

**worker 镜像**：

```
FROM intel/oneapi-basekit:${ONEAPI_VERSION}-devel-${UBUNTU_TAG} AS build
# ... dnnl-devel, cmake ...
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DDNNL_ROOT=/opt/intel/oneapi/dnnl/${DNNL_VERSION} \
      -DSYCL_INFER_AOT=${AOT} -DSYCL_INFER_AOT_DEVICE=${AOT_DEVICE} \
 && cmake --build build --target sycl-infer -j$(nproc)

FROM intel/oneapi-runtime:${ONEAPI_VERSION}-${UBUNTU_TAG}
# ... dnnl-runtime, intel-level-zero-gpu, intel-opencl-icd, level-zero ...
```

两个必须与**宿主机**对齐的东西：

* **用户态运行时**（oneAPI runtime + Level Zero 计算运行时）要和宿主机的内核驱动匹配；
* **AOT 的设备镜像**是给某一个 SKU 编译的，所以 `SYCL_INFER_AOT_DEVICE` 必须是那个 SKU 的
  ocloc target（A770 是 `acm-g10`，Iris Xe 是 `adl-p`）——用 `ocloc query` 查，不要继承默认值。

### 11.3 设备透传

```yaml
devices:
  - /dev/dri:/dev/dri      # 宿主机上的所有卡
```

**指定某张卡**：用标准的 `SYCL_DEVICE_FILTER`（DPC++ 通用过滤变量），在配置的
`models[].device.filter` 里给（路由器把它传给 worker）。**具体过滤串先在目标机上用 `sycl-ls`
确认**（写法随实现而异）。

这正是"多进程"方案的一个白拿的好处：**每进程选卡零代码改动**。

### 11.4 卷

| 挂载 | 用途 |
|---|---|
| `/models` | GGUF（只读） |
| `/var/lib/sycl-infer/pc/<id>` | 前缀缓存磁盘层（`--pc-dir`），**每模型一个子目录** |
| `/var/log/sycl-infer` | worker 日志（`router.log_dir`） |

### 11.5 compose 骨架

`docker/compose.yaml` 用的是"路由器拉起 worker"的拓扑（一个容器、两个二进制）：

```yaml
services:
  router:
    build:
      context: ..
      dockerfile: docker/Dockerfile.all
      args: { ONEAPI_VERSION: "2026.2", AOT_DEVICE: "acm-g10" }
    ports: ["8080:8080"]
    devices: ["/dev/dri:/dev/dri"]
    volumes:
      - ./router.example.json:/etc/sycl-infer/router.json:ro
      - /srv/models:/models:ro
      - pc-qwen35:/var/lib/sycl-infer/pc/qwen35-0.8b
```

**远程 worker 的拓扑**（路由器在一台机器、worker 在另一台）不用 compose 的
"一个路由器 + 多个 worker" 写法：worker 是各自独立的容器（`Dockerfile.worker`），路由器用配置里的
`"remote": {"http": ..., "control": ...}` 去连。**注意那台 worker 的两个端口都必须在私网上**：
控制面没有鉴权，HTTP 也没有。

---

## 12. 实现计划与现状

### 12.1 文件（已落地）

| 文件 | 内容 |
|---|---|
| `src/router/main.cpp` | 入口：`--config` / `--check`，起 supervisor 与 HTTP 服务，信号处理 |
| `src/router/config.{h,cpp}` | 配置结构、JSON-with-comments 解析、校验、派生命令行 |
| `src/router/registry.{h,cpp}` | 模型注册表、每个模型的状态机、`model` → worker 的路由 |
| `src/router/rpc_client.{h,cpp}` | 控制面客户端（JSON-RPC over TCP） |
| `src/router/supervisor.{h,cpp}` | spawn / 监控 / 重启 / 优雅停机 / 日志收集 / 显存账本（§8、§9） |
| `src/router/proxy.{h,cpp}` | HTTP 代理 + SSE 透传 + 取消传播（§6） |
| `src/common/jsonrpc.h` | 控制面协议（**两端共用**，header-only） |
| `src/common/exit_codes.h` | worker 的退出码约定（78 = 启动失败，别重试） |
| `src/server/control.{h,cpp}` | worker 侧的控制面服务（§7） |
| `docker/` | 三个 Dockerfile、compose、示例配置（§11） |

### 12.2 CMake

* `si_router_core`（静态库）+ `sycl-infer-router`（可执行文件）。
* 两者都用 `-fno-sycl`（源文件级，与 `SI_CPU_SOURCES` 同一个手法），**不链接
  `sycl_infer_core`**——那会拖进 oneDNN 与 SYCL 运行时。
* **`SI_ROUTER_ONLY=ON`**：整个工程只构建路由器（连 `icpx` 都不找），这是容器镜像的构建路径，
  也是"路由器不带 GPU 也能构建"的保证。
* 验收：`ldd build/sycl-infer-router` 里没有 `libsycl.so`。已验证（只有 libc/libstdc++/libssl）。
* 路由器测试用 `si_add_router_test`，同样 `-fno-sycl`、同样不链接 `sycl_infer_core`。

### 12.3 测试（**全部 CPU-only，不需要 GPU 和模型**）

这是把路由器做成独立程序的最大好处——本仓库其它东西都要 GPU 才能测，路由器不用。
五个测试共 ~15 s（`SI_ROUTER_ONLY=ON` 的构建里也是这五个）。

| 测试 | 覆盖 |
|---|---|
| `tests/router/test_router_config.cpp` | JSON-with-comments 的解析（行注释、块注释、字符串里的 `//` 不是注释）、全部校验规则、派生的 worker 命令行、远程模型 |
| `tests/router/test_router_proxy.cpp` | 路由与错误形状（404 / 503 / 401）、`model` 字段改写、`/v1/models` 汇总、**SSE 逐块透传**（断言到达时间铺开而不是一批到达）、**取消传播**（裸 socket + RST → 断言 worker 观察到断开） |
| `tests/router/test_router_supervisor.cpp` | 拉起 worker → 就绪 → 杀掉 → 自动重启；**起不来的 worker 不重启**（启动失败不是崩溃）；**退出码 78 的启动失败也不重启**（worker 自己说了"别重试"）；**同组模型串行启动且顺序确定**（第一个慢起来时第二个必须还没被拉起） |
| `tests/router/test_router_control.cpp` | 控制面客户端（握手、错误、连不上）、**模型身份取自控制面**（假 worker 的 `/v1/models` 故意报另一个 id）、**经验准入**（卡上报 600 MB 空闲，第二个模型要 900 MB → 必须被拒）、**优雅停机**（假 worker 收到 shutdown 后故意多活 2 秒模拟 flush，测试断言它走完了——见 §8.3 那个双 waitpid 的 bug） |
| `tests/server/test_control.cpp` | worker 侧控制面：`hello` / `stats` / `health` / `shutdown`、错误的协议版本、未知方法、垃圾输入、终止回调、**认证**（无令牌 / 错令牌被拒、未握手就查报告被拒、对的令牌放行、没配令牌时是开放的） |

假 worker 是测试自己 re-exec 自己（`TEST_ROUTER_FAKE_WORKER=1`），与 `test_spec` 同一手法。
`tests/server/test_control.cpp` 链接的是 `sycl_infer_core`（控制面在 worker 侧），但它不碰 GPU。

**写这类测试的两条教训**（都是被自己的测试咬到的）：

* **读别的进程写的文件要读全**。优雅停机那个用例让假 worker 往一个文件里追加"我走到哪一步了"，
  而用例只读了前 63 字节——于是当 SIGTERM 那条路径多写了一行时，`-done` 落在缓冲区之外，用例
  间歇性变红（约每四次一次）。**截断的读 + 并发的写 = 一个必然 flaky 的测试**。
* **假 worker 的信号处理要和真 worker 一样**。真 worker 的 SIGTERM 处理函数只置一个原子量，
  真正的收尾在普通线程里做（`src/server/server.cpp`）。假 worker 最初在信号处理函数里直接
  `sleep_for` + 写文件——那不是 async-signal-safe 的，而且在被中断的 `sleep_for` 里再睡一次
  会死锁。**假 worker 偏离真 worker 的设计，用例测的就不是被测对象了。**

### 12.4 里程碑

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 配置 + 注册表 + supervisor（spawn/监控/重启）+ 非流式 HTTP 代理 | **已实现** |
| M2 | SSE 透传 + 取消传播 + 路由器侧测试 | **已实现** |
| M3 | worker 控制面（`src/server/control.cpp`）+ `hello`/`stats` + 显存的经验准入 | **已实现** |
| M4 | 远程 worker（只连接）+ Docker 镜像 + compose | **已实现** |

### 12.5 实现时与本设计的差异

* **就绪信号以 HTTP 为准，控制面随后补身份与显存**（§7.4）。控制面是在 `serve()` 里、
  HTTP 监听之前启动的，用它判定就绪会有一个"控制面已通、HTTP 还没开"的窗口。
* **重启的分类是按"死在哪一段"而不是日志文本**：起不来 → 不重启（配置/环境问题，重试只会
  再失败一次）；服务中崩溃 → 可重启（受 `router.max_restarts` 限制）；干净退出 → 不重启
  （那是有人主动停的）。日志尾巴只用来**生成错误信息**，不用来做判断。
  **并且 worker 自己会说**：启动失败时它退出码 78（§8.2），路由器看到 78 就确定不重试，不必
  再依赖阶段假设；阶段规则留着当兜底（被 OOM killer 杀掉之类没有这个码的）。
* **路由器的配置校验要求 GGUF 在路由器所在的主机上也存在**（不只是在 worker 的挂载里）。
  这让"挂载漏了"在启动时就被发现，代价是路由器和 worker 不能有各自独立的挂载命名空间。
* **预算的语义是"模型跨所有卡的总和"**，所以静态检查用 `budget / device.count` 与单卡容量比较
  （`device.count` 是配置里声明的卡数）；实测检查用总预算与组的空闲总和比较。
* **控制面有共享密钥**（§5.2），但没有 mTLS、没有 per-peer 身份：令牌泄露 = 任何人都能停掉那个
  worker 并读它的设备账目。远程部署仍然应当把两个端口放在私网上。
* **显存账目是模型侧自算的**（§7.3）：驱动的 free-memory 数字在这台机器上不可信（它不统计
  SYCL 分配），所以引擎自己数自己的分配（`dev_alloc_on` + oneDNN 分区的权重 store），控制面
  上报 `total - used`。代价是它**看不见别的进程**占的显存，所以静态预算（①）不能省。

---

## 13. 风险与未决

| 风险 | 影响 | 应对 |
|---|---|---|
| **代理多一跳** | 多一次网络往返与一次内存拷贝 | 生成是 ~60 ms/token 量级，代理开销可忽略；**但代理必须逐块转发、不得缓冲**（§6.4） |
| **取消传播断链** | 客户端断开后 worker 继续生成到 `max_tokens`，白烧显存与算力 | 与 `src/server/sse.h` 的既有契约逐跳对齐；M2 的测试必须断言上游请求被中止 |
| **路由器成为单点** | 路由器挂了所有模型都不可用 | 路由器无状态（配置在文件里），可以多副本 + 前置 LB；**但同一组模型只能被一个路由器 spawn**（否则两个进程抢同一份显存）→ 需要外部的"选主"或干脆用 `remote` 模式让 worker 独立存在 |
| **控制面只有共享密钥** | 令牌泄露 = 任何人都能停掉那个 worker 并读它的设备账目 | 令牌用文件形式（不出现在命令行）+ 0600；远程 worker 的端口仍应放在私网上 |
| **驱动的 free-memory 数字不可信** | 用它做准入会放行一个装不下的模型（或一直拒绝） | 不用它：模型侧自算（§7.3、§12.5），实测数字与 AGENTS.md 的手算表一致 |
| **跨进程并发 GPU 作业** | AGENTS.md 记录 bench-host 上"stacking two GPU jobs makes a card drop out with `UR_RESULT_ERROR_DEVICE_LOST`" | **部署前必须在目标机复现**：若复现且原因是显存互相挤占 → 用 §9 的串行启动 + 预算；若是驱动问题 → 考虑单进程多模型（§2.3 的代价清单），或错峰启动 |
| **worker 加载很慢** | 27B ~80 s，路由器要等 | `startup_timeout_s` 调大；`/health` 区分 `starting` 与 `up`；`/router/status` 显示加载进度（来自 worker 日志） |
| **磁盘前缀缓存目录冲突** | 多个 worker 写同一目录 | 每模型一个子目录（`memory.prefix_cache_dir`），且 `--pc-dir` 的记录本来就按模型指纹分子目录 |
| **数据面搬上 RPC** | 若将来要做，路由器要重做 OpenAI 整形 | 帧格式预留 `generate` 事件，但**不要在没有第二个硬需求时做**（§3.2） |

---

## 14. 与其它文档的关系

* [09-server.md](09-server.md)：worker 侧的 OpenAI HTTP 与 SSE 契约，路由器是它的**代理**而不是替代。
* [05-kv-cache.md](05-kv-cache.md)：KV 池的定尺与按需增长，是 §9 第 ③ 层退让机制的依据。
* [06-prefix-cache.md](06-prefix-cache.md)：三层前缀缓存与它的预算参数（`--pc-vram-mb` / `--pc-dir`）。
* [12-build-and-testing.md](12-build-and-testing.md)：新增 target、测试目标与验证流程。
* [AGENTS.md](../../AGENTS.md)：环境变量总表与全局不变量。
