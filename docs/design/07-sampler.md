# 设计 07：采样器

覆盖 `src/engine/sampler.{h,cpp}`。

---

## 1. 数据结构

```cpp
struct gen_params {                 // sampler.h
    int    max_tokens = 256;
    float  temperature = 1.0f;
    float  top_p = 0.95f;
    int    top_k = 40;
    float  min_p = 0.0f;
    float  repeat_penalty = 1.0f;
    int    repeat_last_n = 64;
    float  presence_penalty = 0.0f;
    float  frequency_penalty = 0.0f;
    uint64_t seed = 0;
    bool   ignore_eos = false;
    std::unordered_map<int, float> logit_bias; // OpenAI logit_bias
    bool   logprobs = false;                   // report token logprobs
    int    top_logprobs = 0;                   // alternatives (0..20)
    bool   need_score = false;                 // internal best_of scoring
};

struct sample_logprobs {            // filled only when requested
    float logprob = 0.f;                    // sampled token
    std::vector<std::pair<int, float>> top; // highest-first alternatives
};

struct sampler_state {              // sampler.h
    std::mt19937_64 rng;
    void seed(uint64_t s) { rng.seed(s); }
};
```

`gen_params` 由 CLI flag、HTTP 请求体或调度器默认值填充；`sampler_state` 每个序列一个。
`logit_bias`/`logprobs`/`need_score` 全部默认为空/false，`sample_token` 据此完全跳过对应的
额外计算，默认路径与之前逐字节相同。

---

## 2. 算法（`sample_token`，`sampler.cpp`）

输入是最后一步的 logits、`n_vocab`、`gen_params`、最近 token 历史 `recent` 与 `sampler_state`；
可选输出 `sample_logprobs * out`（默认 nullptr）。

```
1. 惩罚/偏置 pass（仅当任一惩罚非默认且 recent 非空，或 logit_bias 非空）
   统计最近 repeat_last_n 个 token 的出现次数；对每个不同 token：
     - repeat_penalty != 1: 正 logit 除以它，负 logit 乘以它
     - 加 presence_penalty + frequency_penalty * count
   再对每个 logit_bias 条目：scr[token_id] += bias（请求侧已 clamp 到 [-100,100]）。
   在 logits 的副本 scr 上操作，不修改引擎缓冲。

2. 贪心短路：temperature <= 0 或 top_k == 1 → argmax。
   若 out != nullptr，用 log-softmax 填 chosen token 的 logprob（温度 >0 时按 1/T 缩放）。

3. 温度 + 候选集：scr *= 1/temperature；mx = max；
   保留 [mx-20, mx] 内的候选（20 nats 硬阈值）；降序排序。
   （thread_local 复用 vector，避免重复分配。）

4. softmax 候选；maxp = 最大概率；minp_thr = maxp * min_p。

5. 截断：上限 min(size, top_k)；累加概率，
   遇到 min_p（i>0）或累计 >= top_p 停止；至少保留一个。

6. 采样：在保留集上重新归一化，均匀取 [0, ksum)，返回累计和首次达到抽样的 token；
   兜底返回最后一个保留候选。

7. 若 out != nullptr：对（缩放后的）完整词表做 log-softmax，写 chosen logprob；
   若 top_logprobs > 0，再 partial_sort 出最高若干并写 out->top。此步只在请求 logprobs
   或 best_of 打分时执行。
```

`recent` 是 prompt+生成历史，因此惩罚能看到完整上下文（最多 `repeat_last_n`）。

logprobs 定义在**惩罚/偏置、温度缩放后的完整词表 softmax** 上，不受 top_k/top_p/min_p 截断影响；
`top` 至少包含 `top_logprobs` 个候选项（chosen 在其中时数值与 `logprob` 相等）。

---

## 3. 边界与外部职责

* **无 grammar / FSM**、无 mirostat/DRY。
* **已支持 logit_bias 与 token logprobs**（服务器侧解析请求，见
  [09-server.md](09-server.md) §8）；best_of 用 chosen logprob 求和打分。
* **无 stop-string 支持**：stop 字符串是服务器侧的后处理过滤（见
  [09-server.md](09-server.md)），不在采样器里。
* **EOS 判定在引擎/调度器**：`engine::is_eos(tok)` = `tok == tk.eos_id || tok == tk.eot_id`；
  `ignore_eos` 控制是否忽略。采样器本身不处理 EOS。
* 显式 `seed` 提供可复现性；seed 为 0 时调度器/引擎改用时钟或 `random_device` 播种。
