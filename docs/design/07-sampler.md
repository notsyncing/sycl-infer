# 设计 07：采样器

覆盖 `src/engine/sampler.{h,cpp}`。

---

## 1. 数据结构

```cpp
struct gen_params {                 // sampler.h:8-20
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
};

struct sampler_state {              // sampler.h:22-27
    std::mt19937_64 rng;
    void seed(uint64_t s) { rng.seed(s); }
};
```

`gen_params` 由 CLI flag、HTTP 请求体或调度器默认值填充；`sampler_state` 每个序列一个。

---

## 2. 算法（`sample_token`，`sampler.cpp:9-107`）

输入是最后一步的 logits、`n_vocab`、`gen_params`、最近 token 历史 `recent` 与 `sampler_state`。

```
1. 惩罚 pass（仅当任一惩罚非默认且 recent 非空）
   统计最近 repeat_last_n 个 token 的出现次数；对每个不同 token：
     - repeat_penalty != 1: 正 logit 除以它，负 logit 乘以它
     - 加 presence_penalty + frequency_penalty * count
   在 logits 的副本 scr 上操作，不修改引擎缓冲。

2. 贪心短路：temperature <= 0 或 top_k == 1 → argmax。

3. 温度 + 候选集：scr *= 1/temperature；mx = max；
   保留 [mx-20, mx] 内的候选（20 nats 硬阈值）；降序排序。
   （thread_local 复用 vector，避免重复分配。）

4. softmax 候选；maxp = 最大概率；minp_thr = maxp * min_p。

5. 截断：上限 min(size, top_k)；累加概率，
   遇到 min_p（i>0）或累计 >= top_p 停止；至少保留一个。

6. 采样：在保留集上重新归一化，均匀取 [0, ksum)，返回累计和首次达到抽样的 token；
   兜底返回最后一个保留候选。
```

`recent` 是 prompt+生成历史，因此惩罚能看到完整上下文（最多 `repeat_last_n`）。

---

## 3. 边界与外部职责

* **无 grammar / FSM**、无 logit-bias、无 mirostat/DRY。
* **无 stop-string 支持**：stop 字符串是服务器侧的后处理过滤（见
  [09-server.md](09-server.md)），不在采样器里。
* **EOS 判定在引擎/调度器**：`engine::is_eos(tok)` = `tok == tk.eos_id || tok == tk.eot_id`；
  `ignore_eos` 控制是否忽略。采样器本身不处理 EOS。
* 显式 `seed` 提供可复现性；seed 为 0 时调度器/引擎改用时钟或 `random_device` 播种。
