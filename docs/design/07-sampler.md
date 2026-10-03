# 设计 07：采样器

覆盖 `src/engine/sampler.{h,cpp}`。请求字段的解析、logprobs 的上报与 stop 判定都在服务器侧，见
[09-server.md](09-server.md) §2.4 与 §8。

---

## 1. 数据结构

```cpp
struct gen_params {                        // sampler.h:10-36
    int    max_tokens = 256;
    float  temperature = 1.0f;
    float  top_p = 0.95f;
    int    top_k = 40;
    float  min_p = 0.0f;
    float  repeat_penalty = 1.0f;
    int    repeat_last_n = 64;
    float  presence_penalty = 0.0f;
    float  frequency_penalty = 0.0f;
    uint64_t seed = 0;                     // 0 = 不指定（见 §5）
    bool   ignore_eos = false;
    std::unordered_map<int, float> logit_bias; // OpenAI: token id -> 加性偏置
    bool   logprobs = false;               // 报告被采样 token 的 logprob
    int    top_logprobs = 0;               // 备选项个数（服务器夹到 0..20）
    bool   need_score = false;             // 内部：best_of 打分也要 chosen logprob

    bool wants_logprobs() const { return logprobs || need_score; }   // :33-35
};

struct sample_logprobs {                   // sampler.h:40-43，filled only when requested
    float logprob = 0.f;                            // 被采样 token 的 logprob
    std::vector<std::pair<int, float>> top;         // 最高在前的备选项
};

struct sampler_state {                     // sampler.h:45-50
    std::mt19937_64 rng;
    void seed(uint64_t s) { rng.seed(s); }
};
```

填充者只有两处：

| 来源 | 覆盖的字段 |
|---|---|
| `parse_params`（`src/server/server.cpp:100-158`） | 全部 |
| CLI `gen`（`src/main.cpp:396-401`） | 只有 `max_tokens`、`temperature`、`top_p`、`top_k` |

CLI 的 `--temp` 默认 **0.7**（`main.cpp:148`），不是结构体默认的 1.0；CLI 也没有 `--seed`、`--min-p`
和三个 penalty 的 flag，所以 CLI 路径上 `do_pen` 恒为 false、`seed` 恒为 0（走时钟或
`random_device`，见 §5）。

`logit_bias`/`logprobs`/`need_score` 默认为空/false 时，`out` 是 `nullptr`、惩罚 pass 不进：默认路径
既不复制 logits 也不跑 logprobs pass。

`sampler_state` 每个序列一个（`sequence::ss`，`src/server/scheduler.h:24`），在 `scheduler::submit`
里播种（`scheduler.cpp:55`）。

---

## 2. 算法（`sample_token`，`src/engine/sampler.cpp:54-181`）

输入是最后一步的 logits、`n_vocab`、`gen_params`、调用方给的 token 历史 `recent` 与 `sampler_state`；
可选输出 `sample_logprobs * out`（默认 nullptr）。

```
0. inv_t = temperature > 0 ? 1/temperature : 1                                    (:88)

1. 惩罚 + 偏置：在 logits 的副本 scr 上做，引擎缓冲不动                          (:56-86)
   仅当「任一 penalty != 默认 且 recent 非空」或「logit_bias 非空」才进入：
   - 统计 recent 尾部 repeat_last_n 个 token 的出现次数                         (:65-69)
   - 对每个出现过的 id t（unordered_map 的遍历顺序）：
       repeat_penalty != 1:  v > 0 ? v / repeat_penalty : v * repeat_penalty    (:72-74)
       v += presence_penalty + frequency_penalty * count                        (:75)
   - 每个 logit_bias 条目 scr[id] += bias，id 必须落在 [0, n_vocab)              (:78-84)
   ★ logit_bias 就加在这里：在温度缩放之前，也在 20-nat 候选过滤与
     top_k / top_p / min_p 之前 —— 所以它一定影响采样结果，贪心（T<=0）时同样生效。

2. 贪心短路：temperature <= 0 或 top_k == 1 → 在（已偏置的）logits 上取 argmax     (:90-101)
   out != nullptr 时同样调 fill_logprobs。

3. 候选集：mx = max(lg*inv_t)，保留 [mx-20, mx] 的项（20 nats 硬阈值，相对概率
   ≈ e^-20 ≈ 2.1e-9），按值降序排序；cand 是 thread_local 复用                 (:103-120)
   cand 为空（所有 logit 非有限）→ 回退 argmax                                  (:121-130)

4. 候选 softmax：cand[i] ← exp(v-mx)，sum 为其和；maxp = cand[0]/sum；
   minp_thr = maxp * min_p。分母是全部候选的和，不是截断后的                    (:132-139)

5. 截断（降序，最多遍历 nmax 项）：nmax = min(size, top_k)，
   top_k <= 0 表示不设上限                                                    (:141-144)
     i > 0 且 min_p > 0 且 p < minp_thr → break（所以 top-1 永远保留）           (:149-151)
     cum += p; keep = i+1
     cum >= top_p → break   ← nucleus 的实际规则是「最小的、累计和 >= top_p 的前缀」 (:154-156)
   keep <= 0 时强制为 1                                                         (:158-160)

6. 抽样：只重算前 keep 个的权重和 ksum（无需显式归一化，相对权重就够），
   r ~ uniform_real_distribution<double>(0, ksum)，走前缀和取第一个 acc >= r 的候选；
   初值 cand[keep-1] 就是兜底答案                                             (:161-176)

7. out != nullptr → fill_logprobs(lg, n_vocab, inv_t, chosen, top_logprobs, out) (:177-179)
```

`recent` 由调用方给，两条路径语义不同：

* **服务器（调度器）**：`sequence::recent` 在 prefill 结束后初始化为**整个 prompt**
  （`scheduler.cpp:269`），之后每个生成 token 追加（`:275`、`:381`）—— 惩罚窗口能看到完整上下文，
  最多回看 `repeat_last_n`（默认 64）个 token。
* **单序列引擎路径**：`engine::generate_impl`（`engine.cpp:2455`）与 `generate_mtp`
  （`engine_mtp.cpp:877`、`:1435`）传的是 `out`，只含**已生成**的 token，不含 prompt。CLI 又不设置
  penalty，所以这条路上 `do_pen` 实际恒为 false。

---

## 3. 边界与外部职责

* **无 grammar / FSM**，无 mirostat / DRY。
* **已支持 `logit_bias` 与 token logprobs**：请求侧解析见 [09-server.md](09-server.md) §2.4 与 §8，
  语义见本篇 §2/§4；`best_of` 用 chosen logprob 求和打分（§4）。
* **无 stop-string 支持**：stop 字符串是服务器侧的后处理过滤（见 [09-server.md](09-server.md) §5
  停止字符串），不在采样器里；`stop_token_ids` 也没有实现（`parse_stop` 只收字符串或字符串数组，
  `server.cpp:197-213`）。
* **EOS 判定在引擎/调度器**：`engine::is_eos(tok)` = `tok == tk.eos_id || tok == tk.eot_id`
  （`src/engine/engine.h:617-619`），调度器里是同一条条件的内联版（`scheduler.cpp:293`、`:385`）；
  `ignore_eos` 控制是否忽略。采样器本身不处理 EOS。
* 采样器**不读任何环境变量**，因此采样行为没有 `PF_*` A/B 开关。

---

## 4. logprobs 报告（`fill_logprobs`，`sampler.cpp:17-50`）

* 定义域是**惩罚 + `logit_bias` 之后、按 `inv_t` 缩放后的完整词表**，与 top_k / top_p / min_p 的截断
  无关：第 7 步传进去的是 `lg`，不是候选数组 `cand`（`sampler.cpp:177-179`）。
* 两遍扫描：第一遍取 `mx`（`:18-24`），第二遍在 `double` 下累加 `exp(lg[i]*inv_t - mx)`（`:26-28`），
  `log_z = log(sum) + mx`（`:29`），`out->logprob = lg[chosen]*inv_t - log_z`（`:30`）。
  `temperature <= 0` 时 `inv_t = 1`，所以贪心路径报告的是 **T=1** 的 log-softmax。
* `top`：`top_logprobs <= 0` 直接返回空（`:32-34`）；否则 `k = min(top_logprobs, n_vocab)`（`:35`），
  对整词表建 `pair<float,int>` 再 `partial_sort` 前 k 个（`:39-45`），按 logprob **降序**。比较器只看
  `.first`，而 `std::partial_sort` 不稳定，所以同分项之间的顺序不保证。`tv` 是 `thread_local` 复用，
  但每个要报告的 token 仍要付两遍词表扫描加一次 partial sort。
* 开关：调度器只在 `gp.wants_logprobs()` 时传 `&lp`，否则传 `nullptr`（`scheduler.cpp:271-274`、
  `374-377`）；单序列引擎路径**从不**传 `out`（`engine.cpp:2455`、`engine_mtp.cpp:877`），所以 CLI
  没有 logprobs。
* 上限：`top_logprobs` 被服务器夹到 `[0, 20]`（`server.cpp:151`、`:155`）。chat 的 `logprobs` 是 bool
  配 `top_logprobs`；completions 的 `logprobs` 是整数，同时决定 `logprobs` 与 `top_logprobs`
  （`server.cpp:143-156`）。
* `need_score`：`best_of > n` 时由服务器打开（`server.cpp:1850`、`:1856-1858`），它只要求 `logprob`
  —— 每个生成 token 的 chosen logprob 累加进 `choice_out::score`（`server.cpp:794-797`），再按分数降序
  取前 `n`（`server.cpp:1863-1867`）。它不会顺手置位 `top_logprobs`，所以纯打分时 `top` 仍为空
  （`tests/engine/test_sampler.cpp:71-82` 就断言这一点）。

---

## 5. 随机数与可复现性

* 发生器是 `std::mt19937_64`（`sampler.h:46`），每序列一个 `sampler_state`；抽样只用一个
  `std::uniform_real_distribution<double>(0.0, ksum)`（`sampler.cpp:166`）。
* 播种三处，`seed == 0` 时各用不同的兜底：

| 位置 | `seed == 0` 时的兜底 |
|---|---|
| `scheduler::submit`（`scheduler.cpp:55`） | `steady_clock::now().time_since_epoch().count()` |
| `engine::generate_impl`（`engine.cpp:2371`） | `std::random_device{}()` |
| `engine::generate_mtp`（`engine_mtp.cpp:723`） | `std::random_device{}()` |

* 显式 `seed != 0` 时三处都用请求的 seed，且第 `c` 个 choice 拿 `seed + c`（`gp_for_choice`，
  `server.cpp:738-744`）。因为每序列一条独立的流，同 seed + 同请求的**随机数序列**与批处理无关；
  但同 seed 下并发请求的最终输出仍可能不同 —— 那不是采样器，是批量 GEMM 的累加顺序（见
  [09-server.md](09-server.md)）。
* **贪心短路不消耗随机数**（`sampler.cpp:90-101`），所以 `temperature <= 0` 或 `top_k == 1` 的输出与
  seed 无关。
* **MTP 路径没有随机采样**：它只在贪时下进入（`engine.cpp:2365`），verify 里的 bonus token 或走
  `sample_token`、或直接用设备 argmax（`engine_mtp.cpp:1354-1357`、`:1435`），对贪心二者等价。

---

## 6. 测试

`tests/engine/test_sampler.cpp`（CPU only，不需要模型），失败返回非 0：

| 断言 | 内容 |
|---|---|
| `logit_bias forces token` / `bans token` | `temperature=0` 下 `logit_bias[2]=100` 采样出 2、`logit_bias[0]=-100` 采样出 1 —— 证明偏置加在贪心 argmax **之前** |
| `chosen logprob uniform` / `top size honored` / `top logprob uniform` | 4 个全 0 logit：chosen logprob = `-log 4`，`top.size() == top_logprobs`，首项数值相同 |
| `top logprobs normalized` | 5 个不等 logit + `top_logprobs=5` → `Σ exp(top[i]) == 1` |
| `need_score computes chosen logprob` / `skips top alternatives` | `temperature=0` + `need_score`：填 `logprob`，`top` 保持空 |