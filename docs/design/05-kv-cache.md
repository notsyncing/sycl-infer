# 设计 05：分页 KV Cache 与动态块池

覆盖 `src/engine/engine_kvpool.cpp` 与 `engine.h` 中的 KV 字段、`src/backend/gpu/kernels/kv_type.{h,cpp}` 以及
`qk_norm_rope` / `attn` 的 KV 访问布局。前缀缓存见 [06-prefix-cache.md](06-prefix-cache.md)。

---

## 1. 为什么是分页 KV + 动态池

* **分页**：连续批处理需要为不同长度、不同生命周期的序列腾挪 KV。固定 `kBlockSize = 32` token 的物理
  块 + 每序列块表（block table）让分配/回收 O(1)，也天然支持前缀缓存的“32 token 粒度”复用。
* **动态池**：`--ctx` 可达 20480+，如果启动就提交全部 KV，会浪费大量显存（集显与系统内存共享）。
  解决办法是 **虚拟 USM 预留地址范围 + 物理 extent 按需提交**。地址范围只预留一次，使录制进图的
  每层基址稳定；物理内存随实际并发增长/收缩。

---

## 2. 字节几何

* `kv_block_bytes()`（`engine_kvpool.cpp:30-36`）：一个 attention 层中一个块 **K 的字节数**
  `n_head_kv * kBlockSize * kv_dtype_row_bytes(kv_k_dtype(), head_dim)`；V 用
  `kv_v_block_bytes()`（`--kv-type K:V` 时两者不同，否则相等）。
* `kv_layer_stride` / `kv_v_layer_stride`：每 attention 层 K / V 的字节数 =
  `n_blocks * kv_block_bytes()` / `n_blocks * kv_v_block_bytes()`。三处设置：
  `kv_setup` 的两条分支与 `alloc_buffers`（`engine.cpp:390`）。这里的 `n_blocks` 是**预留**大小，这
  正是池增长时图基址仍然有效的原因。
* `kv_scale_stride` / `kv_v_scale_stride`（`engine_kvpool.cpp`）：仅 i8/i4，`n_blocks *
  n_head_kv * kBlockSize * (head_dim/kI8Q) * sizeof(half)`（K/V 同型时相等）。scale 平面是普通
  `malloc_device`（不是虚拟内存），按完整预留分配。
* `attn_layers()` 统计非递归（full attention）层；GDN 层不占 KV。

`kv_bytes_total()` / `kv_bytes_cap()` 报告已提交/预留的 K+V 总字节。

---

## 3. 虚拟 USM 预留（`kv_setup`，`engine_kvpool.cpp:81-169`）

由 `alloc_buffers` 在 `n_blocks = pool_cap` 之后调用 `kv_setup(n_attn, pool_initial)`。
`kv_setup` 开头有两个分支，都会**跳过虚拟 USM**：

* **CPU（`cpu_mode`）**：`kv_virtual = false`，把整个 `pool_cap` 一次性用 `sycl::malloc_host`
  提交（K/V 与 int8 scale 平面都是主机 USM），`kv_grow` 只做空闲链表记账。没有虚拟内存与
  map/shrink，因此 CPU 上 `kv_read_vec`/前缀缓存的磁盘序列化都是主机拷贝。
* **多设备（`multi_dev`）**：`kv_layer_stride = pool_cap * block_bytes` 全局一致，但每个设备只有
  一份包含**该设备注意力层**的池（`dev_kpool_[d]`/`dev_vpool_[d]` + scale 平面），由
  `layer_attn_local_[il]` 索引；block id 全局一致，所以 block table 只需一份。`kv_release_pool`
  释放这些按设备分配的池。

下面的流程针对单设备 GPU（虚拟 USM）。

1. **2 MB 颗粒度推导**：`mb2 = 2<<20`，`align = ceil(mb2 / block_bytes)`，`kv_align_blocks = align`。
   原因（注释 `:84-87`）：Level Zero 要求 ≥ 2 MB 的映射必须 2 MB 对齐，所以层 stride 与 extent 起点都
   保持在 2 MB 边界。fp32 KV 一块 64 KB → 每 2 MB 32 块；2 字节元素类型减半 → 每 2 MB 64 块。
2. `pool_cap` 向上取整到 `align` 的倍数；`initial_blocks` clamp 到 `pool_cap` 并对齐。
3. **预留**：用 `sx::reserve_virtual_mem` 分别预留 K（`kv_vbase`）和 V（`kv_vbase_v`）两个范围，
   大小 `round_up(pool_cap * block_bytes * n_attn, gran)`；`kv_reserve_bytes` 记录大小。
4. **对齐探测**：`kv_map2 = ok && kv_vbase % mb2 == 0 && kv_vbase_v % mb2 == 0 &&
   (pool_cap*block_bytes) % mb2 == 0`。虚拟 USM 不可用时捕获异常并**回退**。
5. **次 2 MB 回退**：虚拟但不对齐时，增长限制在 < 2 MB 的 extent（此时只要求 64 KB 对齐）；`initial_blocks`
   上限 16 并告警。
6. **固定回退**（`!kv_virtual`）：`pool_cap = initial_blocks`，`n_blocks = initial_blocks`，用
   `malloc_device` 分配 `initial_blocks * block_bytes * n_attn`；**不能增长或收缩**。
7. 虚拟情况下 `n_blocks = pool_cap`，`kv_layer_stride` 基于预留。
8. **int8 scale 平面**一次性按 `n_blocks` 分配，不参与虚拟增长。
9. 初始化 `block_used_`（0）、`block_extent_`（-1）、`pc_block_node_`（-1）；`pool_blocks = 0`。
10. `d_kpool = (void*)kv_vbase`、`d_vpool = (void*)kv_vbase_v`；`kv_initializing = true` 下提交初始 extent。

---

## 4. 物理 extent（`kv_grow` / `kv_shrink` / `kv_release_pool`）

### 4.1 增长（`engine_kvpool.cpp:171-229`）

* 已达 cap 或 `want < 1` 直接返回。
* `kv_map2` 时 `want` 向上取整到 `kv_align_blocks`；否则上限 16（< 2 MB 规则，固定回退同样适用）。
* `count = min(want, pool_cap - pool_blocks)`，`first = pool_blocks`（连续、最低优先）。
* `map_bytes` 取 device granularity 向上取整。
* **每层物理映射**：对每个 attention 层 `il`，`off = il*layer_bytes + first*block_bytes`，为
  `(layer, K/V)` 各构造一个 `sx::physical_mem` 并 `map()` 到 `kv_vbase+off` / `kv_vbase_v+off`
  （`read_write`）。注释（`engine.h:401-403`）：一个 `physical_mem` 对象只能映射一次，因此每层每 K/V
  一个，存进 `kv_extent::phys_k/phys_v`。
* 映射异常时该 extent 保持未映射、打印错误并返回 false。
* extent 的块推进 `free_blocks_`，设 `block_extent_[b]`；追加 extent、`extent_used_.push_back(0)`、
  `pool_blocks += count`、`pool_grows++`。
* 非初始化时 `pool_print("grow")`。

### 4.2 收缩（`engine_kvpool.cpp:231-270`）

* 无虚拟 USM 时空操作。
* 循环条件：顶部 extent 存在、`extent_used_.back() == 0`、且 `first >= pool_initial`。`pool_initial` 守卫
  保证历史固定池不抖动，只释放突发增长出来的部分。
* 从最小堆中移除该 extent 的块（排空重建），逐层 `unmap`，清 `phys_k/phys_v`，把 `block_extent_` 置 -1，
  `pool_blocks -= count`，弹出，`pool_shrinks++`，`pool_print("shrink")`。

### 4.3 拆除（`engine_kvpool.cpp:272-309`）

* 虚拟：unmap 每个 extent 的每层 K/V，`sx::free_virtual_mem` 两个基址，清零 `kv_vbase*`。
* 固定：`sycl::free` 池与 scale 平面。由 `~engine` 在 `pool_print("exit")` 后调用。

---

## 5. 块分配器与块表

* **空闲表**是 `std::priority_queue<int, vector<int>, greater<int>>` 最小堆（`engine.h:383`），因此分配
  总是取最小块 id。这保证高地址 extent 容易完全空闲，从而可被收缩。
* `alloc_block()`（`engine_kvpool.cpp:322-349`）：
  1. 弹出最小空闲块（防御性跳过已用块）；
  2. 置 `block_used_[b]=1`、`extent_used_[block_extent_[b]]++`、更新 `pool_peak`；
  3. 若空闲表为空：前缀缓存开启时先 `pc_evict_lru()`，再在 cap 内 `kv_grow(pool_chunk)`，否则返回 -1。
  * `pool_chunk` 默认 64 块（`engine.h:152`），`PF_KV_GROW` 可覆盖。
* `free_block(b)`：边界与重复释放守卫；清 `block_used_`、减 `extent_used_`、推回最小堆；虚拟模式下若该块
  属于顶部 extent，调 `kv_shrink()`。
* `set_table(slot, blocks)`：把最多 `max_blocks` 个块 id 写入 `h_tables` 第 `slot` 行，`memcpy` 整行到
  `d_tables`（`[kMaxB][max_blocks]`）。`max_blocks = ceil(max_seq/kBlockSize)`。
* 记账：`pool_free_blocks() = free_blocks_.size()`，`pool_used_blocks() = pool_blocks - free`。
  每块元数据 `block_used_`、`block_extent_`、`extent_used_`。
* `pool_print(tag)` 输出已提交/上限块数与 MB、in-use、peak、grow/shrink 次数、`(fixed)` 标记。

---

## 6. 不变量

* 地址范围只预留一次；只有 `physical_mem` 的提交/解映射改变物理占用。图因此持有稳定的基址。
* extent 在块 id 上连续（`first = pool_blocks`）；`kv_map2` 时层 stride 与 extent 起点在 2 MB 边界，
  满足 Level Zero 对齐要求。
* 最低优先分配 + 只从完全空闲的顶 extent（`>= pool_initial`）收缩，提供近似 LIFO 的回收且无碎片。
* int8 scale 平面独立、固定大小、**不在**虚拟范围里。
* `PF_KV_CAP_MB` 同时是 KV 池的块上限来源，也是三层前缀缓存预算之和的上限，见
  [06-prefix-cache.md](06-prefix-cache.md)。

---

## 7. KV 存储类型

由 `kv_type.h` / `kv_type.cpp` 定义与选择，详见 [03-kernels.md](03-kernels.md#kv-存储类型kv_typeh）。
回顾：

* `i8`（默认）：对称 int8 `[-127,127]` + 每 32 head dim 一个 fp16 scale；数据平面
  `[block][kv head][token][head_dim]`，scale 平面 `[block][kv head][token][head_dim/32]`。
  3 KB/token/layer（K+V）。
* `i4`：每字节打包两个对称 4-bit 值 `[-7,7]`（低 nibble = 偶数 dim，二补码）+ 同样的每 32 head dim
  fp16 scale；数据平面 `[block][kv head][token][head_dim/2]`，scale 平面与 i8 相同。
  1.5 KB/token/layer，是 i8 的一半。
* `bf16`：2 字节元素；`f16`：同尺寸、指数更小；`f32`：历史路径，12 KB/token/layer。

int8/int4 的量化与反量化在 kernel 内完成（`qk_norm_rope` 写、`attn` 读），中间算术保持 fp32。
KV 类型通过 `--kv-type`（CLI）或 `PF_KV_TYPE`（env）选择，见 [03-kernels.md](03-kernels.md#11-kv-存储类型kv_typeh)。
`--kv-type K:V` 可分别指定 K 与 V（仅 i4/i8 可混）。i4 的精度/性能测量见
[`reports/int4_kv.md`](../../reports/int4_kv.md)：端到端与 i8 持平，收益是容量。误差由 V 主导，
所以 `i4:i8`（K 用 i4、V 用 i8）用 i8 的 75% 字节拿到接近 i8 的精度，并装得下 262144 上下文；
见 [`reports/turboquant_and_perf.md`](../../reports/turboquant_and_perf.md)。
