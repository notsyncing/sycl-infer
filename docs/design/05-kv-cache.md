# 设计 05：分页 KV Cache 与动态块池

覆盖 `src/engine/engine_kvpool.cpp`、`engine.h` 中的 KV 字段与块池、`src/backend/gpu/kernels/kv_type.{h,cpp}`
以及 `qk_norm_rope` / `attn` 的 KV 访问布局。定尺（`--blocks` / `--ctx` / `--kv-cap-mb`）在
`src/engine/engine.cpp` 与 `src/main.cpp`；前缀缓存见 [06-prefix-cache.md](06-prefix-cache.md)。

---

## 1. 为什么是分页 KV + 动态池

* **分页**：连续批处理需要为不同长度、不同生命周期的序列腾挪 KV。固定 `kBlockSize = 32` token 的物理
  块 + 每序列块表（block table）让分配/回收 O(1)，也天然支持前缀缓存的“32 token 粒度”复用。
* **动态池**：`--ctx` 默认就是 20480（`main.cpp:42` 的 `kDefaultCtx`），`--ctx full` 可以直接取 GGUF 的
  context_length（实测 27B 用 `--kv-type i4:i8` 把 262144 token 的 context 装进 2x A770）。如果启动就
  提交全部 KV，会浪费大量显存（集显与系统内存共享）。解决办法是 **虚拟 USM 预留地址范围 + 物理 extent
  按需提交**：地址范围只预留一次，使录制进图的每层基址稳定；物理内存随实际并发增长/收缩。

---

## 2. 字节几何

### 2.1 K / V 两侧的块与层

* `kv_block_bytes()`（`engine_kvpool.cpp:29-31`）：一个 attention 层中一个块 **K 的字节数**
  `n_head_kv * kBlockSize * kv_dtype_row_bytes(kv_k_dtype(), head_dim)`；V 用 `kv_v_block_bytes()`
  （`engine_kvpool.cpp:33-35`）。只有 `--kv-type K:V` 混型时两者不同，否则相等。
* `kv_layer_stride` / `kv_v_layer_stride`（`engine.h:195-196`）：每 attention 层 K / V 的字节数 =
  `n_blocks * kv_block_bytes()` / `n_blocks * kv_v_block_bytes()`。写入点不止一处：`alloc_buffers`
  先按构造期算好的 `n_blocks`（= `pool_cap`）写一次 K stride（`engine.cpp` 的 `kv_layer_stride = n_blocks * kv_block_bytes()`），紧接着调用
  `kv_setup`（`engine.cpp:1718`），而 `kv_setup` 的每个分支都会**重写两侧**：多设备分支
  `engine_kvpool.cpp:193-194`、CPU 分支 `:246-247`、虚拟/固定回退 `:324-329`。这里的 `n_blocks` 是
  **预留**大小，这正是池增长时图基址仍然有效的原因。
* `attn_layers()`（`engine_kvpool.cpp:102-115`）：统计非递归（full attention）层；GDN 层不占 KV。
  `mtp_on` 时**再加一**——MTP draft 层是一个 full-attention qwen35 块，它拥有自己那份 paged KV，
  索引为 `attn_layers() - 1`（存最后）。把它算进 `attn_layers()` 让池定尺、前缀缓存 blob 及其
  序列化循环都自动覆盖它，无需额外插桩。
* MTP 的 KV 落在 `--mtp-device` 那张卡的池上，是该分区自己注意力层之后的下一个本地索引
  （`mtp_attn_local_`，`engine_kvpool.cpp:202-208`；解析在 `:128-138`）。注意 MTP 的硬门禁要求
  `multi_dev && md_xmx`（`engine.cpp:141-149`），**所以单设备（虚拟 USM）路径永远不会带 MTP**——
  这也是 `kv_setup` 的非多设备分支不需要给 MTP 多留一层的原因。

### 2.2 scale 平面

`kv_scale_stride` / `kv_v_scale_stride`（`engine.h:229-230`）仅在任一侧带 scale 时存在
（`has_scales = kv_dtype_has_scales(k) || kv_dtype_has_scales(v)`，`engine_kvpool.cpp` 里解 `has_scales` 处）：
`n_blocks * n_head_kv * kBlockSize * (head_dim/kI8Q) * sizeof(half)`（`kI8Q = 32`，
`kernels.h:12`；写入点 `engine_kvpool.cpp:214-216`、`:249-251`、`:335-336`）。

**两侧永远相等**（三处都是 `kv_v_scale_stride = kv_scale_stride;`）：scale 平面的几何只依赖
`head_dim` 与 `kI8Q`，而 i4 与 i8 共用同一个 per-32 fp16 平面——只有数据平面的行字节数不同。
scale 平面是一次性普通 USM 分配（虚拟路径 `dalloc_bytes` → `malloc_device`，CPU 路径
`malloc_host`，多设备路径 `dev_alloc_on`），**不是虚拟内存**，按完整预留分配，不参与增长/收缩。

#### 平面减半的代价（实测，未采纳）

scale 平面是 i4 负载的 12.5%、i8 的 6.25%。两种减半方式在真实 dump 上量过（`PF_DUMP_KV`，
0.8B 层 3/7/11，48 token，对引擎自己的量化器，以当前 G=32/f16 为基准的相对 RMS 重建误差）：

| 方案 | 平面 | i8 | i4 |
|---|---|---:|---:|
| per-64 分组 + f16 scale | −50% | **1.182x** | **1.176x** |
| per-32 分组 + uint8 code + 每 (32-token block, group) 一个 f16 base | −47% | **1.011x** | **1.000x** |

原因不是运气：uint8 code 把每组 scale 解析到 1/255 = 0.4%，**低于** i8 的量化步长（amax 的 1/254）、
比 i4 步长还小 ~17 倍，于是藏在自己所依附的量化误差里；而 per-64 的 18% 是 **format 无关**的，因为
量化器是 scale-invariant 的——误差是 `amax/p`，p 取多少都一样，所以加宽分组对两个位宽惩罚相同。
这是**分组**的代价，不是**位宽**的代价。

跳过它的理由只剩收益：**256k 时 +1.5% 上下文，1M 时 +6%**（而 1M 无论如何要 4 张卡 + 3.4 h
prefill），对 `kI8Q` 这 12 处 layout 引用不划算。注意**不是**因为 on-disk 格式会坏：`pc_disk.cpp`
的记录头带 `kVersion`，`read_header` 对任何不匹配返回 1（unknown format），启动扫描把它留在盘上、
计入预算，但**从不解析也从不服务**——所以布局变动的代价只是一次冷缓存。两个数都是重建 RMS，
不是 AGENTS 里记的端到端 mean|diff|，而且都来自 0.8B：uint8 方案的代价取决于一个 32-token block 内
各组最大值有多接近，那是模型属性，上 27B 前应当重测。

### 2.3 总量报告

`kv_bytes_total()` / `kv_bytes_cap()`（`engine.h:648-653`）报告已提交/预留的 K+V 总字节，用的是
`(kv_block_bytes() + kv_v_block_bytes()) * attn_layers() * {pool_blocks,pool_cap}`——所以混型时
K 与 V 各按自己的行宽计入。

---

## 3. 池的建立（`kv_setup`，`engine_kvpool.cpp:182-363`）

由 `alloc_buffers` 以 `n_blocks = pool_cap`、`pool_initial` 为启动块数调用（`engine.cpp:1718`）。
`kv_setup` 开头有**三条**互斥分支，前两条都会**跳过虚拟 USM**。

### 3.1 CPU（`cpu_mode`，`engine_kvpool.cpp:239-267`）

`kv_virtual = false`，`initial_blocks` 被**强制改成 `pool_cap`**，把整个预留一次性用
`sycl::malloc_host` 提交（K/V 与 scale 平面都是主机 USM），随后 `kv_grow` 只做空闲链表记账。没有
虚拟内存与 map/shrink，因此 CPU 上 `kv_read_vec` / 前缀缓存的磁盘序列化都是主机拷贝。

### 3.2 多设备（`multi_dev`，`engine_kvpool.cpp` 的 `kv_setup` 多设备分支）

`kv_layer_stride = pool_cap * block_bytes` 全局一致，但每个设备只有一份包含**该设备注意力层**的池
（`dev_kpool_[d]`/`dev_vpool_[d]` + scale 平面，`engine.h` 的 `dev_kpool_`/`dev_vpool_`），由 `layer_attn_local_[il]` 索引；
block id 全局一致，所以 block table 只需一份。分配走 `dev_alloc_on(d, ...)`（`engine.cpp` 的 `dev_alloc_on`）：
GPU 分区给该设备队列上的 device USM，CPU 分区给主队列上的 host USM。没有注意力层的分区被跳过。
`kv_release_pool`（`engine_kvpool.cpp:474-497`）按分区释放，GPU 池在各自队列上 `sycl::free`、
CPU 池在主队列上。

`kv_layer_ptrs(a, ...)`（`engine_kvpool.cpp:126-162`）与 `attn_dev(a)`（`:164-180`）把**全局**注意力
层号解析到“设备 + 该设备的本地层号”，前缀缓存的序列化/反序列化就靠它们（`engine_prefix_cache.cpp` 的 `pc_serialize_block`、
`:154`）。

下面的流程针对单设备 GPU（虚拟 USM）。

### 3.3 虚拟路径的步骤

1. **2 MB 颗粒度推导**：`mb2 = 2<<20`。原因（注释 `engine_kvpool.cpp:269-274`）：Level Zero 要求
   ≥ 2 MB 的映射必须 2 MB 对齐，所以层 stride 与 extent 起点都保持在 2 MB 边界。块大小决定每 2 MB
   能放多少块，于是分别算 `align_k = ceil(mb2 / block_bytes)`、`align_v = ceil(mb2 / v_block_bytes)`，
   `align = lcm(align_k, align_v)`，`kv_align_blocks = align`。**取最小公倍数而不是 K 的那一个**，
   是因为 K 与 V 可以有不同的块大小（`--kv-type i4:i8`），两个方向都要落在 2 MB 边界上。
   注释给的例子是 fp32 下一块 64 KB（每 2 MB 32 块）、2 字节元素减半（每 2 MB 64 块）。
   按两个参考模型的实际形状（`head_dim = 256`，0.8B `n_head_kv = 2`、27B `n_head_kv = 4`，
   见 [11-qwen35-model.md](11-qwen35-model.md) §2/§9）算出来的每 2 MB 块数：

   | 类型 | 0.8B 块字节 / 每 2 MB | 27B 块字节 / 每 2 MB |
   |---|---|---|
   | `f32` | 64 KB / 32 | 128 KB / 16 |
   | `bf16` | 32 KB / 64 | 64 KB / 32 |
   | `i8` | 16 KB / 128 | 32 KB / 64 |
   | `i4` | 8 KB / 256 | 16 KB / 128 |

   `i4:i8` 因此取 `lcm(256,128) = 256`（0.8B）/`lcm(128,64) = 128`（27B）——混型按**较大**的颗粒走。
2. `pool_cap` 向上取整到 `align` 的倍数（不足一个颗粒时提到一个颗粒）；`initial_blocks` 先 clamp 到
   `pool_cap`、再向上对齐、之后再 clamp 一次（`engine_kvpool.cpp:279-293`），保证“已提交 ≤ 预留”且
   extent 起点不落在半个颗粒上。
3. **预留**：分别对 K（`kv_vbase`）与 V（`kv_vbase_v`）调用 `sx::reserve_virtual_mem`。大小先按
   **1 MB** 向上取整（`k_reserve` / `v_reserve`，`:296-297`），再按设备内存粒度 `gran` 对齐
   （`:299-303`）；`kv_reserve_bytes` 与 `kv_reserve_bytes_v`（`engine.h:252-253`）分别记录两侧的
   实际预留大小——V 的块宽可以不等于 K，所以两边是独立的一次预留。
4. **对齐探测**：`kv_map2 = ok && kv_vbase % mb2 == 0 && kv_vbase_v % mb2 == 0 &&
   (pool_cap*block_bytes) % mb2 == 0 && (pool_cap*v_block_bytes) % mb2 == 0`（`:305-306`）——**K 与 V
   两个 stride 都要满足**。虚拟 USM 不可用时捕获异常、打印 `[kv] virtual USM unavailable` 并**回退**。
5. **次 2 MB 回退**（虚拟但不对齐）：此时只要求 64 KB 对齐（预留本身保证了），所以增长限制在
   < 2 MB 的 extent —— `kv_grow` 的上限 16 块（`:374-376`）——并把 `initial_blocks` 压到 16、打印
   `[kv] reservation is not 2 MB aligned: growth uses <2 MB extents`（`:311-319`）。
6. **固定回退**（`!kv_virtual`）：`pool_cap = n_blocks = initial_blocks`，stride 随之缩小，用
   `malloc_device` 分配 `initial_blocks * block_bytes * n_attn`；**不能增长也不能收缩**
   （`:320-325`、`:348-353`）。`pool_print` 会打上 `(fixed)` 标记。
7. 虚拟情况下 `n_blocks = pool_cap`，`kv_layer_stride` / `kv_v_layer_stride` 基于预留（`:326-330`）。
8. **scale 平面**一次性按 `n_blocks` 分配（`kv_scale_stride * n_attn`），不参与虚拟增长。
9. 初始化 `block_used_`（0）、`block_extent_`（-1）、`pc_block_node_`（-1）；`pool_initial =
   initial_blocks`；`pool_blocks = 0`（`:343-347`）。
10. `d_kpool = (void*)kv_vbase`、`d_vpool = (void*)kv_vbase_v`；`kv_initializing = true` 下提交初始
    extent，失败即抛 `KV pool reservation failed`（`:354-362`）。`kv_initializing` 期间不打
    `pool_print`，最后统一打一次 `init`。

---

## 4. 物理 extent（`kv_grow` / `kv_shrink` / `kv_release_pool`）

### 4.1 增长（`engine_kvpool.cpp:366-429`）

* `want < 1` 或 `pool_blocks >= pool_cap` 直接返回 false。
* `kv_map2` 时 `want` 向上取整到 `kv_align_blocks`；否则上限 16 块（< 2 MB 规则；固定回退同样走这条
  分支，但那时 `kv_virtual == false`，`physical_mem` 的 map 段整段跳过，只记账）。
* `count = min(want, pool_cap - pool_blocks)`，`first = pool_blocks`（块 id 连续，从最低优先）。
* K 与 V 各自的 `map_bytes` 按设备 granularity 向上取整（取不到就退回不取整，`:383-388`）。
* **每层物理映射**：对每个 attention 层 `il`（层数取 `attn_layers()`），`off = il*layer_bytes +
  first*block_bytes`，为 `(layer, K)` 与 `(layer, V)` 各构造一个 `sx::physical_mem` 并 `map()` 到
  `kv_vbase+off` / `kv_vbase_v+off`（`read_write`）。注释（`engine.h:806-807`）：一个 `physical_mem`
  对象只能映射一次，所以每层每侧一个，存进 `kv_extent::phys_k/phys_v`（`engine.h:801-809`）。
* 映射异常时该 extent 保持未映射、打印 `[kv] grow failed (...)` 并返回 false——那些块只是不可用，
  池停在原处。
* extent 的块推进 `free_blocks_`，设 `block_extent_[b]`；追加 extent、`extent_used_.push_back(0)`、
  `pool_blocks += count`、`pool_grows++`。
* 虚拟且非初始化时 `pool_print("grow")`。

### 4.2 收缩（`engine_kvpool.cpp:431-472`）

* 无虚拟 USM 时空操作。
* 循环条件：顶部 extent 存在、`extent_used_.back() == 0`、且 `kv_extents_.back().first >= pool_initial`。
  `pool_initial` 守卫保证历史固定池不抖动，只释放突发增长出来的部分。
* 从最小堆中移除该 extent 的块（排空后重建，保持最小堆不变量），逐层 `unmap`，清
  `phys_k/phys_v`，把该 extent 的 `block_extent_` 置 -1，`pool_blocks -= ex.count`，弹出，
  `pool_shrinks++`，`pool_print("shrink")`。

### 4.3 拆除（`engine_kvpool.cpp:474-536`）

* 多设备：见 §3.2。
* 虚拟：逐 extent 逐层 unmap，`sx::free_virtual_mem` 释放两个基址（用各自记录的
  `kv_reserve_bytes` / `kv_reserve_bytes_v`），`kv_vbase = kv_vbase_v = 0`。
* 固定：`sycl::free` 池与 scale 平面。

由 `~engine` 在 `pool_print("exit")` / `pc_print_stats("exit")` 之后调用（`engine.cpp:625-633`）。

---

## 5. 块分配器与块表

* **空闲表**是 `std::priority_queue<int, vector<int>, greater<int>>` 最小堆（`engine.h` 的 `free_blocks_`），因此分配
  总是取最小块 id。这保证高地址 extent 容易完全空闲，从而可被收缩。
* `alloc_block()`（`engine_kvpool.cpp:549-576`）：
  1. 弹出最小空闲块（防御性跳过 `block_used_` 已置位的块，保证不重复发放）；
  2. 置 `block_used_[b]=1`、`extent_used_[block_extent_[b]]++`、更新 `pool_peak`；
  3. 若空闲表为空：前缀缓存开启时先 `pc_evict_lru()`（可能把节点降级到 RAM/磁盘再腾块），再在 cap 内
     `kv_grow(pool_chunk)`，两者都失败才返回 -1。
  * `pool_chunk` 默认 64 块（`engine.h:246`），`PF_KV_GROW` 可覆盖并被夹到 ≥ 1
    （`engine.cpp:497-505`）。
* `free_block(b)`（`engine_kvpool.cpp:578-594`）：越界与重复释放守卫；清 `block_used_`、减
  `extent_used_`、推回最小堆；虚拟模式下若该块属于**顶部** extent，调 `kv_shrink()`。
* `set_table(slot, blocks)`（`engine_kvpool.cpp:596-612`）：把最多 `max_blocks` 个块 id 写入
  `h_tables` 第 `slot` 行，再整行拷到 `d_tables`（`[kMaxB][max_blocks]`）。
  `max_blocks = ceil(max_seq/kBlockSize)`（`engine.cpp:467`）。**多设备路径下 `d_tables` 是主机 USM、
  由每个队列的 kernel 读，所以那一行用主机 `memcpy` 直接写，而不是在主队列上排一次拷贝**——后者
  第二张卡并没有排在它之后。
* 记账：`pool_free_blocks() = free_blocks_.size()`，`pool_used_blocks() = pool_blocks - free`
  （`engine.h:634-639`，均为 inline）。每块元数据 `block_used_`、`block_extent_`、`extent_used_`。
* `pool_print(tag)`（`engine_kvpool.cpp:538-547`）输出已提交/上限块数与 MB、in-use、peak、
  grow/shrink 次数，非虚拟时追加 `(fixed)`。MB 用
  `(kv_block_bytes() + kv_v_block_bytes()) * attn_layers()`，所以混型按真实几何报。

---

## 6. 不变量

* 地址范围只预留一次；只有 `physical_mem` 的提交/解映射改变物理占用。图因此持有稳定的基址。
* extent 在块 id 上连续（`first = pool_blocks`）；`kv_map2` 时层 stride 与 extent 起点在 2 MB 边界，
  满足 Level Zero 对齐要求；混型时取 K/V 两个颗粒的最小公倍数，两个方向都对齐。
* 最低优先分配 + 只从完全空闲的顶 extent（`>= pool_initial`）收缩，提供近似 LIFO 的回收且无碎片。
* scale 平面独立、固定大小、**不在**虚拟范围里；K 与 V 的 scale stride 恒相等。
* block id 全局一致：多设备下每张卡只存自己那几层，但同一张块表对所有卡有效。
* `PF_KV_CAP_MB` 同时是 KV 池的块上限来源，也是三层前缀缓存预算之和的上限，见
  [06-prefix-cache.md](06-prefix-cache.md) §6。

---

## 7. KV 存储类型

由 `kv_type.h` / `kv_type.cpp` 定义与选择，完整说明见 [03-kernels.md](03-kernels.md) §11。回顾：

* `i8`（默认）：对称 int8 `[-127,127]` + 每 32 head dim 一个 fp16 scale；数据平面
  `[block][kv head][token][head_dim]`，scale 平面 `[block][kv head][token][head_dim/32]`。
  3 KB/token/layer（K+V）。
* `i4`：每字节打包两个对称 4-bit 值 `[-7,7]`（低 nibble = 偶数 dim，二补码）+ 同样的每 32 head dim
  fp16 scale；数据平面 `[block][kv head][token][head_dim/2]`，scale 平面与 i8 相同。
  1.5 KB/token/layer，是 i8 的一半。
* `bf16`：2 字节元素；`f16`：同尺寸、指数更小；`f32`：历史路径，12 KB/token/layer。

int8/int4 的量化与反量化在 kernel 内完成（`qk_norm_rope` 写、`attn` 读），中间算术保持 fp32。
KV 类型通过 `--kv-type`（CLI）或 `PF_KV_TYPE`（env，`PF_KV_F32=1` / `PF_KV_BF16=0` 是 f32 的别名）
选择，解析顺序与拼写表见 `kv_type.cpp:19-37` 与 `:86-102`；`--kv-type` 通过
`kv_dtype_set_kv`（`kv_type.h:48`）在 engine 构造前覆盖进程默认值。
`--kv-type K:V` 可分别指定 K 与 V，**只有携带 scale 的 i4/i8 允许混型**
（`kv_dtype_mix_ok`，`kv_type.cpp:43-52`）——它们的行字节数不同但共用同一个 per-32 平面。

**误差由 V 主导，不是 K**：27B 对 fp32 CPU 参考的 mean|diff| 是 i8 0.035、`i4:i8` 0.061、i4 0.183、
`i8:i4` 1.49。所以 `i4:i8`（4-bit 键、8-bit 值）用 i8 的 75% 字节（27B 上 24 KB/token 对 i8 的 32、
i4 的 16）拿到接近 i8 的精度，并能装下 262144 上下文；`i8:i4` 是镜像，明显更差，反过来印证敏感侧是 V。

辅助接口（`kv_type.h`）：`kv_dtype_bytes`（`:52-54`）给存储字节数、`kv_dtype_bits`（`:57-66`）给每元素
位数（i4 每字节两元素）、`kv_dtype_row_bytes`（`:69-71`）给一行（一个 token 的一个 kv head）的字节数、
`kv_dtype_has_scales`（`:74-76`）判断是否有独立 scale 平面、`kv_ld_host`（`:81-97`）是主机侧元素读取
（i8/i4 返回反量化后的值）。pool 的 scale 处理在 `engine::kv_read_vec`
（`engine_kvpool.cpp` 的 `engine::kv_block_elem_off`）：它按 `[block][kv head]` 单元取数据区间与被触及单元的 scale 行，主机上
反量化成 fp32，供 stage test 与 CPU 侧读出。

---

## 8. 池的定尺

尺寸参数分三个层次：`--ctx` 定每序列能有多长、`--blocks` 定启动即提交多少、`--kv-cap-mb` 给增长封顶，
`PF_KV_GROW` 决定每次增长多少（`main.cpp:326-340`、`engine.cpp:466-505`）：

| 参数 | 含义 | 默认 |
|---|---|---|
| `--ctx N`（`PF_CTX`，或 `full`） | 每序列最大 token 数 | 20480 |
| `--blocks N` | **启动即提交**的块数（`pool_initial`） | 见下 |
| `--kv-cap-mb N`（`PF_KV_CAP_MB`） | **预留**地址范围的块上限（`pool_cap`） | `<0` = 按 `--ctx` 自动 |
| `PF_KV_GROW` | 每次增长提交的块数（`pool_chunk`） | 64 |

* `--blocks` 的默认：未显式给 `--ctx`/`--ctx full` 时是 512 块（16k token，懒提交）；显式给 `--ctx`
  时是 `max(512, ceil(ctx/32))`；`--ctx full` 回到 512（可能非常大的 context 不适合 eager）。
  显式 `--kv-cap-mb` 会把启动块数抬到至少 `ceil(ctx/32)`，否则池装不下整个 context，prefill 会失败。
* `--kv-cap-mb` 的三种取值（`engine.cpp:482-488`）：`> 0` 换算成
  `N MB / ((kv_block_bytes + kv_v_block_bytes) * n_attn)` 个块（`n_attn` 含 MTP 层，所以 MTP 的那份
  切片计入 `--kv-cap-mb`）；`< 0`（CLI 默认 `INT_MIN` 归一化成 -1）= `max_blocks`，即刚好够 `--ctx`
  用；`0` = 不增长。
* `pool_cap = max(n_blocks_, cap_blocks)`——**`--kv-cap-mb` 小于 `--blocks` 时不会缩小预留**，
  它只是给增长封顶；真正的驻留上限永远是启动块数加上按需增长的部分。
* 前缀缓存开启时预留至少 `pc_max_states` 块：每个**有状态**的缓存节点还占一个 KV 块
  （`engine.cpp:489-494`）。这仍然是地址空间而非已提交内存。

---

## 9. 相关环境变量

`PF_CTX`、`PF_KV_CAP_MB`、`PF_KV_GROW`、`PF_KV_TYPE`、`PF_KV_F32`、`PF_KV_BF16`。总表以
`AGENTS.md#environment-variables` 为准。前缀缓存的三层预算另见
[06-prefix-cache.md](06-prefix-cache.md) §9。

---

## 10. 进一步阅读

* [06-prefix-cache.md](06-prefix-cache.md)：块被前缀缓存节点占用/释放的规则、`pc_block_node_` 与
  `alloc_block()` 的互相调用。
* [03-kernels.md](03-kernels.md) §11：KV 存储类型与选择优先级；§6.3 KV 写入。
* [04-engine.md](04-engine.md)：块表 `step_info` 与多设备分区的关系。
* `tests/backend/gpu/test_pc_gpu.cpp`、`test_pc_ram_gpu.cpp`、`tests/backend/cpu/test_pc_cpu.cpp`：
  在真实池上验证块分配、spill/promote 往返；测试矩阵见
  [12-build-and-testing.md](12-build-and-testing.md)。
