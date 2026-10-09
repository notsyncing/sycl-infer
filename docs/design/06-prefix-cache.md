# 设计 06：跨请求前缀缓存

覆盖 `src/engine/engine_prefix_cache.cpp`、`pc_ram.{h,cpp}`、`pc_disk.{h,cpp}`、`engine.h` 中的
`pc_*` 字段，以及 `engine.cpp` 构造期的三层预算定尺。KV 块池见 [05-kv-cache.md](05-kv-cache.md)。

---

## 1. 动机

多个请求常共享同一段 prompt 前缀（系统提示、few-shot 示例、多轮历史）。如果每个请求都从头
prefill，会重复大量计算。前缀缓存按 32-token 块缓存已 prefill 的结果，命中时直接复用。

难点在于本模型是**混合架构**：attention 层的 KV 可以按块缓存，但 GDN 层持有对全部历史 token 的
固定递归状态，conv 层持有最后 `conv_k-1` 个 tap。**光有 KV 块无法重建递归状态**，因此一个缓存
节点只有在携带边界处的递归状态检查点时才可用作恢复点。

> **CPU 与多设备**：CPU 单设备完全支持该缓存（KV 都在主机 USM，`kv_setup` 的 CPU 分支把整个预留
> 一次性 `malloc_host` 提交，检查点池 `d_pc_states` 也走 `alloc_bytes` 的 host USM 分支，
> `engine_prefix_cache.cpp` 的 `pc_serialize_block`），序列化是主机拷贝，`test_pc_cpu` 覆盖磁盘 spill/promote 往返。
>
> 多设备（`--layer-map`）同样支持：block id 是全局的，`pc_block_node_` / `pc_map_` 也只有一份。
> recr 状态检查点只有一个池 `d_pc_states`，全 GPU 的映射下它是**主设备 device USM**
> （`alloc_bytes`，`engine.cpp` 的 `alloc_bytes`），混合 CPU/GPU 映射下 `host_act` 让它落在主机 USM；
> `pc_restore_state` 逐分区用该设备自己的队列拷进 `as_[dev].gdn_state/conv_state` 再 `sync_all()`
> （`engine_prefix_cache.cpp:485-501`）。块的序列化/恢复通过 `engine::kv_layer_ptrs` 把每个全局注意力
> 层解析到所属设备的池（`dev_kpool_[d] + layer_attn_local_[il] * kv_layer_stride`）与该设备的队列
> （`engine_prefix_cache.cpp` 的 `pc_serialize_block`、`:154`），块内偏移与单设备布局一致。
>
> **MTP 路径**也支持：`generate_mtp` 对 prompt 调 `pc_admit`（`engine_mtp.cpp:736`，命中数还会打印
> `[mtp] prefix cache: reused N prompt tokens`）、prefill 后调 `pc_commit`（`engine_mtp.cpp:847-849`），
> 与调度器是同一套契约。

---

## 2. 缓存节点

```cpp
struct pc_node {                    // engine.h:842-850
    uint64_t hash;                  // 链式哈希
    int32_t  block;                 // KV 池块 id
    int32_t  refcount;              // 持有该块的活跃序列数
    uint64_t lru;
    int32_t  state;                 // 检查点槽或 -1
    int32_t  depth;                 // 链中的块数（1 = root）
    int32_t  toks[kBlockSize];      // 精确 token id 校验
};
```

索引（`engine.h` 的 `pc_nodes_`/`pc_map_`）：`pc_nodes_`（vector，swap-remove）、`pc_map_`（hash → 节点 idx）、
`pc_block_node_`（块 id → 节点 idx）。后两者必须与 `pc_nodes_` 同步维护，所以下面每处改动节点集合的
地方都要一起改这三个容器。

### 2.1 链式哈希

`pc_hash_block(chain, toks)`（`engine_prefix_cache.cpp:23-33`）：FNV-1a（offset basis
`0xcbf29ce484222325`，prime `0x100000001b3`）按字节遍历该块 32 个 token id（每个 token 4 字节），
以父链 `chain` 为种子（零链用 offset basis）。根块的哈希覆盖了它之前的所有 token，因此节点只在它
被创建的精确上下文中有效——同一段文本出现在两个不同前缀的同一位置会得到两个不同的 hash，这是特性
而不是浪费。

### 2.2 精确校验

64 位哈希不足以保证唯一，因此 `pc_find`（`engine_prefix_cache.cpp:441-450`）命中后必须 `memcmp` 存的
`toks`，不符返回 -1。RAM（`pc_ram.cpp:19-28`）与磁盘（`pc_disk.cpp:175-184`）的 `find` 做同样的
token 校验，所以三层之间传递的 key 永远是 (hash, 32 个 token)。

`pc_add_node(chain, toks, block, depth)`（`engine_prefix_cache.cpp:452-474`）：哈希已存在、块越界、
或块已被别的节点占用则拒绝；否则 `refcount=1`（创建序列持有）、`lru=++pc_clock`，拷贝 token，追加节点
并更新两个映射。`depth` 由调用者给，是链里的块序号（`pc_commit` 传 `i+1`，`:882`）。

### 2.3 LRU 与引用计数

`pc_clock` 单调递增，在 add/promote/commit/touch/retire/attach 时 bump。`refcount` 统计活跃序列持有
数：`pc_add_node` 初始 1，`pc_admit` 每 pin 一个块递增（`:698`），`pc_retire` 递减并在归零时 stamp
`lru`（`:911-913`），使节点可被驱逐。驱逐只考虑 `refcount <= 0` 且块有效的节点。

### 2.4 驱逐

`pc_evict_lru()`（`engine_prefix_cache.cpp:595-617`）线性扫一遍 `pc_nodes_`，取 `refcount <= 0` 且
`block >= 0` 中 `lru` 最小的一个；有下层时先 `pc_demote_node` 再 `pc_evict_node`，并计
`pc_stat_evict_nodes`。它同时是 `alloc_block()` 的降压阀（块用尽时先回收缓存节点，见
[05-kv-cache.md](05-kv-cache.md) §5）。

`pc_evict_node(ni)`（`:574-593`）释放检查点（`pc_state_drop`）、清 `pc_block_node_`、从 `pc_map_`
删除，然后 **swap-remove**：若删的不是最后一个元素，要把末元素搬到 `ni` 并同步修 `pc_map_`、
被搬移块的 `pc_block_node_` 以及 `pc_state_owner`。遗漏反向块索引会使后续 `pc_retire` 读到
已经越界的节点下标；`test_pc_gpu` 用非末节点驱逐覆盖此不变量。最后 `free_block(nd.block)`
把 KV 块还给池子。

---

## 3. 递归状态检查点

### 3.1 检查点池

`d_pc_states` 是 `[pc_max_states][pc_state_floats]` 设备内存（分配点 `engine.cpp:1898-1907`，
`pc_max_states == 0` 时不分配）。`pc_state_floats = n_gdn * (gdn_per_slot() + conv_per_slot())`
（`engine.cpp:267`），其中 `gdn_per_slot = dt_rank*d_state*d_state`（`engine_prefix_cache.cpp:12-15`）、
`conv_per_slot = (conv_k-1)*qkv_dim()`（`:17-20`）（**不是 `3*d_inner`**；27B 上 `qkv_dim()=10240`，
见 [11-qwen35-model.md](11-qwen35-model.md) §2）。

槽记账：`pc_state_owner[st]`（节点 idx 或 -1）、`pc_state_free`（空闲栈，初始化为倒序填充，
`engine.cpp:1903-1906`）、`pc_state_stamp[st]`（LRU）。

* `pc_state_take()`（`:504-548`）：`pc_max_states <= 0` 或无池直接 -1；优先取空闲栈顶；否则驱逐
  `pc_state_stamp` 最小的**已提交**检查点（owner 越界或 -1 的槽被跳过）。存在下层且该 owner 节点
  `refcount <= 0` 时，先把**整个节点**降级到 RAM/磁盘（检查点在那里仍然活着），再摘掉 state/owner
  并 `pc_evict_node`，腾块又腾槽同时完成；否则只分离检查点（`node.state = -1`）。**为当前 forward
  预留的槽没有 owner，永不被驱逐**——所以这一次捕获不会被自己挤掉。
* `pc_state_drop(st)`（`:550-560`）：从 owner 分离，归还空闲栈。
* `pc_state_release(st)`（`:562-572`）：有 owner 走 drop，无 owner 直接归还。
* `pc_restore_state(slot, st)`（`:476-502`）：对每个递归层把检查点中的 GDN 切片（`gdn_per`）和 conv
  切片（`conv_per`，紧跟其后）拷回该序列的 `d_gdn_state`/`d_conv_state`；多设备下目标是
  `as_[dev]` 里该分区的**本地**层号（`layer_gdn_local_`），并 `sync_all()`。

### 3.2 检查点内容与 kernel 写入

一个检查点槽内，每层布局是 `[GDN 状态 gdn_per][conv 状态 conv_per]`：

```
pc_snap = base + slot*stride + layer_off + [0, gdn_per)             // GDN
          base + slot*stride + layer_off + gdn_per + [0, conv_per)  // conv
```

`pc_snap`（`kernels.h:125-131`）由 `record_forward` 逐层填好（`engine_graph.cpp:931-938`：
`layer_off = gi * (gdn_per + conv_per)`，`gi` 是**全局** GDN 层号）。conv/GDN kernel 在 token 完成一个
32 块边界时，通过 `step_info::pc_row_slot` 查目标槽并写入：GDN 在 token 索引 `(pbase + t + 1) % 32 == 0`
处写（`gpu/kernels/gdn.cpp:122-135`），conv 在整行结束时若 `end_tok % 32 == 0` 写（`gpu/kernels/conv.cpp:92`）。
`pc_row_slot` 是 host USM 的 `step_info` 字段，`kPcMapLen = 1024`（`kernels.h:27`），因此捕获只覆盖
`max_seq <= 32768` 的边界索引。

> **同一个机制的第二种用法**：MTP 的 dry verify 需要把递归状态回滚到被接受的 draft 长度，于是
> `step_info::mtp_dt` 打开时改为**每个 token** 都快照，索引从 `pc_row_slot[t]` 取，目标是
> `d_mtp_hist_[dev]` 而不是 `d_pc_states`（`kernels.h:98-101`、`gdn.cpp:122-123`、
> `engine_graph.cpp:948-958`）。布局与前缀缓存的检查点相同，conv 平面必须逐层存在，否则会落进下一层
> 的 GDN 切片。

---

## 4. 生命周期

### 4.1 Admit（`pc_admit`，`engine_prefix_cache.cpp:619-779`）

调度器在每个序列准入时调用（`scheduler.cpp` 的 `pc_admit` 调用点），返回命中的 token 数（0 表示需要 `zero_slot`，
`scheduler.cpp:92-94`）。

1. 重置 `pc_slot_[slot]`；缓存关闭返回 0。
2. 置 `tracking = true`，让后续 prefill 捕获检查点。
3. **保留最后一个不完整块**并至少留一个 token 给真实 forward：`max_full = ((L-1)/32)*32`，
   `nblk = max_full/32`；`nblk <= 0` 视为 miss。原因：第一个采样 token 需要真实 forward 的 logits。
4. **第一遍链式遍历**：逐块计算哈希，按 **VRAM → RAM → disk** 解析（`pc_find` / `pcr->find` /
   `pcd->find`）；缺失即 break。`avail = i+1`。状态的存在性按 VRAM 节点的 `state`、RAM 的
   `meta.has_state`、磁盘的 `has_state` 依次查，任一层有就在该边界记入 `deep`。
   若 `avail<=0 || deep<=0` 则 miss。
5. **第二遍 pin/promote（到 `deep`）**：resident 节点 `refcount++` 并 touch；否则
   `pc_promote_ram` → `pc_promote_disk`，失败则 break 并标记 `failed`。pin 的块推进 `blocks`。
   提升过程中若要腾块，`pc_promote_bytes` 里的 `alloc_block` 会顺带降级别的未引用节点。
6. **部分失败恢复**：pin 到 `reach` 就失败时，把 `deep` 重算为“已到达且任一层仍有状态”的最深边界，
   unpin 其上（`i` 从 `deep` 到 `reach`），`blocks.resize(deep)`；`deep <= 0` 即 miss。
7. **边界检查点保证**：边界节点可能 resident 但状态丢了（VRAM 池淘汰），用
   `pc_attach_state_from_lower` 从 RAM/磁盘拉回；失败则 unpin 并 miss。然后 `pc_restore_state`。
8. `pcr->touch` / `pcd->touch` 记一次命中，设 `pc_slot_[slot].chain = hashes[deep-1]`、
   `registered = deep`，更新统计，返回 `deep * kBlockSize`。

> 只认分批前缀：`deep` 还要收敛到“从零算起的分批前缀”。batched GEMM 的激活 scales
> 随 M 变，混着“新整块+零散尾巴”的一批（如 33-token 的 512..544）与其前 32 行同 token
> 的纯 32 批算出不同的 qkv，经 GDN 递推放大到检查点可测分叉（0.8B 双卡 depth=17：
> sum 1231.91 vs 1237.31），恢复+不同 M 尾巴再放大到 logit 换 argmax（27B 双卡 25.8）。
> `pc_admit` 重算与 `prefill_text` 相同的 cold splits（只依赖 prompt 长度），在有状态的
> 边界里取最深且落在前缀上的（如 545 取 512 而非 544），暖重做与 cold 完全相同的后缀批。
> 100-token 单批没有中间前缀时按设计 miss（宁可重算，不做错恢复）；`test_spec
> TEST_SPEC_LONG`（545-token，cold/warm/plain 三方 max|diff|=0）是常设覆盖。

`PF_PC_DEBUG`（与 `SCHED_DEBUG` 一样是“环境变量存在即开”，不看值）会逐块打印
`[pcdbg] admit L=.. blk=.. h=.. vram=.. ram=.. disk=..`（`:653-656`）与
`[pcdbg] commit slot=.. blk=.. done=.. h=.. found=..`（`:877-880`），是排查“为什么没命中/为什么没
被捕获”的唯一入口。

### 4.2 捕获（`pc_capture_begin` + kernel 快照，`engine_prefix_cache.cpp:786-859`）

在 `prefill_chunk` 与 `prefill_batch`（均在 `engine.cpp`）之前调用：

* 目标 `step_info` 是 `pf_info_ ? pf_info_ : d_info`（`:787`）——多设备流水线里每个 chunk 有自己的
  `step_info`，快照表必须写进本次 chunk 用的那一份。
* 释放上一 forward 未提交的预留，清 `pc_pending_`，`pc_active=0`。
* 前置条件：缓存开、slot 有效、`tracking`、`n>0`、`pos0>=0`、`pos0%32==0`，且**连续性**
  `registered*32 == pos0`（链式哈希只能从连续链推导）；`last >= kPcMapLen` 也整体放弃。
* 计算 `first/last` 边界，清零 `pc_row_slot[b]`（`first+1 .. last`）。
* 从 `ps.chain` 向前走，收集**缺节点或缺状态**的边界到 `need`。
* 取 `nsel = min(nneed, pc_max_states)` 个槽。选的是**最深**的 `nsel` 个（`k` 从 `nneed-nsel` 到
  `nneed`），因为后续请求从最深匹配边界恢复；分配时仍**浅→深**迭代，于是最深的槽最新
  （注释 `:839-842`）。`pc_state_take()` 会在池满时淘汰 LRU 检查点。
* 写 `pc_row_slot[b] = st`，push `{hash, st}` 到 `pc_pending_`；最后发布 `pc_base/pc_stride` 并置
  `pc_active=1`。`pc_active` 与 `pc_row_slot` 都在 kernel 体内读，所以图回放也安全。

### 4.3 提交（`pc_commit`，`engine_prefix_cache.cpp` 的 `engine::pc_commit`）

调度器在每个 prefill chunk 之后调用（`scheduler.cpp:257-259`），MTP 在 prompt prefill 之后调用：

* 缓存关闭或 slot 非法：释放全部 pending 槽并清空，直接返回。
* 把 `[registered, done/32)` 的块注册为节点（`pc_add_node(h, tk, blocks[i], i + 1)`），并推进
  `ps.chain` / `ps.registered`——只有完整 32-token 块成为节点；即使 `pc_add_node` 因已存在而返回 -1，
  链仍照常前进（这条链已经注册过）。
* 逐个附加 pending `{hash, slot}`：节点存在且尚无状态则设 `state`/`owner`/`stamp`、bump `lru` 并计
  `pc_stat_states`；否则释放该槽。
* 清空 `pc_pending_`。

### 4.4 退休（`pc_retire`，`engine_prefix_cache.cpp:904-919`）

准入失败（`scheduler.cpp:105`）或序列退休（`scheduler.cpp` 的 `pc_retire` 调用点）时调用。重置 slot tracking；对每个
块：若被节点拥有且确属该节点则 `refcount--`，归零时 stamp `lru` 使其可驱逐；否则直接 `free_block`。

---

## 5. 三层 VRAM → RAM → 磁盘

查找顺序 **VRAM → RAM → disk**；逐出/降级顺序 **VRAM → RAM → disk → dropped**。一次提升是**移动**，
所以一条记录只存在于一层。

### 5.1 三层职责

| 层 | 数据结构 | 上限来源 | 存储内容 |
|---|---|---|---|
| VRAM | `pc_nodes_` + KV 块 + `d_pc_states` | `pc_max_states` 检查点 + KV 池 | resident 节点（块 + 槽） |
| RAM | `pc_ram_store`（`unordered_map<hash, pc_ram_entry>`） | `PF_PC_RAM_MB`（默认 512，0 关闭） | 已序列化的记录（blob + state） |
| disk | `pc_disk_store`（目录 + 内存索引） | `PF_PC_DISK_MB`（默认 1024，0 = 无界） | 每节点一个 `*.pcn` 文件 |

RAM 层的记录与磁盘记录是**同一种结构**（RAM 条目的 `meta` 就是 `pc_disk_meta`），所以降级只是“把
同一条记录放到下一层”，提升只是“把它读回来”，两层之间没有格式转换。

### 5.2 RAM 层（`pc_ram.{h,cpp}`）

`pc_ram_entry = {pc_disk_meta meta; vector<uint8_t> blob; vector<float> state;}`（`pc_ram.h:13-17`）。
`put` 拒绝大于整个预算的记录（`pc_ram.cpp:35-37`，这种记录只能直接去磁盘），并顺手刷新
`meta.blob_bytes / state_bytes / file_bytes`、bump `meta.lru`、清 `meta.name`
（`:38-42`）——这就是它能直接喂给磁盘层的原因。`enforce_budget` 把超预算的最小 LRU 项移入
`evicted` 交给调用者写磁盘（`:81-96`）；`take` 提升用（`:53-62`）；`drain` 关机 flush 用（`:98-105`）。
查找也做 token 校验（`:19-28`）。该层不做 I/O，可独立单测（`pc_ram.h:19-22`）。

### 5.3 磁盘层（`pc_disk.{h,cpp}`）

**记录格式**：每个节点一个不可变文件，名 `%016llx.pcn`（`pc_disk.cpp:52-56`）。固定小端头
（`kHeaderBytes = 168`，`pc_disk.cpp:50`）：

```
magic(4)="PCD1" version(4)=1 header_bytes(4)=168 endian(4)=0x01020304
flags(4) hash(8) depth(4) blob_bytes(4) state_bytes(4) toks(32*4)
```

字节序由 `wr_le32/rd_le32`（`:31-46`）显式写出/读入，大端主机上做字节交换，所以文件不依赖宿主字节序。
随后是 block blob 和可选 state。`write_record`（`:216-255`）先写 `.tmp` 再原子 `rename`，记录永不撕裂。
`read_header`（`:140-173`）返回 0=成功、1=版本未知（保留、不删）、2=损坏（可删）；magic、
`header_bytes`、`endian` 不符或 `blob_bytes/state_bytes` 为负都算 2。`load`（`:186-214`）可以只读
state（`blob == nullptr` 时 seek 跳过 blob）。
`open`（`:69-138`）创建目录、扫描 `*.pcn`、删除 class-2、把 class-1 计入 `opaque_bytes_`、
拒绝 `depth<=0`，用文件 mtime 初始化 `lru` 与 `lru_clock_`，重建 `index_`/`bytes_`。
**目录即索引，无独立索引文件。**
`enforce_budget`（`:310-326`）把 `bytes_ + opaque_bytes_` 一起与预算比较——未知版本的文件虽然永不
被逐出，却仍然占预算。

**模型指纹**：`pc_disk_init`（`engine_prefix_cache.cpp:40-93`）对模型身份与所有影响序列化布局的形状做
哈希：字面量 `"sycl-infer-pc"`、磁盘格式 generation `1`、`kv_k_dtype`/`kv_v_dtype`、`kBlockSize`、
全部 hparams（`n_layer/n_embd/n_ff/n_head/n_head_kv/head_dim/n_rot/n_vocab/d_state/n_group/dt_rank/
d_inner/conv_k/full_attn_interval`，`:60-63`）、`rope_sections`、`gguf.map_size`、
`general.architecture/name/file_type`，结果作为 `pc_dir` 下的 `%016llx` 子目录。两个模型可安全共享一个
基目录。换模型、换 KV 类型或换头数都会自动换目录，旧记录不会被误读。

### 5.4 层的调度（`engine_prefix_cache.cpp`）

* `pc_block_blob_bytes()`（`:104-113`）：每 attention 层 `kb + vb`（K+V），任一侧带 scale 时再加
  `2*sb`（K、V 的 scale 平面；`sb = n_head_kv * kBlockSize * (head_dim/kI8Q) * 2`），乘 `attn_layers()`。
* `pc_serialize_block` / `pc_deserialize_block`（`:115-168`）：逐层拷贝 K、V、K-scale、V-scale，
  拷贝发到该层所属的队列（`dev_queue(attn_dev(l))`），末尾 `sync_all()`——反序列化的源是调用者的
  临时 blob，必须在它析构前完成。
* `pc_serialize_state` / `pc_deserialize_state`（`:170-179`）：搬 `pc_state_floats` 个 float，
  反序列化后显式 `q.wait()`。
* **降级** `pc_demote_node`（`:250-301`）：无下层返回 false；RAM 或磁盘已有该 hash 且**数据不更少**
  （无状态时、或已有记录带状态）就只 `touch` 并返回，所以刚提升上来又被驱逐的节点是免费的。
  否则序列化一次，先 `put` 到 RAM，超预算项再 `pc_ram_to_disk`（RAM → 磁盘 → 磁盘 LRU 丢弃）；
  没有 RAM 层、或记录大于整个 RAM 预算（`put` 拒绝）时直接写磁盘。
* **提升** `pc_promote_bytes`（`:318-352`）：先校验几何与当前引擎一致（`blob_bytes ==
  pc_block_blob_bytes()`、`state_bytes == pc_state_floats*4`），否则返回 -1（“布局变了，不可加载”）；
  分配块、反序列化、必要时取检查点槽、`pc_add_node`、接线 state/owner/stamp。
  `pc_promote_ram`（`:354-373`）用 `take`（移动），失败时把记录 `put` 回去以免丢数据；
  `pc_promote_disk`（`:375-391`）读出后成功才 `erase`（移动语义），读取失败则 `erase` 掉损坏记录。
* `pc_attach_state_from_lower`（`:396-439`）：resident 节点缺状态时的补救，**从 RAM 是移动、从磁盘是
  只读**——磁盘路径不 `erase`，因为它只取走状态、记录还要留给下一次查找（VRAM 槽再次被淘汰时可以再
  attach 一次）；RAM 路径取用后记录即消失（take），失败则放回。
* `pc_node_to_disk`（`:183-210`）：绕过 RAM 的 flush 路径，同样先 `find` 判断磁盘上是否已有不更差的
  记录。

### 5.5 关机 flush

`pc_flush_to_disk()`（`:214-243`）：无 `pc_dir`、缓存关、或池已不存在（`pools_ready`，多设备下要求
至少一个 `dev_kpool_[d]` 非空）时无操作。否则 (1) `pcr->drain` 后逐条 `pc_ram_to_disk`；
(2) 把 resident `pc_nodes_` 按 **LRU 最旧优先**写磁盘，使 flush 期间的磁盘预算淘汰丢掉的是最久未用的
前缀，而不是刚写入的；最后打印 `[pcd] flushed on exit: N nodes, disk now M records (X MB)`。
`~engine` 在池与检查点仍存活时调用它（`engine.cpp:521-523`，位于 `pool_print("exit")` 与
`kv_release_pool()` 之前）。`SIGINT`/`SIGTERM` 只置一个原子标志，由 `serve()` 的 watchdog 线程转成
`srv.stop()` 让进程正常展开（`server.cpp:47-53`、`:1916-1917`），于是调度器先关、engine 析构再 flush。

---

## 6. 三层预算与 `PF_KV_CAP_MB`

全部在 `engine` 构造期一次定下（`engine.cpp` 里解三个前缀缓存预算那段）：

| 层 | 字段 | 环境变量 / flag | 默认 |
|---|---|---|---|
| VRAM | `pc_vram_bytes` | `PF_PC_VRAM_MB`（`--pc-vram-mb`）、`PF_PC_MEM_MB`（别名）、或直接 `PF_PC_STATES` | `PF_PC_STATES=8` |
| RAM | `pc_ram_bytes` | `PF_PC_RAM_MB`（`--pc-ram-mb`） | 512 MB |
| disk | `pc_disk_bytes` | `PF_PC_DISK_MB`（`--pc-disk-mb`） | 1024 MB（0 = 无界） |
| disk 目录 | `pc_dir` | `PF_PC_DIR`（`--pc-dir`） | 空 = 关闭磁盘层 |

* `pc_enabled = !(PF_PREFIX_CACHE==0)`（`:255-256`）。
* 每节点 VRAM 占用 `per_node = pc_state_floats*4 + pc_block_blob_bytes()`（`:274`，即“检查点 + 该块
  在三层里的序列化体积”）。给 VRAM 预算时 `states = max(1, vram_mb*1MB / per_node)`（`:295`）；
  **`vram_mb == 0` 直接得到 `states = 0`**（`:292-293`），也就是把 VRAM 层整个关掉；都没有则默认 8。
  显式的 `PF_PC_STATES` 优先于按字节推导。
* `pc_max_states == 0` 关闭整个缓存（`:339-341`）：没有检查点就没有可恢复点。
* RAM 层只在 `pc_enabled && pc_ram_bytes > 0` 时 `pc_ram_init()`（`:344-346`），`0` 即关闭该层。
* **`PF_KV_CAP_MB` 同时限制 KV 池与三层预算之和**。当 `pc_enabled && kv_cap_mb > 0` 且
  `vram+ram+disk > cap` 时，按固定顺序削减：**先磁盘，再 RAM，最后 VRAM**（VRAM 削减后重算
  `states` 并重新量化字节，`:324-328`）。磁盘预算被削到 0 时清空 `pc_dir`，**关闭磁盘层**（有意覆盖
  “0 = 无界”，`:355-357`）。削完打印 `[pc] tiers clamped to --kv-cap-mb ...`。
* 未给显式 cap 时，三层配置本身构成总预算。
* 前缀缓存开启时，KV 预留至少 `pc_max_states` 块（每个有状态节点也占一个 KV 块，
  `engine.cpp:489-494`）；这仍然是虚拟地址空间，不是已提交内存。

---

## 7. 统计

`pc_print_stats`（`engine_prefix_cache.cpp:921-952`）输出节点数、有状态节点数、总引用、命中率
（`hits/(hits+misses)`）、复用 token、捕获状态数、驱逐节点/状态数、检查点尺寸×数量 = 总量，以及
RAM 的 stores/loads/spills/records 与磁盘的 spills/loads/records。相关计数器字段见 `engine.h:667-668`
（命中/未命中/复用 token/捕获/驱逐）与 `engine.h` 的 `pc_flush_to_disk`/`pc_store_ram`（磁盘与 RAM 的 spill/load/store）。
每层还各有自己的 `print_stats`（`pc_ram.cpp:107-113`、`pc_disk.cpp:328-335`），由 `~engine` 在
`pool_print("exit")` 之后逐层调用（`engine.cpp:625-632`）。

服务端把每序列的命中数 `sequence::reused` 作为 OpenAI `usage.prompt_tokens_details.cached_tokens`
以及 `usage.prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` 返回（详见
[09-server.md](09-server.md) §2.6/§8.2）。多模态路径绕过前缀缓存，故 `cached_tokens` 恒为 0。

## 8. 不变量：`sync_all()` 必须在 CPU 后端上真的等待

本文档在 §3.1/§4.1/§4.4/§5.4 多次依赖 `engine::sync_all()`。**在 CPU 分区上它曾经是
一个空操作，而这个空操作是一条堆 use-after-free，不只是慢。**

`cpu_backend.cpp` 在 `SI_CPU_SOURCES` 里、被 `-fno-sycl` 编译（在 device pass 之外），
所以它拿不到队列，`compute_backend::synchronize()` 的 override 编译通过但什么也不做。
`engine::sync_all()` 有约 24 个调用点，于是在 CPU 分区上**每一个都是 no-op**。前缀缓存
随后把一个临时 `std::vector` 交给异步的 `queue::memcpy` 就返回——拷贝还在飞，vector 已经
被释放。

诊断轨迹值得记住，因为它说明了窄判据为什么危险：**KV 比对通过了（36/36 块逐字节相同）
而 logits 不一致**，因为被破坏的是更靠后的一次主机→设备拷贝，不是 KV 那次。`test_pc_cpu`
原本只比对三块 layer-0 的 K，扩到覆盖全部块偏移相位的 36 个单元后才定位到故障。

修复是在构造时传入一个 waiter：单设备 `[this]{ q.wait(); }`，多设备
`[this, i]{ dev_queue(i).wait(); }`。**CPU 分区的 `dev_queues_` 条目是 null，不能直接
用它当下标——会段错误。**

一般化的教训：**一个只因基类声明才存在的 override 是一个等着发作的静默 no-op**；若某个
virtual 在某个后端上不起作用，就断言它，不要留空。

---

## 9. 相关环境变量

`PF_PREFIX_CACHE`、`PF_PC_STATES`、`PF_PC_VRAM_MB`、`PF_PC_MEM_MB`、`PF_PC_RAM_MB`、`PF_PC_DIR`、
`PF_PC_DISK_MB`、`PF_PC_DEBUG`、`PF_KV_CAP_MB`。KV 池本身的环境变量见
[05-kv-cache.md](05-kv-cache.md) §9；总表以 `AGENTS.md#environment-variables` 为准。

---

## 10. 进一步阅读

* `tests/backend/cpu/test_pc_disk.cpp`（磁盘格式/LRU/重开）、`test_pc_ram.cpp`（RAM 层）、
  `tests/backend/cpu/test_pc_cpu.cpp`（主机后端的 paged attention + 磁盘 tier 往返）、
  `tests/backend/gpu/test_pc_gpu.cpp`（磁盘 spill + promote 往返）、
  `tests/backend/gpu/test_pc_ram_gpu.cpp`（VRAM→RAM→VRAM 往返）。测试矩阵见
  [12-build-and-testing.md](12-build-and-testing.md)。
* [05-kv-cache.md](05-kv-cache.md)：`alloc_block()` / `free_block()` 的降压阀就是 `pc_evict_lru()`，
  而块表 `d_tables` 与 `pc_block_node_` 由 `set_table` / `pc_retire` 分别维护。
* [11-qwen35-model.md](11-qwen35-model.md) §2/§5：GDN 与 conv 的几何，即检查点切片的来源。
* [04-engine.md](04-engine.md)：prefix cache 与多设备 phase 的关系。
