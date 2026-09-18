# 设计 06：跨请求前缀缓存

覆盖 `src/engine/engine_prefix_cache.cpp`、`pc_ram.{h,cpp}`、`pc_disk.{h,cpp}` 及 `engine.h` 中的
`pc_*` 字段。KV 块池见 [05-kv-cache.md](05-kv-cache.md)。

---

## 1. 动机

多个请求常共享同一段 prompt 前缀（系统提示、few-shot 示例、多轮历史）。如果每个请求都从头
prefill，会重复大量计算。前缀缓存按 32-token 块缓存已 prefill 的结果，命中时直接复用。

难点在于本模型是**混合架构**：attention 层的 KV 可以按块缓存，但 GDN 层持有对全部历史 token 的
固定递归状态，conv 层持有最后 `conv_k-1` 个 tap。**光有 KV 块无法重建递归状态**，因此一个缓存
节点只有在携带边界处的递归状态检查点时才可用作恢复点。

> **CPU 与多设备**：CPU 单设备完全支持该缓存（KV/状态都在主机 USM，序列化是主机拷贝），
> `test_pc_cpu` 覆盖磁盘 spill/promote 往返。多设备（`--layer-map`）会在构造时关闭该缓存
> （`pc_enabled = false`）：三层记录需要把每个注意力层的 KV 从对应设备的池里拼出来，暂未实现。

---

## 2. 缓存节点

```cpp
struct pc_node {                    // engine.h:422-430
    uint64_t hash;                  // 链式哈希
    int32_t  block;                 // KV 池块 id
    int32_t  refcount;              // 持有该块的活跃序列数
    uint64_t lru;
    int32_t  state;                 // 检查点槽或 -1
    int32_t  depth;                 // 链中的块数（1 = root）
    int32_t  toks[kBlockSize];      // 精确 token id 校验
};
```

索引（`engine.h:431-433`）：`pc_nodes_`（vector，swap-remove）、`pc_map_`（hash → 节点 idx）、
`pc_block_node_`（块 id → 节点 idx）。

### 2.1 链式哈希

`pc_hash_block(chain, toks)`（`engine_prefix_cache.cpp:23-33`）：FNV-1a 遍历该块 32 个 token id，
以父链 `chain` 为种子（零链用 offset basis）。根块的哈希覆盖了它之前的所有 token，因此节点只在它
被创建的精确上下文中有效。

### 2.2 精确校验

64 位哈希不足以保证唯一，因此 `pc_find` 命中后必须 `memcmp` 存的 `toks`，不符返回 -1。RAM/磁盘
store 也做同样校验。

`pc_add_node(chain, toks, block, depth)`：若哈希已存在、块越界、块已被节点占用则拒绝；`refcount=1`
（创建序列持有），`lru=++pc_clock`，拷贝 token，追加节点并更新两个映射。`depth = 块索引 + 1`。

### 2.3 LRU 与引用计数

`pc_clock` 单调递增，在 add/promote/commit/touch/retire/attach 时 bump。`refcount` 统计活跃序列持有
数：`pc_add_node` 初始 1，`pc_admit` 每 pin 一个块递增，`pc_retire` 递减并在归零时 stamp `lru`，
使节点可被驱逐。驱逐只考虑 `refcount <= 0` 且块有效的节点。

---

## 3. 递归状态检查点

### 3.1 检查点池

`d_pc_states` 是 `[pc_max_states][pc_state_floats]` 设备内存。`pc_state_floats = n_gdn *
(gdn_per_slot() + conv_per_slot())`，其中 `gdn_per_slot = dt_rank*d_state*d_state`、
`conv_per_slot = (conv_k-1)*3*d_inner`（约 19 MB/检查点）。

槽记账：`pc_state_owner[st]`（节点 idx 或 -1）、`pc_state_free`（空闲栈）、`pc_state_stamp[st]`（LRU）。

* `pc_state_take()`：优先取空闲槽；否则驱逐 `pc_state_stamp` 最小的已提交检查点。存在低层且 owner
  节点未被引用时，先把**整个节点**降级，再分离并驱逐；否则只分离检查点（`node.state=-1`）。为当前
  forward 预留的槽没有 owner，永不被驱逐。
* `pc_state_drop(st)`：从 owner 分离，归还空闲栈。
* `pc_state_release(st)`：有 owner 走 drop，无 owner 直接归还。
* `pc_restore_state(slot, st)`：对每个递归层把检查点中的 GDN 切片（`gdn_per`）和 conv 切片（`conv_per`）
  拷回该序列的 `d_gdn_state`/`d_conv_state`。

### 3.2 检查点内容与 kernel 写入

一个检查点槽内，每层布局是 `[GDN 状态 gdn_per][conv 状态 conv_per]`：

```
pc_snap = base + slot*stride + layer_off + [0, gdn_per)             // GDN
          base + slot*stride + layer_off + gdn_per + [0, conv_per)  // conv
```

conv/GDN kernel 在 token 完成一个 32 块边界时，通过 `step_info::pc_row_slot` 查目标槽并写入。

---

## 4. 生命周期

### 4.1 Admit（`pc_admit`，`engine_prefix_cache.cpp:600-760`）

调度器在每个序列准入时调用，返回命中的 token 数（0 表示需要 `zero_slot`）。

1. 重置 `pc_slot_[slot]`；缓存关闭返回 0。
2. 置 `tracking = true`，让后续 prefill 捕获检查点。
3. **保留最后一个不完整块**并至少留一个 token 给真实 forward：`max_full = ((L-1)/32)*32`，
   `nblk = max_full/32`；`nblk <= 0` 视为 miss。原因：第一个采样 token 需要真实 forward 的 logits。
4. **第一遍链式遍历**：逐块计算哈希，按 **VRAM → RAM → disk** 解析；缺失即 break。`avail = i+1`。
   在任意层存在状态的边界记入 `deep`。若 `avail<=0 || deep<=0` 则 miss。
5. **第二遍 pin/promote（到 `deep`）**：resident 节点 `refcount++` 并 touch；否则
   `pc_promote_ram` → `pc_promote_disk`，失败则 break 并标记 `failed`。pin 的块推进 `blocks`。
6. **部分失败恢复**：把 `deep` 重算为“已到达且仍有状态”的最深边界，unpin 其上，截断 `blocks`。
7. **边界检查点保证**：边界节点可能 resident 但状态丢了（VRAM 池淘汰），用
   `pc_attach_state_from_lower` 从 RAM/磁盘拉回；失败则 miss 并 unpin。然后 `pc_restore_state`。
8. touch 记录，设 `pc_slot_[slot].chain = hashes[deep-1]`、`registered = deep`，更新统计，
   返回 `deep * kBlockSize`。

### 4.2 捕获（`pc_capture_begin` + kernel 快照，`:767-839`）

在 `prefill_chunk` / `prefill_batch` 前调用：

* 释放上一 forward 未提交的预留，清 `pc_pending_`，`pc_active=0`。
* 前置条件：缓存开、slot 有效、`tracking`、`n>0`、`pos0>=0`、`pos0%32==0`，且**连续性**
  `registered*32 == pos0`（链式哈希只能从连续链推导）。
* 计算 `first/last` 边界，清零 `pc_row_slot[b]`。
* 从 `ps.chain` 向前走，收集缺少节点或状态的边界到 `need`。
* 取 `nsel = min(nneed, pc_max_states)` 个槽（浅→深），写 `pc_row_slot[b] = st`，push
  `{hash, st}` 到 `pc_pending_`；发布 `pc_base/pc_stride`，最后 `pc_active=1`。

### 4.3 提交（`pc_commit`，`:841-882`）

调度器在 forward 后调用：

* `pc_active=0`；
* 把 `[registered, done/32)` 的块注册为节点（`pc_add_node(..., i+1)`），推进 `ps.chain`/`registered`；
  只有完整 32-token 块成为节点；
* 逐个附加 pending `{hash, slot}`：节点存在且尚无状态则设 state/owner/stamp，bump lru，统计；否则释放槽。

### 4.4 退休（`pc_retire`，`:884-899`）

准入失败或序列退休时调用。重置 slot tracking；对每个块：若被节点拥有则 `refcount--`，归零时 stamp
`lru`；否则直接 `free_block`。

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

### 5.2 RAM 层（`pc_ram.{h,cpp}`）

`pc_ram_entry = {pc_disk_meta meta; vector<uint8_t> blob; vector<float> state;}`。
`put` 拒绝大于整个预算的记录；`enforce_budget` 把超预算的最小 LRU 项移入 `evicted` 由调用者写磁盘；
`take` 提升用；`drain` 关机 flush 用。该层不做 I/O，可独立单测。

### 5.3 磁盘层（`pc_disk.{h,cpp}`）

**记录格式**：每个节点一个不可变文件，名 `%016llx.pcn`。固定小端头（`kHeaderBytes = 168`）：

```
magic(4)="PCD1" version(4)=1 header_bytes(4)=168 endian(4)=0x01020304
flags(4) hash(8) depth(4) blob_bytes(4) state_bytes(4) toks(32*4)
```

随后是 block blob 和可选 state。`write_record` 先写 `.tmp` 再原子 `rename`，记录永不撕裂。
`read_header` 返回 0=成功、1=版本未知（保留、不删）、2=损坏（可删）。`open` 创建目录、扫描
`*.pcn`、删除 class-2、把 class-1 计入 `opaque_bytes_`（永不被逐出）、拒绝 `depth<=0`，用文件 mtime
初始化 `lru` 与 `lru_clock_`，重建 `index_`/`bytes_`。**目录即索引**，无独立索引文件。

**模型指纹**：`pc_disk_init` 对模型身份与所有影响序列化布局的形状做哈希（字面量 `"sycl-infer-pc"`、
磁盘格式 generation、`kv_dtype`、`kBlockSize`、所有 hparams、`rope_sections`、`gguf.map_size`、
`general.architecture/name/file_type`），结果作为 `pc_dir` 下的子目录。两个模型可安全共享一个基目录。

### 5.4 层的调度（`engine_prefix_cache.cpp`）

* `pc_block_blob_bytes()`：每 attention 层 `2*kb`（K+V），int8 时再加 `2*sb`（scale 平面）。
* `pc_serialize_block` / `pc_deserialize_block`：逐层拷贝 K、V、K-scale、V-scale。
* `pc_serialize_state` / `pc_deserialize_state`：搬 `pc_state_floats` 个 float。
* **降级** `pc_demote_node`：无低层返回 false；已有 RAM/disk 记录且数据不更少时只 `touch`（幂等）；
  否则序列化一次 `put` 到 RAM，超预算项再送磁盘；无 RAM 或记录超 RAM 总预算时直接写磁盘。
* **提升** `pc_promote_bytes`：校验几何与当前引擎一致（`blob_bytes`、`state_bytes`），否则返回 -1
  （“布局变了，不可加载”）；分配块、反序列化、必要时取检查点槽、`pc_add_node`、接线状态。
  `pc_promote_ram`/`pc_promote_disk` 分别从 RAM/磁盘取，失败时把记录放回以免丢数据；成功后从源层删除
  （移动语义）。
* `pc_attach_state_from_lower`：resident 节点缺状态时，从 RAM 或磁盘（只读 state）附着。
* `pc_node_to_disk`：绕过 RAM 的 flush 路径。

### 5.5 关机 flush

`pc_flush_to_disk()`：无 `pc_dir` 时无操作。否则 (1) `pcr->drain` 后逐条写磁盘；(2) 把 resident
`pc_nodes_` 按 **LRU 最旧优先** 写磁盘，使 flush 期间的磁盘预算淘汰丢掉的是最久未用的前缀，而不是刚
写入的。`~engine` 在池与检查点仍存活时调用它；`SIGINT`/`SIGTERM` 经 server watchdog 正常停止，从而
走到析构。

---

## 6. 三层预算与 `PF_KV_CAP_MB`

| 层 | 字段 | 环境变量 / flag | 默认 |
|---|---|---|---|
| VRAM | `pc_vram_bytes` | `PF_PC_VRAM_MB`（`--pc-vram-mb`）、`PF_PC_MEM_MB`（别名）、或直接 `PF_PC_STATES` | `PF_PC_STATES=8` |
| RAM | `pc_ram_bytes` | `PF_PC_RAM_MB`（`--pc-ram-mb`） | 512 MB |
| disk | `pc_disk_bytes` | `PF_PC_DISK_MB`（`--pc-disk-mb`） | 1024 MB（0 = 无界） |
| disk 目录 | `pc_dir` | `PF_PC_DIR`（`--pc-dir`） | 空 = 关闭磁盘层 |

* `pc_enabled = !(PF_PREFIX_CACHE==0)`。
* 每节点 VRAM 占用 `per_node = pc_state_floats*4 + pc_block_blob_bytes()`；给 VRAM 预算时
  `states = max(1, vram_mb*1MB / per_node)`；否则默认 8。
* `pc_max_states == 0` 关闭整个缓存。
* **`PF_KV_CAP_MB` 同时限制 KV 池与三层预算之和**。当 `pc_enabled && kv_cap_mb > 0` 且
  `vram+ram+disk > cap` 时，按固定顺序削减：**先磁盘，再 RAM，最后 VRAM**（VRAM 削减后重算
  `states` 并重新量化字节）。磁盘预算被削到 0 时清空 `pc_dir`，**关闭磁盘层**（有意覆盖“0 = 无界”）。
* 未给显式 cap 时，三层配置本身构成总预算。
* 前缀缓存开启时，KV 预留至少 `pc_max_states` 块（每个有状态节点也占一个 KV 块）。

---

## 7. 统计

`pc_print_stats` 输出节点数、有状态节点数、总引用、命中率、复用 token、捕获状态数、驱逐节点/状态数、
检查点尺寸×数量，以及 RAM/磁盘的 store/load/spill 与记录数。相关计数器字段见 `engine.h:330-346`。

---

## 8. 相关环境变量

`PF_PREFIX_CACHE`、`PF_PC_STATES`、`PF_PC_VRAM_MB`、`PF_PC_MEM_MB`、`PF_PC_RAM_MB`、`PF_PC_DIR`、
`PF_PC_DISK_MB`、`PF_PC_DEBUG`、`PF_KV_CAP_MB`。

---

## 9. 进一步阅读

* `reports/prefix_cache.md`：前缀缓存的设计/测量报告。
* `tests/backend/cpu/test_pc_disk.cpp`（磁盘格式/LRU/重开）、`test_pc_ram.cpp`（RAM 层）、
  `tests/backend/cpu/test_pc_cpu.cpp`（主机后端的 paged attention + 磁盘 tier 往返）、
  `tests/backend/gpu/test_pc_gpu.cpp`（磁盘 spill + promote 往返）、
  `tests/backend/gpu/test_pc_ram_gpu.cpp`（VRAM→RAM→VRAM 往返）。
