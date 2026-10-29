# 进程内使用 sycl-infer 的 ONNX executor

`libsycl_infer_onnx` 把 `sycl-infer` 的 ONNX graph/executor 暴露成 C ABI，
宿主进程直接链接/加载它即可运行版面、OCR、公式和表格模型。没有 HTTP
服务、没有子进程，张量只经过一次 wire 编码（同一进程内的字节缓冲）。

## 构建

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-rel2 -DCMAKE_BUILD_TYPE=Release -DSI_ONNX=ON
cmake --build build-rel2 -j2 --target sycl_infer_onnx_c
```

产物：`build-rel2/libsycl_infer_onnx.so.1.0.0` 和 `include/sycl_infer/onnx.h`。

## C ABI

```c
syi_onnx_model *m = NULL;
if (syi_onnx_create("/path/model.onnx", "layout", "GPU",
                    NULL /* drop_outputs CSV */, 1 /* oneDNN */, &m) != 0) {
    fprintf(stderr, "%s\n", syi_onnx_last_error());
}
char *info = NULL; size_t n = 0;
syi_onnx_info_json(m, &info, &n);   /* inputs/outputs/character/device */
syi_onnx_free(info);

/* request/response use src/common/onnx_wire.h verbatim */
uint8_t *resp = NULL; size_t rn = 0;
syi_onnx_run_wire(m, request, request_len, &resp, &rn);
syi_onnx_free(resp);
syi_onnx_destroy(m);
```

输出可以用 `syi_onnx_set_output_mode(m, "<输出名或下标>", "argmax"|"u8"|"none")`
在设备上先归约再回传：`argmax` 把一个 `[..., V]` 的 f32 输出变成
`<name>:argmax`（int64）和 `<name>:max`（f32）两个小输出，`u8` 把输出截断成
uint8。识别模型的 `[B,T,18710]` 概率张量（batch 32 时 115 MB）只需要
CTC 的 argmax/max，layout 的 `[N,200,200]` int32 mask 只需要 0/1 字节，
这两条路径把它们各自降到 16 B/行和 1 B/元素，数值逐位等价（argmax
平局取最小下标，与 numpy 一致）。

`device` 取 `GPU`、`GPU.N`、`CPU` 或 `AUTO`。同一个设备上的模型共享一个
in-order `sycl::queue`（和 ONNX service 一样），因此多模型常驻不会各自
争用设备状态。`drop_outputs` 是逗号分隔的输出名，省掉不读回的 48 MB mask
之类的输出。

## RapidDoc

```python
from rapid_doc import RapidDoc

parser = RapidDoc(
    inference_engine="sycl-infer",  # 或 RAPID_DOC_ENGINE=sycl-infer
    sycl_device="GPU",              # GPU / GPU.0 / CPU
    parse_method="ocr",
    formula_enable=True,
    table_enable=True,
)
result = parser("document.pdf")
```

RapidDoc 通过 `SYCL_INFER_ONNX_LIB` 找库（默认会找相邻的
`sycl-infer/build-rel2`）：

```bash
export SYCL_INFER_ONNX_LIB="$PWD/../sycl-infer/build-rel2/libsycl_infer_onnx.so.1.0.0"
export LD_LIBRARY_PATH="/opt/intel/oneapi/2026.2/lib:/opt/intel/oneapi/compiler/2026.2/lib:/opt/intel/oneapi/dnnl/2026.1/lib:$LD_LIBRARY_PATH"
```

设备默认是 `GPU`；显式 `GPU.0` 可选多卡，`CPU` 用于无 GPU 或显存不足。
每个模型在第一次使用时加载并常驻；`text_cls` 仍走 RapidOCR 自己的本地
session（`Global.use_cls` 默认关闭）。


## A770 / RapidDoc 的安全 batch（实测）

单张 16 GB A770，`error_pdfs` 12 份 957 页全部通过：

| 阶段 | batch |
|---|---:|
| layout | **16**（batch=1/2/4/16/17 已逐页核对；24 在 12 个模型常驻时不稳定，实测 OOR） |
| OCR det | 32 |
| OCR rec | 128 |
| table structure | 64 |
| table rec | 32 |
| formula | 32 |
| OCR rec width ratio | 8 |

**layout batch>1 的旧 bug 已修复**：executor 的 int64 `Transpose`
按列主序（第 0 轴最快）拆输出下标，而 ONNX 张量是行主序。batch=1 时
所有前导维都是 1，这个循环恰好是恒等映射，所以只有 batch>1 才暴露：
layout 模型末尾 `Transpose([B,300], perm=[0,1])` 的 7 列 `orders`
被复制/交错，页面的阅读顺序被挂到错误的检测框上（前 6 列坐标仍然
正确，所以看起来像后处理/掩码问题）。修复在
`src/backend/onnx/ops_exec.cpp` 的 `exec_transpose`，回归测试是
`tests/onnx/test_onnx_batch.cpp`（无需模型，B=1,2,3,8 的 int64/f32
转置）和 `RapidDoc tests/test_sycl_layout_batch.py`（真实 layout 模型，
逐页比较原始 boxes/order/masks）。`im_shape` 保持和 OpenVINO 一致的
单行 `[[h,w]]` 广播；实测单行与 `(N,2)` 平铺的结果相同，不是根因。

`sycl_infer_onnx` 的 `syi_onnx_run_direct` 去掉 wire 编解码，`trim_pool`
在空闲显存池超过阈值时释放，12 份连续处理不会累积 OOM。

实测（单张 Arc A770，同一进程，layout 16 / det 32 / rec 128 / table 64 /
table-rec 32 / formula 关，`error_pdfs` 全部 12 份）：

| | 页数 | 时间 | s/page | 峰值 RSS |
|---|---:|---:|---:|---:|
| 优化前（2026-10-28） | 957 | 2335.1 s | 2.44 | 7306 MiB |
| 第一轮（2026-10-29） | 957 | 1262.4 s | 1.319 | 4270 MiB |
| 第二轮（2026-10-29） | 957 | 1067.2 s | 1.115 | 4325 MiB |
| 第三轮（2026-10-29） | 957 | 961.8 s | 1.005 | 4346 MiB |
| **当前（2026-10-29，第四轮）** | 957 | **952.7 s** | **0.996** | **4272 MiB** |

12/12 全部通过，无 OOM、无 device-lost；`600348` / `600275` / `601857`
以及 `002474` / `300408` 的 Markdown md5 与优化前完全相同。`600348`
（17 页、warm）的阶段拆分（ms/page）：

| 阶段 | 优化前 | 第一轮优化后 |
|---|---:|---:|
| OCR rec | 625 | **154** |
| OCR det | 613 | **241** |
| layout | 306 | **115** |
| table（slanet/unet/分类器） | 296 | **214** |
| PDF 渲染 / 后处理 / Python | 490 | 446 |
| 合计 | **2.38 s/page** | **1.17 s/page** |

变化的来源（细节见 [`../design/17-onnx-service.md`](../design/17-onnx-service.md#the-in-process-pipeline-the-2026-10-29-optimization-pass)）：

* 识别模型的 `[B,T,18710]` 概率张量在设备上直接归约成 argmax/max
  （`syi_onnx_set_output_mode`），并去掉输出的两次 host 拷贝；
* softmax 以前一行一个 work-item（[1536,18710] 只有 1536 个线程，
  25.7 ms），改成一行一个 work-group；
* 1x1、stride 1、group 1 的卷积走 oneDNN matmul（det 有 61 个），
  权重转置一次并缓存；
* 直接卷积的 work-item 下标顺序按 shape 选择：depthwise 用列最快、
  group-1 用通道块最快；
* layout 的 `[N,200,200]` int32 mask 在设备上转成 uint8（4x 少传）；
* `Loop` 体不再每轮重传没变的权重，输出回读合并成一次 queue wait；
* RapidDoc 的 table batch 以前会被**一个 OCR 为空的表格**整体打断，
  退化成逐表 `predict`（94 次 batch-1 解码循环），现在只把无法成批的
  行单跑。

**当前 0.996 s/page（同一构建两次：946.8 s = 0.989 与 952.7 s =
0.996）**，12 份 Markdown md5 与优化前完全一致，峰值 RSS 4272 MiB。
剩余时间分布：OCR rec 169 s、OCR det 152 s、layout 105 s、SLANet 89 s、
UNET 70 s、分类器 6 s，模型合计约 591 s；剩下的约 362 s 是 PDF 渲染、
OCR 后处理、表格 HTML 和 Python。第三轮的两项安全改动（详见
[`../design/17-onnx-service.md`](../design/17-onnx-service.md#the-sparse-retry-only-needs-the-wireless-model)）：

* `Loop` 体的**每个输入都被 pin 住**，因此 body 内部的原地复用重新
  可用：每轮 40 个 `Reshape`/`Squeeze`/`Unsqueeze` 不再做设备拷贝，
  batch-1 SLANet 260 -> 250 ms（`SI_ONNX_LOOP_ALIAS=0` 是 A/B）。
* 稀疏表格重试只跑**无线**（SLANet）模型：有线 UNET 本来就是逐图推理，
  第一轮结果就是它的 batch-1 结果；而且第一轮选中 wired 网格时，重试
  在 3 份表格密集文档的 119 次里**从未**改变最终 HTML（12 次
  wireless-selected 重试中有 2 次改变，仍然重试）。SLANet 的模型时间
  183 -> 100 s，UNET 86 -> 72 s，12 份 md5 全部不变。
  无条件跳过重试则会在 002474/300408 改变 md5，已否决。

第四轮（同日）把执行器的**逐算子 host 路径**本身作为目标：诊断用的
`getenv` 和计时器以前每个算子都执行（body 每轮 157 个算子），现在只在
启用诊断时取；标量比较/布尔运算的结果是 float，原本会走设备内核再被
整数 Cast 阻塞回读，现在小于 8192 元素且输入都在 host 的 elementwise
留在 host，int Cast、小 int Reduce/Scatter 也在 host；`Shape`、`Range`
和全整数 `Concat`/`Slice`/`Tile`/`Expand`/`Gather`/`Split`/`Transpose`
等只读 host 的算子不再上传输入。body 每轮阻塞下载 20 -> 3-6 次，上传
~86 -> ~40 次；SLANet 100.07 -> 88.76 s（-11.3%），UNET 71.60 ->
69.57 s，rec/det/layout 不变，整条流水线 961.8 -> 952.7 s
（`SI_ONNX_OPSTAT=1` 是新的无等待逐算子诊断）。剩下的 body 成本是每轮
约 1.9-3.4 ms host 分发加约 1.4 ms 设备尾；Slice/Range 的边界和循环
条件本来就是 host 值（步进计数器驱动 scatter 行），所以录制 body 仍被
这些 host 值和 oneDNN matmul 阻塞。
