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
在空闲显存池超过阈值时释放，12 份连续处理不会累积 OOM。实测（同进程、
layout 16 / det 32 / rec 128 / table 64 / table-rec 32 / formula 关）
957 页 **2335.1 s（2.44 s/page，24.6 页/分钟）**，峰值 RSS **7.3 GiB**，
12/12 全部通过，无 OOM、无 device-lost。layout 1 与 layout 16 的
`600348` / `600275` / `601857` 输出 Markdown md5 完全相同。

**layout 单独计时**（600348 的前 N 页、包含 mask 回传）：batch 1 **192 ms/page**、
2 202、4 187、8 182、12 178、16 180。加 batch 只省 ~6%，所以端到端
没有明显收益；真正的大头是 OCR。`600348`（17 页、warm）的阶段拆分：
OCR rec **625 ms/page**、OCR det **613**、layout **306**（首轮包含
batch-17 的 oneDNN primitive 创建；稳态 ~180）、table **296**
（slanet 229 + unet 43 + 两个分类器 24）、PDF 渲染/后处理/Python
**490**。合计 2.38 s/page。

**未达到 1 s/page**：仅 OCR det+rec 就 1.24 s/page，继续加 batch 会在
A770 上 OOR（layout 24 在 12 模型常驻时不稳定：一次 138 页全流程成功，
另一次在同一文档的 layout 之后报 `UR_RESULT_ERROR_OUT_OF_RESOURCES`；
16 连续 957 页稳定）。要接近 1 s/page 需要 OCR 检测/
识别的内核级优化，或降低 OCR 的重复推理，而不是 batch 配置。
