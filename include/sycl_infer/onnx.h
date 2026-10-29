#ifndef SYCL_INFER_ONNX_H
#define SYCL_INFER_ONNX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYI_ONNX_ABI_VERSION 1

#if defined(_WIN32)
#  if defined(SYI_BUILD)
#    define SYI_ONNX_API __declspec(dllexport)
#  else
#    define SYI_ONNX_API __declspec(dllimport)
#  endif
#else
#  define SYI_ONNX_API __attribute__((visibility("default")))
#endif

typedef struct syi_onnx_model syi_onnx_model;

/* 0 = success; negative = failure with syi_onnx_last_error(). */
SYI_ONNX_API int syi_onnx_abi_version(void);
SYI_ONNX_API const char *syi_onnx_last_error(void);

/* Load one ONNX file.  `device` is GPU, GPU.N, CPU, AUTO (case-insensitive).
   `drop_outputs` is a comma-separated list (or NULL); `dnnl` selects the
   oneDNN convolution path. */
SYI_ONNX_API int syi_onnx_create(const char *path, const char *id, const char *device,
                                 const char *drop_outputs, int dnnl,
                                 syi_onnx_model **out_model);
SYI_ONNX_API void syi_onnx_destroy(syi_onnx_model *model);

/* A JSON object: id, path, device, loaded, inputs[], outputs[], character,
   vram_bytes; malloc'd, release with syi_onnx_free. */
SYI_ONNX_API int syi_onnx_info_json(const syi_onnx_model *model, char **out_json,
                                    size_t *out_len);

/* Run the binary wire protocol used by sycl-infer-onnx (src/common/onnx_wire.h).
   The response is malloc'd and must be released with syi_onnx_free; its wire
   status is 0 even when the model run failed (the message is in the reply). */
SYI_ONNX_API int syi_onnx_run_wire(const syi_onnx_model *model, const uint8_t *request,
                                   size_t request_len, uint8_t **out_response,
                                   size_t *out_len);

/* Zero-framing path: the caller passes its numpy buffers directly and gets
   pointers to the model-owned output buffers (valid until the next run).  This
   avoids the request pack/decode and response encode/copy of run_wire, which
   dominate the small-call/short-text stages.  dtype codes match onnxwire:
   f32=0, f16=1, i64=2, i32=3, i8=4, u8=5, bool=6. */
typedef struct syi_onnx_input {
    const char *name;
    int32_t dtype;
    int32_t ndim;
    const int64_t *shape;
    const void *data;
    size_t nbytes;
} syi_onnx_input;

typedef struct syi_onnx_output {
    const char *name;
    int32_t dtype;
    int32_t ndim;
    const int64_t *shape;
    const void *data;
    size_t nbytes;
} syi_onnx_output;

SYI_ONNX_API int syi_onnx_run_direct(const syi_onnx_model *model,
                                     const syi_onnx_input *inputs, size_t n_inputs,
                                     syi_onnx_output *outputs, size_t max_outputs,
                                     size_t *out_count);
/* Reduce an output on the device at readback and return only the small
   result.  `output` is the graph output's tensor name; `mode` is "argmax"
   (a float [.., V] output becomes two outputs, "<name>:argmax" int64 and
   "<name>:max" f32, both [..]), "u8" (the output is cast to a uint8 of the
   same shape) or "none".
   The reduction is exact: the argmax breaks ties to the lowest index, as
   numpy and the host decoder do, and the byte cast truncates toward zero.
   It exists for the two large outputs of the document pipeline - the
   recogniser's [B,T,V] probabilities and the layout detector's instance
   masks - whose transfer, not their arithmetic, dominates the call. */
SYI_ONNX_API int syi_onnx_set_output_mode(syi_onnx_model *model, const char *output,
                                          const char *mode);
/* Release pooled device buffers not used by the current values.  A long
   sequence of documents/shapes otherwise accumulates one pool entry per
   shape; syi_onnx_run_* also calls this automatically above a threshold. */
SYI_ONNX_API int syi_onnx_trim(syi_onnx_model *model);
SYI_ONNX_API void syi_onnx_free(void *p);

#ifdef __cplusplus
}
#endif
#endif
