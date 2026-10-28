#ifndef SYCL_INFER_H
#define SYCL_INFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYI_VERSION_MAJOR 0
#define SYI_VERSION_MINOR 1
#define SYI_VERSION_PATCH 0
#define SYI_ABI_VERSION 1

/* Infinite wait for next/wait/shutdown; 0 means a non-blocking poll. */
#define SYI_TIMEOUT_INFINITE 0xFFFFFFFFu

#if defined(_WIN32)
#  if defined(SYI_BUILD)
#    define SYI_API __declspec(dllexport)
#  else
#    define SYI_API __declspec(dllimport)
#  endif
#else
#  define SYI_API __attribute__((visibility("default")))
#endif

typedef struct syi_model syi_model;
typedef struct syi_request syi_request;

typedef struct syi_string {
    const char *data; /* UTF-8; not necessarily writable by the caller */
    size_t size;      /* excludes the trailing NUL */
} syi_string;

typedef enum syi_status {
    SYI_OK = 0,
    SYI_END = 1,
    SYI_AGAIN = 2,
    SYI_INVALID_ARGUMENT = -1,
    SYI_INVALID_STATE = -2,
    SYI_UNSUPPORTED = -3,
    SYI_OUT_OF_MEMORY = -4,
    SYI_BACKEND = -5,
    SYI_MODEL_LOAD = -6,
    SYI_IO = -7,
    SYI_INTERNAL = -8,
    SYI_QUEUE_FULL = -9,
    SYI_SHUTDOWN = -10,
    SYI_CANCELLED = -11
} syi_status;

SYI_API uint32_t syi_abi_version(void);
SYI_API const char *syi_version_string(void);
SYI_API const char *syi_status_string(syi_status status);
SYI_API const char *syi_last_error(void); /* thread-local */
SYI_API void syi_clear_error(void);
SYI_API void syi_free(void *p); /* NULL-safe */

typedef enum syi_device {
    SYI_DEVICE_AUTO = 0,
    SYI_DEVICE_GPU = 1,
    SYI_DEVICE_CPU = 2
} syi_device;

typedef enum syi_kv_type {
    SYI_KV_AUTO = 0,
    SYI_KV_I8 = 1,
    SYI_KV_I4 = 2,
    SYI_KV_F16 = 3,
    SYI_KV_BF16 = 4,
    SYI_KV_F32 = 5
} syi_kv_type;

typedef void (*syi_log_fn)(int32_t level, const char *message, void *user_data);

typedef struct syi_model_options {
    size_t struct_size; /* = sizeof(syi_model_options) */
    uint32_t abi_version;

    const char *model_path; /* required; first shard for a split GGUF */
    const char *layer_map;  /* optional */

    syi_device device;
    int32_t max_seq;  /* 0 = engine default 8192; -1 = model context */
    int32_t max_slots; /* 0 = 16; ceiling 32 */
    int32_t n_blocks; /* 0 = automatic */
    int32_t kv_cap_mb; /* -1 = automatic; 0 = unbounded; >0 = MB */
    syi_kv_type kv_type;
    syi_kv_type kv_type_v; /* AUTO = same as kv_type */

    int32_t cpu_threads; /* 0 = automatic; process-wide pool */
    int32_t n_splits;    /* 0 = automatic */
    int32_t dec_splits;  /* 0 = automatic */

    int32_t batch_window_us; /* 0 = no wait; default 2000 */
    int32_t max_batch_rows;  /* 0 = kMaxB; 1 = strict serial */
    int32_t max_queue_depth; /* 0 = unlimited; default 1024 */
    int64_t max_queue_tokens; /* 0 = unlimited; default 4194304 */
    int32_t request_timeout_ms; /* 0 = unlimited */

    int32_t prefix_cache; /* -1 = default; 0 = off; 1 = on */
    const char *pc_dir;
    int32_t pc_vram_mb;
    int32_t pc_ram_mb;
    int32_t pc_disk_mb;

    int32_t moe_resident;
    int64_t moe_vram_mb;
    int64_t moe_ram_mb;
    int64_t engram_ram_mb;

    int32_t yarn;
    int32_t yarn_orig_ctx;
    float yarn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float yarn_attn_factor;

    int32_t head_split;
    uint64_t flags;
    syi_log_fn log_fn;
    void *log_user_data;

    void *reserved[8];
} syi_model_options;

SYI_API void syi_model_options_init(syi_model_options *options);
SYI_API syi_status syi_model_create(const syi_model_options *options, syi_model **out_model);
SYI_API void syi_model_destroy(syi_model *model);
SYI_API syi_status syi_model_shutdown(syi_model *model, uint32_t timeout_ms);

typedef struct syi_model_info {
    size_t struct_size; /* in/out */
    syi_string model_name;
    syi_string architecture;
    syi_string device_name;
    int32_t n_vocab;
    int32_t n_ctx;
    int32_t n_embd;
    int32_t n_layer;
    int32_t n_head;
    int32_t n_head_kv;
    int32_t head_dim;
    int32_t is_moe;
    int32_t is_split_gguf;
    int32_t max_seq;
    int32_t max_slots;
    syi_kv_type kv_type;
    syi_kv_type kv_type_v;
    syi_device device;
} syi_model_info;

typedef struct syi_model_stats {
    size_t struct_size; /* in/out */
    uint64_t submitted;
    uint64_t admitted;
    uint64_t completed;
    uint64_t cancelled;
    uint64_t failed;
    uint64_t generated_tokens;
    uint64_t prefill_steps;
    uint64_t decode_steps;
    double avg_decode_batch;
    uint32_t max_decode_batch;
    int32_t queued;
    int32_t active;
    int32_t free_slots;
} syi_model_stats;

SYI_API void syi_model_info_init(syi_model_info *info);
SYI_API void syi_model_stats_init(syi_model_stats *stats);
SYI_API syi_status syi_model_get_info(const syi_model *model, syi_model_info *info);
SYI_API syi_status syi_model_get_stats(const syi_model *model, syi_model_stats *stats);

SYI_API syi_status syi_model_tokenize(syi_model *model, const char *text, size_t text_len,
                                     int32_t parse_special, int32_t **out_ids, size_t *out_n);
SYI_API syi_status syi_model_detokenize(syi_model *model, const int32_t *ids, size_t n,
                                       char **out_text, size_t *out_len);

typedef struct syi_chat_message {
    size_t struct_size; /* = sizeof(syi_chat_message) */
    const char *role;
    const char *content;
    const char *name;
    const char *tool_call_id;
    const char *reasoning_content;
} syi_chat_message;

SYI_API void syi_chat_message_init(syi_chat_message *message);

SYI_API syi_status syi_model_render_chat(syi_model *model, const syi_chat_message *messages,
                                         size_t n_messages, int32_t enable_thinking,
                                         const char *tools_json, char **out_text, size_t *out_len);

typedef struct syi_generation_options {
    size_t struct_size; /* = sizeof(syi_generation_options) */
    uint32_t abi_version;

    int32_t max_tokens; /* default 256 */
    int32_t ignore_eos;
    float temperature; /* default 0.7 */
    float top_p;       /* default 0.95 */
    int32_t top_k;     /* default 40; 1 = greedy */
    float min_p;
    float repeat_penalty;
    int32_t repeat_last_n;
    float presence_penalty;
    float frequency_penalty;
    uint64_t seed; /* 0 = random */

    int32_t logprobs;
    int32_t top_logprobs; /* 0..20 */
    int32_t timeout_ms;   /* 0 = model default; -1 = unlimited */

    const int32_t *logit_bias_ids;
    const float *logit_bias_values;
    size_t n_logit_bias;

    const char *const *stop;
    size_t n_stop;

    uint32_t flags;
    void *reserved[8];
} syi_generation_options;

SYI_API void syi_generation_options_init(syi_generation_options *options);

typedef enum syi_finish_reason {
    SYI_FINISH_NONE = 0,
    SYI_FINISH_STOP = 1,
    SYI_FINISH_LENGTH = 2,
    SYI_FINISH_CANCELLED = 3,
    SYI_FINISH_TIMEOUT = 4,
    SYI_FINISH_ERROR = 5
} syi_finish_reason;

typedef enum syi_request_state {
    SYI_REQUEST_CREATED = 0,
    SYI_REQUEST_QUEUED = 1,
    SYI_REQUEST_PREFILL = 2,
    SYI_REQUEST_DECODING = 3,
    SYI_REQUEST_FINISHED = 4,
    SYI_REQUEST_FAILED = 5,
    SYI_REQUEST_CANCELLED = 6
} syi_request_state;

typedef struct syi_logprob {
    int32_t id;
    float logprob;
    syi_string text;
} syi_logprob;

typedef struct syi_token {
    size_t struct_size; /* in/out */
    int32_t id;
    uint32_t flags;
    float logprob;
    syi_string text;
    const syi_logprob *top;
    size_t n_top;
    uint64_t index;
} syi_token;

typedef struct syi_result {
    size_t struct_size; /* in/out */
    syi_string text;
    const int32_t *token_ids;
    size_t n_tokens;
    syi_finish_reason finish_reason;
    uint64_t prompt_tokens;
    uint64_t completion_tokens;
    uint64_t cached_tokens;
    double ttft_ms;
    double total_ms;
    syi_string error;
} syi_result;

SYI_API void syi_token_init(syi_token *token);
SYI_API void syi_result_init(syi_result *result);

SYI_API syi_status syi_request_create(syi_model *model, const syi_generation_options *options,
                                      syi_request **out_request);
SYI_API void syi_request_destroy(syi_request *request);

SYI_API syi_status syi_request_set_text(syi_request *request, const char *text, size_t text_len);
SYI_API syi_status syi_request_set_tokens(syi_request *request, const int32_t *ids, size_t n);
SYI_API syi_status syi_request_set_chat(syi_request *request, const syi_chat_message *messages,
                                        size_t n_messages, int32_t enable_thinking,
                                        const char *tools_json);

SYI_API syi_status syi_request_submit(syi_request *request);
SYI_API syi_status syi_submit_batch(syi_model *model, syi_request *const *requests, size_t n);
SYI_API syi_status syi_wait_batch(syi_request *const *requests, size_t n, uint32_t timeout_ms);

SYI_API syi_status syi_request_get_state(const syi_request *request, syi_request_state *state);
SYI_API syi_status syi_request_next(syi_request *request, syi_token *out, uint32_t timeout_ms);
SYI_API syi_status syi_request_wait(syi_request *request, uint32_t timeout_ms);
SYI_API syi_status syi_request_cancel(syi_request *request);
SYI_API syi_status syi_request_get_result(const syi_request *request, syi_result *out);
SYI_API const char *syi_request_error(const syi_request *request);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SYCL_INFER_H */
