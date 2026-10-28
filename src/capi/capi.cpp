#include "capi_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "chat.h"
#include "common/env.h"
#include "model.h"

namespace si::capi {

static void count_terminal(request_state & r) {
    std::shared_ptr<model_state> m;
    {
        std::lock_guard<std::mutex> lk(r.m);
        if (r.counted) {
            return;
        }
        r.counted = true;
        m = r.model;
    }
    if (!m) {
        return;
    }
    if (r.failed || (r.seq && r.seq->failed)) {
        m->failed.fetch_add(1, std::memory_order_relaxed);
    } else if (r.cancel_requested || r.timed_out || (r.seq && r.seq->timed_out)) {
        m->cancelled.fetch_add(1, std::memory_order_relaxed);
    } else {
        m->completed.fetch_add(1, std::memory_order_relaxed);
    }
    if (r.seq) {
        const uint64_t n = (uint64_t)std::max(0, r.seq->n_generated);
        m->generated_tokens.fetch_add(n, std::memory_order_relaxed);
        m->decode_steps.fetch_add(n, std::memory_order_relaxed);
    }
}

static syi_finish_reason finish_reason_of(const request_state & r, const sequence & s) {
    if (r.timed_out || s.timed_out) {
        return SYI_FINISH_TIMEOUT;
    }
    if (r.failed || s.failed) {
        return SYI_FINISH_ERROR;
    }
    if (r.cancel_requested) {
        return SYI_FINISH_CANCELLED;
    }
    if (s.finish_reason == "error") {
        return SYI_FINISH_ERROR;
    }
    if (s.finish_reason == "timeout") {
        return SYI_FINISH_TIMEOUT;
    }
    if (s.finish_reason == "length") {
        return SYI_FINISH_LENGTH;
    }
    return SYI_FINISH_STOP;
}

static bool is_terminal_state(syi_request_state s) {
    return s == SYI_REQUEST_FINISHED || s == SYI_REQUEST_FAILED || s == SYI_REQUEST_CANCELLED;
}

model_state::~model_state() {
    stop_dispatcher();
    if (sched && !scheduler_stopped) {
        try {
            sched->cancel_all();
            sched->shutdown();
        } catch (...) {
        }
        scheduler_stopped = true;
    }
}

void model_state::start() {
    dispatcher = std::thread([this] { dispatcher_loop(); });
}

void model_state::stop_dispatcher() {
    {
        std::lock_guard<std::mutex> lk(m);
        stopping = true;
        flush_window = true;
    }
    cv.notify_all();
    if (dispatcher.joinable()) {
        dispatcher.join();
    }
}

void model_state::dispatcher_loop() {
    for (;;) {
        std::vector<std::shared_ptr<request_state>> batch;
        int64_t batch_tokens = 0;
        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&] { return stopping || !pending.empty(); });
            if (pending.empty()) {
                if (stopping) {
                    break;
                }
                continue;
            }
            if (!stopping && batch_window_us > 0) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(batch_window_us);
                cv.wait_until(lk, deadline, [&] { return stopping || flush_window; });
            }
            flush_window = false;
            while (!pending.empty()) {
                auto r = pending.front();
                pending.pop_front();
                batch_tokens += (int64_t)r->prompt_tokens.size();
                batch.push_back(std::move(r));
            }
            queued_tokens = std::max<int64_t>(0, queued_tokens - batch_tokens);
            if (stopping) {
                // cancel outside the model lock: submit holds the request lock
                // and then the model lock, so taking them in the other order
                // here would deadlock.
            }
        }
        if (stopping) {
            for (auto & r : batch) {
                std::lock_guard<std::mutex> rl(r->m);
                r->cancel_requested = true;
                r->state = SYI_REQUEST_CANCELLED;
                r->cv.notify_all();
            }
            break;
        }
        for (auto & r : batch) {
            submit_one(r);
        }
    }
    {
        std::lock_guard<std::mutex> lk(m);
        dispatcher_done = true;
    }
    cv.notify_all();
}

syi_status model_state::submit_one(const std::shared_ptr<request_state> & r) {
    {
        std::lock_guard<std::mutex> lk(r->m);
        if (r->cancel_requested) {
            return SYI_OK;
        }
    }
    std::shared_ptr<sequence> seq;
    try {
        seq = sched->submit(r->prompt_tokens, r->gp, r->stops, r->timeout_ms, true);
    } catch (const std::bad_alloc &) {
        r->set_error("out of memory while submitting the request");
        {
            std::lock_guard<std::mutex> lk(r->m);
            r->state = SYI_REQUEST_FAILED;
        }
        r->cv.notify_all();
        count_terminal(*r);
        return SYI_OUT_OF_MEMORY;
    } catch (const std::exception & ex) {
        r->set_error(ex.what());
        {
            std::lock_guard<std::mutex> lk(r->m);
            r->state = SYI_REQUEST_FAILED;
        }
        r->cv.notify_all();
        count_terminal(*r);
        return SYI_INTERNAL;
    }
    {
        std::lock_guard<std::mutex> lk(r->m);
        r->seq = seq;
        r->state = SYI_REQUEST_QUEUED;
    }
    r->cv.notify_all();
    admitted.fetch_add(1, std::memory_order_relaxed);
    prefill_steps.fetch_add(1, std::memory_order_relaxed);
    return SYI_OK;
}

syi_status model_state::shutdown(uint32_t timeout_ms) {
    std::deque<std::shared_ptr<request_state>> cancel;
    {
        std::lock_guard<std::mutex> lk(m);
        accepting = false;
        stopping = true;
        flush_window = true;
        cancel.swap(pending);
        queued_tokens = 0;
    }
    for (auto & r : cancel) {
        std::lock_guard<std::mutex> rl(r->m);
        r->cancel_requested = true;
        r->state = SYI_REQUEST_CANCELLED;
        r->cv.notify_all();
    }
    cv.notify_all();
    if (dispatcher.joinable()) {
        dispatcher.join();
    }
    if (!sched) {
        return SYI_OK;
    }
    sched->cancel_all();
    if (scheduler_stopped) {
        return SYI_OK;
    }
    const bool infinite = timeout_ms == SYI_TIMEOUT_INFINITE;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(infinite ? 0 : timeout_ms);
    if (!infinite && timeout_ms == 0 && sched->inflight() > 0) {
        return SYI_AGAIN;
    }
    while (!infinite && sched->inflight() > 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return SYI_AGAIN;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    try {
        sched->shutdown();
    } catch (...) {
        return SYI_INTERNAL;
    }
    scheduler_stopped = true;
    return SYI_OK;
}

} // namespace si::capi

using namespace si;
using namespace si::capi;

namespace {

int kv_code(syi_kv_type t) {
    switch (t) {
    case SYI_KV_AUTO: return -1;
    case SYI_KV_I8: return (int)kv_dtype_t::i8;
    case SYI_KV_I4: return (int)kv_dtype_t::i4;
    case SYI_KV_F16: return (int)kv_dtype_t::f16;
    case SYI_KV_BF16: return (int)kv_dtype_t::bf16;
    case SYI_KV_F32: return (int)kv_dtype_t::f32;
    }
    return -1;
}

syi_kv_type kv_public(kv_dtype_t t) {
    switch (t) {
    case kv_dtype_t::i8: return SYI_KV_I8;
    case kv_dtype_t::i4: return SYI_KV_I4;
    case kv_dtype_t::f16: return SYI_KV_F16;
    case kv_dtype_t::bf16: return SYI_KV_BF16;
    case kv_dtype_t::f32: return SYI_KV_F32;
    }
    return SYI_KV_AUTO;
}

bool valid_message(const syi_chat_message & m) {
    return m.struct_size >= sizeof(syi_chat_message);
}

std::string copy_cstr(const char *s) {
    return s ? std::string(s) : std::string();
}

} // namespace

extern "C" {

syi_status syi_model_create(const syi_model_options *o, syi_model **out_model) {
    if (!out_model) {
        return fail(SYI_INVALID_ARGUMENT, "out_model is null");
    }
    *out_model = nullptr;
    if (!o || !o->model_path || !*o->model_path) {
        return fail(SYI_INVALID_ARGUMENT, "model_path is required");
    }
    if (o->struct_size < sizeof(syi_model_options)) {
        return fail(SYI_INVALID_ARGUMENT, "syi_model_options.struct_size is too small");
    }
    if (o->abi_version != 0 && o->abi_version != SYI_ABI_VERSION) {
        return fail(SYI_UNSUPPORTED, "unsupported syi_model_options.abi_version");
    }
    int kk = kv_code(o->kv_type);
    int vv = kv_code(o->kv_type_v);
    // AUTO means "the process/env default", resolved here so a zeroed options
    // struct (or syi_model_options_init, which leaves both AUTO) is usable.
    if (kk < 0) {
        kk = (int)kv_k_dtype();
    }
    if (vv < 0) {
        vv = o->kv_type == SYI_KV_AUTO ? kk : (int)kv_v_dtype();
    }
    if (kk < (int)kv_dtype_t::f32 || kk > (int)kv_dtype_t::i4 || vv < (int)kv_dtype_t::f32 ||
        vv > (int)kv_dtype_t::i4 || !kv_dtype_mix_ok((kv_dtype_t)kk, (kv_dtype_t)vv)) {
        return fail(SYI_INVALID_ARGUMENT, "unsupported KV type pair");
    }
    const int slots = o->max_slots == 0 ? 16 : o->max_slots;
    if (slots < 1 || slots > kMaxB) {
        return fail(SYI_INVALID_ARGUMENT, "max_slots must be in [1, kMaxB]");
    }

    engine_config cfg;
    cfg.model_path = o->model_path;
    cfg.max_seq = o->max_seq;
    if (cfg.max_seq == 0) {
        cfg.max_seq = 8192;
    } else if (cfg.max_seq < 0) {
        cfg.max_seq = model_context_length(cfg.model_path);
        if (cfg.max_seq <= 0) {
            cfg.max_seq = 8192;
        }
    }
    cfg.max_slots = slots;
    cfg.n_blocks = o->n_blocks > 0 ? o->n_blocks : 512;
    cfg.kv_cap_mb = o->kv_cap_mb;
    cfg.kv_k = kk;
    cfg.kv_v = vv;
    cfg.cpu_threads = o->cpu_threads;
    cfg.n_splits = o->n_splits > 0 ? o->n_splits : 16;
    cfg.dec_splits = o->dec_splits;
    cfg.prefix_cache = o->prefix_cache;
    cfg.pc_dir = copy_cstr(o->pc_dir);
    cfg.pc_vram_mb = o->pc_vram_mb;
    cfg.pc_ram_mb = o->pc_ram_mb;
    cfg.pc_disk_mb = o->pc_disk_mb;
    cfg.device = o->device == SYI_DEVICE_GPU ? 0 : (o->device == SYI_DEVICE_CPU ? 1 : -1);
    cfg.layer_map = copy_cstr(o->layer_map);
    cfg.moe_resident = o->moe_resident;
    cfg.moe_vram_mb = (size_t)std::max<int64_t>(0, o->moe_vram_mb);
    cfg.moe_ram_mb = (size_t)std::max<int64_t>(0, o->moe_ram_mb);
    cfg.engram_ram_mb = o->engram_ram_mb < 0 ? engine_config::kEngramRamAuto : (size_t)o->engram_ram_mb;
    cfg.yarn = o->yarn != 0;
    cfg.yarn_orig_ctx = o->yarn_orig_ctx;
    cfg.yarn_factor = o->yarn_factor;
    cfg.yarn_beta_fast = o->yarn_beta_fast;
    cfg.yarn_beta_slow = o->yarn_beta_slow;
    cfg.yarn_attn_factor = o->yarn_attn_factor;
    cfg.head_split = o->head_split != 0;

    auto st = std::make_shared<model_state>();
    try {
        st->e = std::make_unique<engine>(cfg);
        st->max_slots = st->e->max_slots;
        st->batch_window_us = std::max(0, o->batch_window_us);
        st->max_batch_rows = o->max_batch_rows;
        st->max_queue_depth = o->max_queue_depth;
        st->max_queue_tokens = o->max_queue_tokens;
        st->request_timeout_ms = std::max(0, o->request_timeout_ms);
        st->log_fn = o->log_fn;
        st->log_user = o->log_user_data;
        const std::string *gn = st->e->m.gguf.get_str("general.name");
        st->model_name = (gn && !gn->empty()) ? *gn : cfg.model_path;
        st->architecture = st->e->mp_ ? st->e->mp_->key : "";
        st->device_name = st->e->cpu_mode ? "cpu" : (st->e->multi_dev ? "gpu:multi" : "gpu");
        st->sched = std::make_unique<scheduler>(*st->e);
        st->sched->decode_row_cap =
            o->max_batch_rows > 0 ? std::min<int>(o->max_batch_rows, kMaxB) : 0;
        st->sched->start();
        st->start();
    } catch (const std::bad_alloc &) {
        return fail(SYI_OUT_OF_MEMORY, "out of memory while creating the model");
    } catch (const std::exception & ex) {
        return fail(SYI_MODEL_LOAD, ex.what());
    } catch (...) {
        return fail(SYI_INTERNAL, "unknown model creation failure");
    }

    auto *h = new (std::nothrow) syi_model();
    if (!h) {
        return fail(SYI_OUT_OF_MEMORY, "out of memory while allocating the model handle");
    }
    h->st = std::move(st);
    *out_model = h;
    return SYI_OK;
}

void syi_model_destroy(syi_model *model) {
    if (!model) {
        return;
    }
    auto st = model->st;
    model->st.reset();
    if (st) {
        st->shutdown(SYI_TIMEOUT_INFINITE);
    }
    delete model;
}

syi_status syi_model_shutdown(syi_model *model, uint32_t timeout_ms) {
    if (!model || !model->st) {
        return SYI_INVALID_ARGUMENT;
    }
    return model->st->shutdown(timeout_ms);
}

syi_status syi_model_get_info(const syi_model *model, syi_model_info *info) {
    if (!model || !model->st || !model->st->e || !info) {
        return fail(SYI_INVALID_ARGUMENT, "model/info is null");
    }
    if (info->struct_size == 0) {
        info->struct_size = sizeof(*info);
    }
    if (info->struct_size < sizeof(syi_model_info)) {
        return fail(SYI_INVALID_ARGUMENT, "syi_model_info.struct_size is too small");
    }
    const auto & st = *model->st;
    const auto & hp = st.e->m.hp;
    auto str = [](const std::string & s) {
        syi_string r{s.c_str(), s.size()};
        return r;
    };
    info->model_name = str(st.model_name);
    info->architecture = str(st.architecture);
    info->device_name = str(st.device_name);
    info->n_vocab = hp.n_vocab;
    info->n_ctx = hp.n_ctx;
    info->n_embd = hp.n_embd;
    info->n_layer = hp.n_layer;
    info->n_head = hp.n_head;
    info->n_head_kv = hp.n_head_kv;
    info->head_dim = hp.head_dim;
    info->is_moe = hp.is_moe();
    info->is_split_gguf = st.e->m.gguf.is_split();
    info->max_seq = st.e->max_seq;
    info->max_slots = st.e->max_slots;
    info->kv_type = kv_public(st.e->kv_k_);
    info->kv_type_v = kv_public(st.e->kv_v_);
    info->device = st.e->cpu_mode ? SYI_DEVICE_CPU : (st.e->device_req == 1 ? SYI_DEVICE_CPU : SYI_DEVICE_GPU);
    return SYI_OK;
}

syi_status syi_model_get_stats(const syi_model *model, syi_model_stats *stats) {
    if (!model || !model->st || !stats) {
        return fail(SYI_INVALID_ARGUMENT, "model/stats is null");
    }
    if (stats->struct_size == 0) {
        stats->struct_size = sizeof(*stats);
    }
    if (stats->struct_size < sizeof(syi_model_stats)) {
        return fail(SYI_INVALID_ARGUMENT, "syi_model_stats.struct_size is too small");
    }
    auto & st = *model->st;
    int pending = 0;
    {
        std::lock_guard<std::mutex> lk(st.m);
        pending = (int)st.pending.size();
    }
    const int waiting = st.sched ? st.sched->waiting_count() : 0;
    const int active = st.sched ? st.sched->inflight() : 0;
    stats->submitted = st.submitted.load(std::memory_order_relaxed);
    stats->admitted = st.admitted.load(std::memory_order_relaxed);
    stats->completed = st.completed.load(std::memory_order_relaxed);
    stats->cancelled = st.cancelled.load(std::memory_order_relaxed);
    stats->failed = st.failed.load(std::memory_order_relaxed);
    stats->generated_tokens = st.generated_tokens.load(std::memory_order_relaxed);
    stats->prefill_steps = st.prefill_steps.load(std::memory_order_relaxed);
    stats->decode_steps = st.decode_steps.load(std::memory_order_relaxed);
    stats->avg_decode_batch = 0.0;
    stats->max_decode_batch = 0;
    stats->queued = pending + waiting;
    stats->active = active;
    stats->free_slots = std::max(0, st.max_slots - active);
    return SYI_OK;
}

syi_status syi_model_tokenize(syi_model *model, const char *text, size_t text_len, int32_t parse_special,
                              int32_t **out_ids, size_t *out_n) {
    if (!model || !model->st || !model->st->e || !out_ids || !out_n || (!text && text_len > 0)) {
        return fail(SYI_INVALID_ARGUMENT, "invalid tokenize arguments");
    }
    try {
        std::string s(text ? text : "", text_len);
        const std::vector<int> ids = model->st->e->tk.encode(s, parse_special != 0);
        *out_n = ids.size();
        if (ids.empty()) {
            *out_ids = nullptr;
            return SYI_OK;
        }
        auto *p = (int32_t *)std::malloc(ids.size() * sizeof(int32_t));
        if (!p) {
            return fail(SYI_OUT_OF_MEMORY, "out of memory for token ids");
        }
        std::memcpy(p, ids.data(), ids.size() * sizeof(int32_t));
        *out_ids = p;
        return SYI_OK;
    } catch (const std::exception & ex) {
        return fail(SYI_INTERNAL, ex.what());
    }
}

syi_status syi_model_detokenize(syi_model *model, const int32_t *ids, size_t n, char **out_text, size_t *out_len) {
    if (!model || !model->st || !model->st->e || !out_text || !out_len || (!ids && n > 0)) {
        return fail(SYI_INVALID_ARGUMENT, "invalid detokenize arguments");
    }
    try {
        std::vector<int> v(ids, ids + n);
        const std::string s = model->st->e->tk.decode(v);
        auto *p = (char *)std::malloc(s.size() + 1);
        if (!p) {
            return fail(SYI_OUT_OF_MEMORY, "out of memory for text");
        }
        std::memcpy(p, s.data(), s.size());
        p[s.size()] = '\0';
        *out_text = p;
        *out_len = s.size();
        return SYI_OK;
    } catch (const std::exception & ex) {
        return fail(SYI_INTERNAL, ex.what());
    }
}

syi_status syi_model_render_chat(syi_model *model, const syi_chat_message *messages, size_t n_messages,
                                 int32_t enable_thinking, const char *tools_json, char **out_text,
                                 size_t *out_len) {
    if (!model || !model->st || !model->st->e || !out_text || !out_len ||
        (!messages && n_messages > 0)) {
        return fail(SYI_INVALID_ARGUMENT, "invalid render_chat arguments");
    }
    try {
        std::vector<chat_msg> msgs;
        msgs.reserve(n_messages);
        for (size_t i = 0; i < n_messages; ++i) {
            if (!valid_message(messages[i])) {
                return fail(SYI_INVALID_ARGUMENT, "syi_chat_message.struct_size is too small");
            }
            chat_msg m;
            m.role = copy_cstr(messages[i].role);
            m.content = copy_cstr(messages[i].content);
            m.name = copy_cstr(messages[i].name);
            m.tool_call_id = copy_cstr(messages[i].tool_call_id);
            m.reasoning_content = copy_cstr(messages[i].reasoning_content);
            msgs.push_back(std::move(m));
        }
        const std::string text =
            render_chat(model->st->e->m.chat_template, msgs, true, enable_thinking != 0,
                        tools_json ? tools_json : "");
        auto *p = (char *)std::malloc(text.size() + 1);
        if (!p) {
            return fail(SYI_OUT_OF_MEMORY, "out of memory for rendered chat");
        }
        std::memcpy(p, text.data(), text.size());
        p[text.size()] = '\0';
        *out_text = p;
        *out_len = text.size();
        return SYI_OK;
    } catch (const std::exception & ex) {
        return fail(SYI_INTERNAL, ex.what());
    }
}

syi_status syi_request_create(syi_model *model, const syi_generation_options *o, syi_request **out_request) {
    if (!out_request) {
        return fail(SYI_INVALID_ARGUMENT, "out_request is null");
    }
    *out_request = nullptr;
    if (!model || !model->st || !model->st->e) {
        return fail(SYI_INVALID_ARGUMENT, "model is null");
    }
    if (o && o->struct_size < sizeof(syi_generation_options)) {
        return fail(SYI_INVALID_ARGUMENT, "syi_generation_options.struct_size is too small");
    }
    auto r = std::make_shared<request_state>();
    r->model = model->st;
    if (o) {
        if (o->abi_version != 0 && o->abi_version != SYI_ABI_VERSION) {
            return fail(SYI_UNSUPPORTED, "unsupported syi_generation_options.abi_version");
        }
        if (o->max_tokens < 1 || o->top_k < 0 || o->repeat_last_n < 0 || o->top_logprobs < 0 ||
            o->top_logprobs > 20 || o->n_stop > 32) {
            return fail(SYI_INVALID_ARGUMENT, "invalid generation option");
        }
        r->gp.max_tokens = o->max_tokens;
        r->gp.ignore_eos = o->ignore_eos != 0;
        r->gp.temperature = o->temperature;
        r->gp.top_p = o->top_p;
        r->gp.top_k = o->top_k;
        r->gp.min_p = o->min_p;
        r->gp.repeat_penalty = o->repeat_penalty;
        r->gp.repeat_last_n = o->repeat_last_n;
        r->gp.presence_penalty = o->presence_penalty;
        r->gp.frequency_penalty = o->frequency_penalty;
        r->gp.seed = o->seed;
        r->gp.logprobs = o->logprobs != 0;
        r->gp.top_logprobs = o->top_logprobs;
        if (o->n_stop > 0 && !o->stop) {
            return fail(SYI_INVALID_ARGUMENT, "stop array is null");
        }
        for (size_t i = 0; i < o->n_stop; ++i) {
            if (!o->stop[i]) {
                return fail(SYI_INVALID_ARGUMENT, "stop string is null");
            }
            const std::string s(o->stop[i]);
            if (s.size() > 256) {
                return fail(SYI_INVALID_ARGUMENT, "stop string is longer than 256 bytes");
            }
            r->stops.push_back(s);
        }
        if (o->n_logit_bias > 0) {
            if (!o->logit_bias_ids || !o->logit_bias_values) {
                return fail(SYI_INVALID_ARGUMENT, "logit_bias arrays are null");
            }
            for (size_t i = 0; i < o->n_logit_bias; ++i) {
                r->gp.logit_bias[o->logit_bias_ids[i]] = o->logit_bias_values[i];
            }
        }
    }
    const int per_request = o ? o->timeout_ms : 0;
    if (per_request < 0) {
        r->timeout_ms = 0;
    } else if (per_request > 0) {
        r->timeout_ms = per_request;
    } else {
        r->timeout_ms = model->st->request_timeout_ms;
    }
    auto *h = new (std::nothrow) syi_request();
    if (!h) {
        return fail(SYI_OUT_OF_MEMORY, "out of memory for the request handle");
    }
    h->st = std::move(r);
    *out_request = h;
    return SYI_OK;
}

void syi_request_destroy(syi_request *request) {
    if (!request) {
        return;
    }
    auto st = request->st;
    if (st) {
        bool terminal = false;
        {
            std::lock_guard<std::mutex> lk(st->m);
            terminal = st->terminal();
        }
        if (!terminal) {
            syi_request_cancel(request);
        }
        // Wait for a running sequence to retire.  The state is shared with the
        // dispatcher/scheduler, so dropping the handle is safe even on timeout.
        if (!terminal) {
            std::shared_ptr<sequence> seq;
            {
                std::lock_guard<std::mutex> lk(st->m);
                seq = st->seq;
            }
            if (seq) {
                std::unique_lock<std::mutex> lk(seq->m);
                seq->cv.wait_for(lk, std::chrono::seconds(5), [&] { return seq->finished.load(); });
            }
        }
        count_terminal(*st);
    }
    request->st.reset();
    delete request;
}

syi_status syi_request_set_text(syi_request *request, const char *text, size_t text_len) {
    if (!request || !request->st || (!text && text_len > 0)) {
        return fail(SYI_INVALID_ARGUMENT, "invalid request/text");
    }
    std::lock_guard<std::mutex> lk(request->st->m);
    if (request->st->submitted || request->st->state != SYI_REQUEST_CREATED) {
        return fail(SYI_INVALID_STATE, "request is already submitted");
    }
    request->st->prompt_text.assign(text ? text : "", text_len);
    request->st->prompt_tokens.clear();
    request->st->has_tokens = false;
    return SYI_OK;
}

syi_status syi_request_set_tokens(syi_request *request, const int32_t *ids, size_t n) {
    if (!request || !request->st || (!ids && n > 0)) {
        return fail(SYI_INVALID_ARGUMENT, "invalid request/tokens");
    }
    std::lock_guard<std::mutex> lk(request->st->m);
    if (request->st->submitted || request->st->state != SYI_REQUEST_CREATED) {
        return fail(SYI_INVALID_STATE, "request is already submitted");
    }
    request->st->prompt_tokens.assign(ids, ids + n);
    request->st->prompt_text.clear();
    request->st->has_tokens = true;
    return SYI_OK;
}

syi_status syi_request_set_chat(syi_request *request, const syi_chat_message *messages, size_t n_messages,
                                int32_t enable_thinking, const char *tools_json) {
    if (!request || !request->st || !request->st->model || !request->st->model->e ||
        (!messages && n_messages > 0)) {
        return fail(SYI_INVALID_ARGUMENT, "invalid request/chat");
    }
    try {
        std::vector<chat_msg> msgs;
        msgs.reserve(n_messages);
        for (size_t i = 0; i < n_messages; ++i) {
            if (!valid_message(messages[i])) {
                return fail(SYI_INVALID_ARGUMENT, "syi_chat_message.struct_size is too small");
            }
            chat_msg m;
            m.role = copy_cstr(messages[i].role);
            m.content = copy_cstr(messages[i].content);
            m.name = copy_cstr(messages[i].name);
            m.tool_call_id = copy_cstr(messages[i].tool_call_id);
            m.reasoning_content = copy_cstr(messages[i].reasoning_content);
            msgs.push_back(std::move(m));
        }
        const std::string text = render_chat(request->st->model->e->m.chat_template, msgs, true,
                                             enable_thinking != 0, tools_json ? tools_json : "");
        const std::vector<int> ids = request->st->model->e->tk.encode(text, true);
        std::lock_guard<std::mutex> lk(request->st->m);
        if (request->st->submitted || request->st->state != SYI_REQUEST_CREATED) {
            return fail(SYI_INVALID_STATE, "request is already submitted");
        }
        request->st->prompt_text = text;
        request->st->prompt_tokens = ids;
        request->st->has_tokens = true;
        return SYI_OK;
    } catch (const std::exception & ex) {
        return fail(SYI_INTERNAL, ex.what());
    }
}

syi_status syi_request_submit(syi_request *request) {
    if (!request || !request->st || !request->st->model || !request->st->model->e ||
        !request->st->model->sched) {
        return fail(SYI_INVALID_ARGUMENT, "invalid request");
    }
    auto & r = *request->st;
    auto & m = *r.model;
    std::vector<int> tokens;
    {
        std::lock_guard<std::mutex> lk(r.m);
        if (r.submitted || r.state != SYI_REQUEST_CREATED) {
            return fail(SYI_INVALID_STATE, "request is already submitted");
        }
        if (r.has_tokens) {
            tokens = r.prompt_tokens;
        } else {
            try {
                tokens = m.e->tk.encode(r.prompt_text, true);
            } catch (const std::exception & ex) {
                return fail(SYI_INTERNAL, ex.what());
            }
        }
        if (tokens.empty()) {
            return fail(SYI_INVALID_ARGUMENT, "prompt is empty");
        }
        if ((int)tokens.size() > m.e->max_seq) {
            return fail(SYI_INVALID_ARGUMENT, "prompt is longer than max_seq");
        }
        std::lock_guard<std::mutex> ml(m.m);
        if (!m.accepting) {
            return fail(SYI_SHUTDOWN, "model is shutting down");
        }
        const int queued = (int)m.pending.size() + (m.sched ? m.sched->waiting_count() + m.sched->inflight() : 0);
        if (m.max_queue_depth > 0 && queued >= m.max_queue_depth) {
            return fail(SYI_QUEUE_FULL, "queue depth limit reached");
        }
        if (m.max_queue_tokens > 0 && m.queued_tokens + (int64_t)tokens.size() > m.max_queue_tokens) {
            return fail(SYI_QUEUE_FULL, "queue token limit reached");
        }
        r.prompt_tokens = tokens;
        r.submitted = true;
        r.state = SYI_REQUEST_QUEUED;
        m.pending.push_back(r.shared_from_this());
        m.queued_tokens += (int64_t)tokens.size();
        m.submitted.fetch_add(1, std::memory_order_relaxed);
    }
    m.cv.notify_all();
    return SYI_OK;
}

syi_status syi_submit_batch(syi_model *model, syi_request *const *requests, size_t n) {
    if (!model || !model->st || !requests || n == 0) {
        return fail(SYI_INVALID_ARGUMENT, "invalid batch");
    }
    for (size_t i = 0; i < n; ++i) {
        if (!requests[i] || !requests[i]->st || requests[i]->st->model.get() != model->st.get()) {
            return fail(SYI_INVALID_ARGUMENT, "batch requests must belong to the same model");
        }
        std::lock_guard<std::mutex> lk(requests[i]->st->m);
        if (requests[i]->st->submitted || requests[i]->st->state != SYI_REQUEST_CREATED) {
            return fail(SYI_INVALID_STATE, "a batch request is already submitted");
        }
    }
    auto & m = *model->st;
    // The window closes immediately: this call is the explicit "these arrive
    // together" form.  Submit in order; queue accounting is done per request.
    {
        std::lock_guard<std::mutex> lk(m.m);
        m.flush_window = true;
    }
    for (size_t i = 0; i < n; ++i) {
        const syi_status st = syi_request_submit(requests[i]);
        if (st != SYI_OK) {
            return st;
        }
    }
    m.cv.notify_all();
    return SYI_OK;
}

syi_status syi_wait_batch(syi_request *const *requests, size_t n, uint32_t timeout_ms) {
    if (!requests || n == 0) {
        return fail(SYI_INVALID_ARGUMENT, "invalid batch");
    }
    const bool infinite = timeout_ms == SYI_TIMEOUT_INFINITE;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(infinite ? 0 : timeout_ms);
    for (size_t i = 0; i < n; ++i) {
        uint32_t remaining = SYI_TIMEOUT_INFINITE;
        if (!infinite) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return SYI_AGAIN;
            }
            remaining = (uint32_t)std::max<int64_t>(
                0, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
        }
        const syi_status st = syi_request_wait(requests[i], remaining);
        if (st != SYI_OK) {
            return st;
        }
    }
    return SYI_OK;
}

syi_status syi_request_get_state(const syi_request *request, syi_request_state *state) {
    if (!request || !request->st || !state) {
        return fail(SYI_INVALID_ARGUMENT, "invalid request/state");
    }
    const auto & r = *request->st;
    std::lock_guard<std::mutex> lk(r.m);
    if (!r.seq) {
        *state = r.state;
        return SYI_OK;
    }
    if (r.seq->finished.load(std::memory_order_acquire)) {
        if (r.timed_out || r.seq->timed_out) {
            *state = SYI_REQUEST_CANCELLED;
        } else if (r.failed || r.seq->failed) {
            *state = SYI_REQUEST_FAILED;
        } else if (r.cancel_requested) {
            *state = SYI_REQUEST_CANCELLED;
        } else {
            *state = SYI_REQUEST_FINISHED;
        }
    } else if (!r.seq->admitted) {
        *state = SYI_REQUEST_QUEUED;
    } else if (r.seq->prompt_pos < (int)r.seq->prompt.size()) {
        *state = SYI_REQUEST_PREFILL;
    } else {
        *state = SYI_REQUEST_DECODING;
    }
    return SYI_OK;
}

syi_status syi_request_next(syi_request *request, syi_token *out, uint32_t timeout_ms) {
    if (!request || !request->st || !out) {
        return fail(SYI_INVALID_ARGUMENT, "invalid request/token");
    }
    auto & r = *request->st;
    std::shared_ptr<sequence> seq;
    {
        std::unique_lock<std::mutex> lk(r.m);
        if (!r.submitted) {
            return fail(SYI_INVALID_STATE, "request has not been submitted");
        }
        if (!r.seq) {
            if (is_terminal_state(r.state)) {
                return SYI_END;
            }
            const bool infinite = timeout_ms == SYI_TIMEOUT_INFINITE;
            if (timeout_ms == 0) {
                r.cv.wait_for(lk, std::chrono::milliseconds(0), [&] { return r.seq != nullptr; });
            } else if (infinite) {
                r.cv.wait(lk, [&] { return r.seq != nullptr; });
            } else {
                r.cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return r.seq != nullptr; });
            }
            if (!r.seq) {
                return is_terminal_state(r.state) ? SYI_END : SYI_AGAIN;
            }
        }
        seq = r.seq;
    }
    std::unique_lock<std::mutex> sl(seq->m);
    auto ready = [&] { return seq->read_pos < seq->tokens.size() || seq->finished.load(); };
    const bool infinite = timeout_ms == SYI_TIMEOUT_INFINITE;
    if (timeout_ms == 0) {
        seq->cv.wait_for(sl, std::chrono::milliseconds(0), ready);
    } else if (infinite) {
        seq->cv.wait(sl, ready);
    } else {
        seq->cv.wait_for(sl, std::chrono::milliseconds(timeout_ms), ready);
    }
    if (seq->read_pos >= seq->tokens.size()) {
        if (seq->finished.load(std::memory_order_acquire)) {
            sl.unlock();
            count_terminal(r);
            return SYI_END;
        }
        return SYI_AGAIN;
    }
    sequence::token_out t = std::move(seq->tokens[seq->read_pos++]);
    sl.unlock();

    std::lock_guard<std::mutex> lk(r.m);
    r.last_text = std::move(t.text);
    r.last_top.clear();
    r.last_top_text.clear();
    r.last_top_text.reserve(t.top.size());
    for (const auto & p : t.top) {
        r.last_top_text.push_back(r.model && r.model->e ? r.model->e->tk.token_piece(p.first) : std::string());
    }
    r.last_top.reserve(t.top.size());
    for (size_t i = 0; i < t.top.size(); ++i) {
        syi_logprob lp{};
        lp.id = t.top[i].first;
        lp.logprob = t.top[i].second;
        lp.text.data = r.last_top_text[i].c_str();
        lp.text.size = r.last_top_text[i].size();
        r.last_top.push_back(lp);
    }
    if (out->struct_size == 0) {
        out->struct_size = sizeof(*out);
    }
    out->id = t.id;
    out->flags = 0;
    out->logprob = t.logprob;
    out->text.data = r.last_text.c_str();
    out->text.size = r.last_text.size();
    out->top = r.last_top.empty() ? nullptr : r.last_top.data();
    out->n_top = r.last_top.size();
    out->index = r.next_index++;
    return SYI_OK;
}

syi_status syi_request_wait(syi_request *request, uint32_t timeout_ms) {
    if (!request || !request->st) {
        return fail(SYI_INVALID_ARGUMENT, "invalid request");
    }
    auto & r = *request->st;
    std::shared_ptr<sequence> seq;
    {
        std::lock_guard<std::mutex> lk(r.m);
        if (!r.submitted) {
            return fail(SYI_INVALID_STATE, "request has not been submitted");
        }
        seq = r.seq;
    }
    if (!seq) {
        // The dispatcher has not handed it to the scheduler yet.  Wait on the
        // request condition (it notifies when seq is installed).
        std::unique_lock<std::mutex> lk(r.m);
        const bool infinite = timeout_ms == SYI_TIMEOUT_INFINITE;
        auto ready = [&] { return r.seq != nullptr || is_terminal_state(r.state); };
        if (timeout_ms == 0) {
            r.cv.wait_for(lk, std::chrono::milliseconds(0), ready);
        } else if (infinite) {
            r.cv.wait(lk, ready);
        } else {
            r.cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
        }
        if (!r.seq) {
            return is_terminal_state(r.state) ? SYI_OK : SYI_AGAIN;
        }
        seq = r.seq;
    }
    std::unique_lock<std::mutex> sl(seq->m);
    const bool infinite = timeout_ms == SYI_TIMEOUT_INFINITE;
    auto done = [&] { return seq->finished.load(); };
    if (timeout_ms == 0) {
        seq->cv.wait_for(sl, std::chrono::milliseconds(0), done);
    } else if (infinite) {
        seq->cv.wait(sl, done);
    } else {
        seq->cv.wait_for(sl, std::chrono::milliseconds(timeout_ms), done);
    }
    const bool finished = seq->finished.load(std::memory_order_acquire);
    sl.unlock();
    if (!finished) {
        return SYI_AGAIN;
    }
    count_terminal(r);
    return SYI_OK;
}

syi_status syi_request_cancel(syi_request *request) {
    if (!request || !request->st) {
        return fail(SYI_INVALID_ARGUMENT, "invalid request");
    }
    auto & r = *request->st;
    std::shared_ptr<sequence> seq;
    {
        std::lock_guard<std::mutex> lk(r.m);
        if (r.terminal()) {
            return SYI_OK;
        }
        r.cancel_requested = true;
        r.state = SYI_REQUEST_CANCELLED;
        seq = r.seq;
    }
    if (seq) {
        seq->cancel();
    }
    r.cv.notify_all();
    return SYI_OK;
}

syi_status syi_request_get_result(const syi_request *request, syi_result *out) {
    if (!request || !request->st || !out) {
        return fail(SYI_INVALID_ARGUMENT, "invalid request/result");
    }
    if (out->struct_size == 0) {
        out->struct_size = sizeof(*out);
    }
    auto & r = *request->st;
    std::shared_ptr<sequence> seq;
    {
        std::lock_guard<std::mutex> lk(r.m);
        seq = r.seq;
        if (!seq) {
            if (r.state != SYI_REQUEST_CANCELLED && r.state != SYI_REQUEST_FAILED) {
                return fail(SYI_INVALID_STATE, "request has not finished");
            }
            r.result_text.clear();
            r.result_error = r.error;
            r.result_ids.clear();
            syi_string t{r.result_text.c_str(), r.result_text.size()};
            syi_string e{r.result_error.c_str(), r.result_error.size()};
            out->text = t;
            out->token_ids = nullptr;
            out->n_tokens = 0;
            out->finish_reason = r.state == SYI_REQUEST_FAILED ? SYI_FINISH_ERROR
                                                               : (r.timed_out ? SYI_FINISH_TIMEOUT : SYI_FINISH_CANCELLED);
            out->prompt_tokens = 0;
            out->completion_tokens = 0;
            out->cached_tokens = 0;
            out->ttft_ms = 0.0;
            out->total_ms = 0.0;
            out->error = e;
            return SYI_OK;
        }
    }
    std::unique_lock<std::mutex> sl(seq->m);
    if (!seq->finished.load(std::memory_order_acquire)) {
        return fail(SYI_INVALID_STATE, "request has not finished");
    }
    r.result_text = seq->text;
    r.result_error = seq->error.empty() ? r.error : seq->error;
    r.result_ids.clear();
    r.result_ids.reserve(seq->tokens.size());
    for (const auto & t : seq->tokens) {
        r.result_ids.push_back(t.id);
    }
    const syi_finish_reason fr = finish_reason_of(r, *seq);
    const auto t_done = seq->t_done.time_since_epoch().count() ? seq->t_done : std::chrono::steady_clock::now();
    double ttft = 0.0;
    if (seq->t_first.time_since_epoch().count()) {
        ttft = std::chrono::duration<double, std::milli>(seq->t_first - seq->t_submit).count();
    }
    const double total = std::chrono::duration<double, std::milli>(t_done - seq->t_submit).count();
    syi_string text{r.result_text.c_str(), r.result_text.size()};
    syi_string err{r.result_error.c_str(), r.result_error.size()};
    out->text = text;
    out->token_ids = r.result_ids.empty() ? nullptr : r.result_ids.data();
    out->n_tokens = seq->tokens.size();
    out->finish_reason = fr;
    out->prompt_tokens = (uint64_t)seq->prompt_tokens;
    out->completion_tokens = (uint64_t)std::max(0, seq->n_generated);
    out->cached_tokens = (uint64_t)std::max(0, seq->reused);
    out->ttft_ms = ttft;
    out->total_ms = total;
    out->error = err;
    sl.unlock();
    count_terminal(r);
    return SYI_OK;
}

const char *syi_request_error(const syi_request *request) {
    if (!request || !request->st) {
        return "";
    }
    const auto & r = *request->st;
    std::lock_guard<std::mutex> lk(r.m);
    if (!r.error.empty()) {
        return r.error.c_str();
    }
    if (r.seq && !r.seq->error.empty()) {
        return r.seq->error.c_str();
    }
    return "";
}

} // extern "C"
