#include "capi_internal.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

namespace {
thread_local std::string g_error;
}

namespace si::capi {
void set_last_error(const std::string & msg) {
    g_error = msg;
}
syi_status fail(syi_status status, const std::string & msg) {
    g_error = msg;
    return status;
}
} // namespace si::capi

extern "C" {

uint32_t syi_abi_version(void) {
    return SYI_ABI_VERSION;
}

const char *syi_version_string(void) {
    return "0.1.0";
}

const char *syi_status_string(syi_status status) {
    switch (status) {
    case SYI_OK: return "ok";
    case SYI_END: return "end";
    case SYI_AGAIN: return "again";
    case SYI_INVALID_ARGUMENT: return "invalid argument";
    case SYI_INVALID_STATE: return "invalid state";
    case SYI_UNSUPPORTED: return "unsupported";
    case SYI_OUT_OF_MEMORY: return "out of memory";
    case SYI_BACKEND: return "backend failure";
    case SYI_MODEL_LOAD: return "model load failure";
    case SYI_IO: return "I/O failure";
    case SYI_INTERNAL: return "internal error";
    case SYI_QUEUE_FULL: return "queue full";
    case SYI_SHUTDOWN: return "shutting down";
    case SYI_CANCELLED: return "cancelled";
    }
    return "unknown status";
}

const char *syi_last_error(void) {
    return g_error.c_str();
}

void syi_clear_error(void) {
    g_error.clear();
}

void syi_free(void *p) {
    std::free(p);
}

void syi_model_options_init(syi_model_options *o) {
    if (!o) {
        return;
    }
    std::memset(o, 0, sizeof(*o));
    o->struct_size = sizeof(*o);
    o->abi_version = SYI_ABI_VERSION;
    o->device = SYI_DEVICE_AUTO;
    o->max_seq = 0;
    o->max_slots = 16;
    o->n_blocks = 0;
    o->kv_cap_mb = 0;
    o->kv_type = SYI_KV_AUTO;
    o->kv_type_v = SYI_KV_AUTO;
    o->batch_window_us = 2000;
    o->max_batch_rows = 0;
    o->max_queue_depth = 1024;
    o->max_queue_tokens = 4194304;
    o->request_timeout_ms = 0;
    o->prefix_cache = -1;
    o->pc_vram_mb = -1;
    o->pc_ram_mb = -1;
    o->pc_disk_mb = -1;
    o->engram_ram_mb = -1;
}

void syi_generation_options_init(syi_generation_options *o) {
    if (!o) {
        return;
    }
    std::memset(o, 0, sizeof(*o));
    o->struct_size = sizeof(*o);
    o->abi_version = SYI_ABI_VERSION;
    o->max_tokens = 256;
    o->temperature = 0.7f;
    o->top_p = 0.95f;
    o->top_k = 40;
    o->repeat_penalty = 1.0f;
    o->repeat_last_n = 64;
    o->timeout_ms = 0;
}

void syi_chat_message_init(syi_chat_message *m) {
    if (!m) {
        return;
    }
    std::memset(m, 0, sizeof(*m));
    m->struct_size = sizeof(*m);
}

void syi_model_info_init(syi_model_info *i) {
    if (!i) {
        return;
    }
    std::memset(i, 0, sizeof(*i));
    i->struct_size = sizeof(*i);
}

void syi_model_stats_init(syi_model_stats *s) {
    if (!s) {
        return;
    }
    std::memset(s, 0, sizeof(*s));
    s->struct_size = sizeof(*s);
}

void syi_token_init(syi_token *t) {
    if (!t) {
        return;
    }
    std::memset(t, 0, sizeof(*t));
    t->struct_size = sizeof(*t);
}

void syi_result_init(syi_result *r) {
    if (!r) {
        return;
    }
    std::memset(r, 0, sizeof(*r));
    r->struct_size = sizeof(*r);
}

} // extern "C"
