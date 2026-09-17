#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "chat.h"
#include "chat_util.h"
#include "engine.h"
#include "image.h"
#include "model.h"
#include "multimodal.h"
#include "server.h"
#include "vision.h"

using namespace si;

// Context length when --ctx is not given: fits the long-context benchmarks
// (16k depth + 512 prefill + 32 decode) out of the box; the KV pool commits
// memory lazily, so this costs virtual address space, not committed RAM.
static const int kDefaultCtx = 20480;

static void usage(const char * prog) {
    fprintf(stderr,
            "usage:\n"
            "  %s --model <gguf> [--ctx N] [--blocks N] [--kv-cap-mb N] [--port N] [--host H] serve\n"
            "  %s --model <gguf> gen --prompt \"...\" [--max-tokens N] [--temp T] [--raw]\n"
            "  %s --model <gguf> --mmproj <mmproj.gguf> gen --image <file> --prompt \"...\"\n"
            "\n"
            "  --ctx N     max sequence length in tokens (default %d, or PF_CTX).\n"
            "              A prompt longer than this is rejected with HTTP 400.\n"
            "  --ctx full  use the model's maximum context length read from the GGUF\n"
            "              (<arch>.context_length); the KV pool still grows lazily.\n"
            "  --blocks N  KV pool blocks committed at startup (default: 512, or the\n"
            "              whole context when --ctx asks for more).  Each block is\n"
            "              %d tokens; the pool grows on demand up to the cap.\n"
            "  --kv-cap-mb N  upper bound for the dynamically grown KV pool, counted\n"
            "              on the K side (default: auto = exactly what --ctx needs;\n"
            "              env PF_KV_CAP_MB overrides the flag default).\n"
            "  KV values are stored as PF_KV_TYPE (f32|bf16|f16, default bf16; bf16\n"
            "  halves the KV bytes, all math stays fp32).  PF_KV_F32=1 = PF_KV_TYPE=f32.\n"
            "\n"
            "env: PF_PREFIX_CACHE=0 disables the cross-request prompt prefix cache\n"
            "     (default on); PF_PC_STATES=N bounds the state checkpoints\n"
            "     (default 8, ~19 MB each).  PF_GEMM_DNNL=0 forces the dp4a\n"
            "     prefill path (oneDNN int8 GEMMs are the default).\n",
            prog, prog, prog, kDefaultCtx, kBlockSize);
}

int main(int argc, char ** argv) {
    std::string model_path = "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    std::string prompt;
    std::string host = "0.0.0.0";
    int port = 8080;
    int ctx = 0;                 // 0 = auto (kDefaultCtx)
    if (const char * e = getenv("PF_CTX")) ctx = atoi(e);
    bool ctx_full = false;       // --ctx full: read the maximum from the GGUF
    int blocks = 0;              // KV blocks committed at startup; 0 = auto
    int kv_cap_mb = INT_MIN;     // INT_MIN = auto (size the cap from --ctx)
    if (const char * e = getenv("PF_KV_CAP_MB")) kv_cap_mb = atoi(e);
    int max_tokens = 256;
    float temp = 0.7f;
    float top_p = 0.95f;
    int top_k = 40;
    bool raw = false;
    std::string mmproj_path;
    std::vector<std::string> image_paths;
    std::string cmd;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--model") model_path = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--ctx") {
            std::string v = next();
            if (v == "full") ctx_full = true;
            else ctx = atoi(v.c_str());
        }
        else if (a == "--blocks") blocks = atoi(next().c_str());
        else if (a == "--kv-cap-mb") kv_cap_mb = atoi(next().c_str());
        else if (a == "--port") port = atoi(next().c_str());
        else if (a == "--host") host = next();
        else if (a == "--max-tokens") max_tokens = atoi(next().c_str());
        else if (a == "--temp") temp = (float) atof(next().c_str());
        else if (a == "--top-p") top_p = (float) atof(next().c_str());
        else if (a == "--top-k") top_k = atoi(next().c_str());
        else if (a == "--raw") raw = true;
        else if (a == "--mmproj") mmproj_path = next();
        else if (a == "--image") image_paths.push_back(next());
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else cmd = a;
    }

    if (cmd.empty()) { usage(argv[0]); return 1; }

    if (ctx_full) {
        try {
            ctx = model_context_length(model_path);
        } catch (const std::exception & ex) {
            fprintf(stderr, "error: cannot read model context length: %s\n", ex.what());
            return 1;
        }
        if (ctx <= 0) {
            fprintf(stderr, "error: model does not declare a context length (%s)\n",
                    model_path.c_str());
            return 1;
        }
    }

    const bool ctx_auto = ctx <= 0;
    if (ctx_auto) ctx = kDefaultCtx;

    try {
        // The block pool must hold max_seq tokens or prefill cannot use the
        // whole context.  Historical default: 512 blocks committed at startup
        // (16k tokens) which is enough for short contexts and keeps the memory
        // footprint low; the cap is auto-sized to the context so the pool can
        // grow into the rest on demand.  An explicit --ctx keeps the old eager
        // sizing (max(512, ceil(ctx/32))), an explicit --blocks is honored as
        // the initial committed size.  --ctx full can name a very large context,
        // so it commits lazily like auto instead of eagerly.
        const bool lazy_blocks = ctx_auto || ctx_full;
        const int need_blocks = (ctx + kBlockSize - 1) / kBlockSize;
        int n_blocks = blocks > 0 ? blocks : (lazy_blocks ? 512 : std::max(512, need_blocks));
        if (kv_cap_mb != INT_MIN && n_blocks < need_blocks) n_blocks = need_blocks;

        engine e(model_path, ctx, 16, n_blocks, kv_cap_mb == INT_MIN ? -1 : kv_cap_mb);
        {
            const double kv_mb = (double) e.kv_bytes_total() / (1024.0 * 1024.0);
            const double cap_mb = (double) e.kv_bytes_cap() / (1024.0 * 1024.0);
            const double per_tok_kb = e.pool_blocks > 0
                                          ? (double) e.kv_bytes_total() /
                                                (double) ((size_t) e.pool_blocks * kBlockSize) / 1024.0
                                          : 0.0;
            fprintf(stderr,
                    "[ctx] max_seq=%d tokens%s, kv_blocks=%d (%d tokens), kv_pool=%.0f MB, "
                    "kv_cap=%d blocks (%.0f MB, %s), kv_type=%s (%.0f KB/token)\n",
                    e.max_seq, ctx_full ? " (full)" : (ctx_auto ? " (auto)" : ""), n_blocks,
                    n_blocks * kBlockSize,
                    kv_mb, e.pool_cap, cap_mb, e.kv_virtual ? "virtual USM" : "fixed",
                    kv_dtype_name(kv_dtype()), per_tok_kb);
            // device memory report (best effort: the free_memory aspect is not
            // implemented by every backend)
            try {
                if (e.q.get_device().has(sycl::aspect::ext_intel_free_memory)) {
                    const uint64_t free_b = e.q.get_device().get_info<sycl::ext::intel::info::device::free_memory>();
                    fprintf(stderr, "[ctx] device free memory %.0f MB, KV reservation %.0f MB%s\n",
                            (double) free_b / (1024.0 * 1024.0), cap_mb,
                            (double) free_b < cap_mb * 0.5 ? " (WARNING: KV cap exceeds half the free memory)" : "");
                }
            } catch (...) {}
        }
        if (cmd == "serve") {
            server_config cfg;
            cfg.host = host;
            cfg.port = port;
            cfg.mmproj_path = mmproj_path;
            return serve(e, cfg);
        }
        if (cmd == "gen") {
            gen_params gp;
            gp.max_tokens = max_tokens;
            gp.temperature = temp;
            gp.top_p = top_p;
            gp.top_k = top_k;
            utf8_stream_buffer ub;
            auto emit = [&](int tok) {
                fputs(ub.push(e.tk.token_piece(tok)).c_str(), stdout);
                fflush(stdout);
                return true;
            };
            if (!image_paths.empty()) {
                if (mmproj_path.empty())
                    throw std::runtime_error("--image requires --mmproj <mmproj.gguf>");
                vision_model vm;
                vm.load(mmproj_path);
                std::vector<mm_image> imgs;
                image_preproc_cfg cfg;
                cfg.patch_size = vm.hp.patch_size;
                cfg.merge = vm.hp.merge;
                const int patch_area = cfg.patch_size * cfg.patch_size * cfg.merge * cfg.merge;
                cfg.min_pixels = 8 * patch_area;
                cfg.max_pixels = kMaxImgTokens * patch_area;
                for (int c = 0; c < 3; c++) {
                    cfg.mean[c] = vm.hp.mean[c];
                    cfg.std[c] = vm.hp.std[c];
                }
                for (const std::string & p : image_paths) {
                    std::vector<uint8_t> rgb;
                    int w = 0, h = 0;
                    std::string err;
                    if (!mm_image_decode_file(p, rgb, w, h, &err))
                        throw std::runtime_error("image: " + err);
                    imgs.push_back(mm_image_preprocess(rgb.data(), w, h, cfg));
                }
                chat_msg m;
                m.role = "user";
                for (size_t i = 0; i < imgs.size(); i++) m.parts.push_back({true, ""});
                m.parts.push_back({false, prompt});
                const std::string rendered =
                    render_chat(e.m.chat_template, {m}, /*add_generation_prompt=*/true, false);
                mm_prompt mp =
                    mm_build_prompt_device(vm, e.q, e.tk, rendered, imgs, e.m.hp.n_embd,
                                           e.d_img_embd);
                e.generate_mm(mp, gp, emit);
                fputs(ub.flush().c_str(), stdout);
                printf("\n");
                return 0;
            }
            std::vector<int> toks;
            if (raw) {
                toks = e.tk.encode(prompt, /*parse_special=*/true);
            } else {
                std::vector<chat_msg> msgs = {{"user", prompt}};
                toks = e.tk.encode(render_chat(e.m.chat_template, msgs, true, false));
            }
            e.generate(toks, gp, emit);
            fputs(ub.flush().c_str(), stdout);
            printf("\n");
            return 0;
        }
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    usage(argv[0]);
    return 1;
}
