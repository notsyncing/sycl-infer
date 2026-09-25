#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "chat.h"
#include "chat_util.h"
#include "cpu_isa.h"
#include "cpu_types.h"
#include "engine.h"
#include "image.h"
#include "model.h"
#include "multimodal.h"
#include "server.h"
#include "vision.h"

using namespace si;

static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        return false;
    }
    out.resize((size_t)n);
    const bool ok = out.empty() || fread(out.data(), 1, out.size(), f) == out.size();
    fclose(f);
    return ok;
}

// Context length when --ctx is not given: fits the long-context benchmarks
// (16k depth + 512 prefill + 32 decode) out of the box; the KV pool commits
// memory lazily, so this costs virtual address space, not committed RAM.
static const int kDefaultCtx = 20480;
static const char * kDefaultModel = "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";

static void usage(const char * prog) {
    fprintf(stderr,
            "usage:\n"
            "  %s [--model <gguf>] [common] <command> [options]\n"
            "\n"
            "commands:\n"
            "  serve                        OpenAI-compatible HTTP server\n"
            "  gen                          generate once from --prompt\n"
            "\n"
            "common (both commands):\n"
            "  --model <gguf>               model path (default: %s)\n"
            "  --device cpu|gpu|auto        compute backend (default auto = PF_DEVICE,\n"
            "                               else gpu)\n"
            "  --cpu-threads N              CPU backend worker threads (default: physical\n"
            "                               cores, env PF_CPU_THREADS; 0 = auto)\n"
            "  --layer-map L:dev,...        pipeline-parallel layer placement across\n"
            "                               devices, e.g. 0-11:gpu,12-23:cpu (closed\n"
            "                               ranges, must cover every layer)\n"
            "  --ctx N | full               max sequence length (default %d, env PF_CTX);\n"
            "                               `full` reads <arch>.context_length from the GGUF\n"
            "  --blocks N                   KV blocks committed at startup (default 512 or\n"
            "                               ceil(ctx/%d); grows lazily up to the cap)\n"
            "  --kv-cap-mb N                cap for the grown KV pool (default: auto from\n"
            "                               --ctx; env PF_KV_CAP_MB).  Also caps the sum of\n"
            "                               the three prefix-cache tiers, shrinking disk,\n"
            "                               then RAM, then VRAM.\n"
            "  --mmproj <mmproj.gguf>       vision projector needed by --image/--video\n"
            "  --mtp N                      MTP (NextN) speculative draft length: run\n"
            "                               the model's draft head N times per verify\n"
            "                               (0/absent = off; env PF_MTP; needs 2+ GPUs)\n"
            "  --mtp-device N               device partition the MTP draft layer (and\n"
            "                               its KV slice) runs on (default 0; env\n"
            "                               PF_MTP_DEV).  The MTP slice counts toward\n"
            "                               --kv-cap-mb and all prefix-cache tiers.\n"
            "  --audio-mmproj <gguf>        audio tower needed by --audio\n"
            "  --kv-type T                  KV cache storage type: i4|i8|bf16|f16|f32,\n"
            "                               or K:V to size K and V independently\n"
            "                               (e.g. i4:i8: 4-bit keys, 8-bit values)\n"
            "                               (default i8; overrides PF_KV_TYPE; all math\n"
            "                               stays fp32, i4 packs two values per byte)\n"
            "\n"
            "serve:\n"
            "  --host H                     bind address (default 0.0.0.0)\n"
            "  --port N                     HTTP port (default 8080)\n"
            "\n"
            "gen:\n"
            "  --prompt \"...\"               prompt text (chat-templated unless --raw)\n"
            "  --raw                        encode the prompt verbatim (no chat template)\n"
            "  --thinking                   chat template `enable_thinking` (default off)\n"
            "  --image <file>               attach an image (repeatable; needs --mmproj)\n"
            "  --video <file>               attach a video (repeatable; needs --mmproj)\n"
            "  --audio <file>               attach audio (repeatable; needs --audio-mmproj)\n"
            "  --max-video-frames N         frames sampled from each video (default 16)\n"
            "  --max-video-side N           video frames above this side are rejected (default 768)\n"
            "  --max-tokens N               generation limit (default 256)\n"
            "  --temp T                     sampling temperature (default 0.7)\n"
            "  --top-p P                    nucleus sampling (default 0.95)\n"
            "  --top-k K                    top-k sampling (default 40)\n"
            "\n"
            "prefix cache (three LRU tiers; lookups promote, evictions demote\n"
            "VRAM -> RAM -> disk -> dropped):\n"
            "  --pc-vram-mb N               device/VRAM budget (alias --pc-mem-mb; env\n"
            "                               PF_PC_VRAM_MB); selects the checkpoint count\n"
            "  --pc-ram-mb N                host/RAM budget (default 512, env PF_PC_RAM_MB;\n"
            "                               0 disables the tier)\n"
            "  --pc-dir DIR                 enable the disk tier, records under DIR\n"
            "                               (model-specific subdir, read at startup)\n"
            "  --pc-disk-mb N               disk budget (default 1024, env PF_PC_DISK_MB;\n"
            "                               0 = unbounded)\n"
            "\n"
            "  -h, --help                   this message\n"
            "\n"
            "env: PF_CTX, PF_KV_CAP_MB, PF_KV_TYPE, PF_PREFIX_CACHE, PF_PC_STATES,\n"
            "  PF_DEVICE, PF_CPU_ISA, PF_CPU_THREADS, PF_DP4A, PF_DP4A_DEC,\n"
            "  PF_GEMM_DNNL, PF_NOGRAPH, PF_PROF, PF_TIME.\n",
            prog, kDefaultModel, kDefaultCtx, kBlockSize);
}

int main(int argc, char ** argv) {
    std::string model_path = kDefaultModel;
    std::string prompt;
    std::string host = "0.0.0.0";
    int port = 8080;
    int ctx = 0; // 0 = auto (kDefaultCtx)
    if (const char * e = getenv("PF_CTX")) {
        ctx = atoi(e);
    }
    bool ctx_full = false;   // --ctx full: read the maximum from the GGUF
    int blocks = 0;          // KV blocks committed at startup; 0 = auto
    int kv_cap_mb = INT_MIN; // INT_MIN = auto (size the cap from --ctx)
    if (const char * e = getenv("PF_KV_CAP_MB")) {
        kv_cap_mb = atoi(e);
    }
    int max_tokens = 256;
    float temp = 0.7f;
    float top_p = 0.95f;
    int top_k = 40;
    bool raw = false;
    bool thinking = false; // --thinking: chat template `enable_thinking`
    std::string mmproj_path;
    std::string audio_mmproj_path;
    std::vector<std::string> image_paths;
    std::vector<std::string> audio_paths;
    std::vector<std::string> video_paths;
    int max_video_frames = 16;
    int max_video_side = 768;
    // three-tier prefix cache: --pc-vram-mb (device), --pc-ram-mb (host), and
    // --pc-disk-mb (directory via --pc-dir); --pc-mem-mb is a VRAM alias
    std::string pc_dir;
    int pc_disk_mb = -1;
    int pc_mem_mb = -1;
    int pc_ram_mb = -1;
    int pc_vram_mb = -1;
    int device = -1; // -1 auto (PF_DEVICE), 0 gpu, 1 cpu
    std::string layer_map; // multi-device: "0-13:gpu,14-27:cpu"
    int mtp_k = -1;        // MTP draft length (0/absent = off, env PF_MTP)
    int mtp_dev = 0;       // device partition the MTP draft layer runs on
    std::string cmd;
    bool bad_arg = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        // `a` is the option currently being parsed; a missing value is a hard
        // error rather than silently overwriting the default with ""
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", a.c_str());
                bad_arg = true;
                return "";
            }
            return argv[++i];
        };
        if (a == "--model") {
            model_path = next();
        } else if (a == "--prompt") {
            prompt = next();
        } else if (a == "--ctx") {
            std::string v = next();
            if (v == "full") {
                ctx_full = true;
            } else {
                ctx = atoi(v.c_str());
            }
        } else if (a == "--blocks") {
            blocks = atoi(next().c_str());
        } else if (a == "--kv-cap-mb") {
            kv_cap_mb = atoi(next().c_str());
        } else if (a == "--port") {
            port = atoi(next().c_str());
        } else if (a == "--host") {
            host = next();
        } else if (a == "--max-tokens") {
            max_tokens = atoi(next().c_str());
        } else if (a == "--temp") {
            temp = (float)atof(next().c_str());
        } else if (a == "--top-p") {
            top_p = (float)atof(next().c_str());
        } else if (a == "--top-k") {
            top_k = atoi(next().c_str());
        } else if (a == "--raw") {
            raw = true;
        } else if (a == "--thinking" || a == "--enable-thinking") {
            thinking = true;
        } else if (a == "--mmproj") {
            mmproj_path = next();
        } else if (a == "--audio-mmproj") {
            audio_mmproj_path = next();
        } else if (a == "--image") {
            image_paths.push_back(next());
        } else if (a == "--audio") {
            audio_paths.push_back(next());
        } else if (a == "--video") {
            video_paths.push_back(next());
        } else if (a == "--max-video-frames") {
            max_video_frames = atoi(next().c_str());
        } else if (a == "--max-video-side") {
            max_video_side = atoi(next().c_str());
        } else if (a == "--pc-dir") {
            pc_dir = next();
        } else if (a == "--pc-disk-mb") {
            pc_disk_mb = atoi(next().c_str());
        } else if (a == "--pc-mem-mb") {
            pc_mem_mb = atoi(next().c_str());
        } else if (a == "--pc-ram-mb") {
            pc_ram_mb = atoi(next().c_str());
        } else if (a == "--pc-vram-mb") {
            pc_vram_mb = atoi(next().c_str());
        } else if (a == "--kv-type") {
            const std::string v = next();
            kv_dtype_t kt, vt;
            if (!kv_dtype_parse_pair(v.c_str(), kt, vt)) {
                fprintf(stderr, "error: --kv-type expects i4|i8|bf16|f16|f32, or K:V "
                                "(e.g. i8:i4)\n");
                return 1;
            }
            kv_dtype_set_kv(kt, vt);
        } else if (a == "--layer-map") {
            layer_map = next();
        } else if (a == "--mtp") {
            mtp_k = std::atoi(next().c_str());
        } else if (a == "--mtp-device") {
            mtp_dev = std::atoi(next().c_str());
        } else if (a == "--device") {
            const std::string v = next();
            if (v == "cpu" || v == "host") {
                device = 1;
            } else if (v == "gpu") {
                device = 0;
            } else if (v == "auto") {
                device = -1;
            } else {
                fprintf(stderr, "error: --device expects cpu|gpu|auto\n");
                return 1;
            }
        } else if (a == "--cpu-threads") {
            cpu_set_thread_count(atoi(next().c_str()));
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "error: unknown option %s\n", a.c_str());
            usage(argv[0]);
            return 1;
        } else {
            cmd = a;
        }
    }

    if (bad_arg) {
        usage(argv[0]);
        return 1;
    }
    if (cmd.empty()) {
        usage(argv[0]);
        return 1;
    }

    if (ctx_full) {
        try {
            ctx = model_context_length(model_path);
        } catch (const std::exception & ex) {
            fprintf(stderr, "error: cannot read model context length: %s\n", ex.what());
            return 1;
        }
        if (ctx <= 0) {
            fprintf(stderr, "error: model does not declare a context length (%s)\n", model_path.c_str());
            return 1;
        }
    }

    const bool ctx_auto = ctx <= 0;
    if (ctx_auto) {
        ctx = kDefaultCtx;
    }

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
        if (kv_cap_mb != INT_MIN && n_blocks < need_blocks) {
            n_blocks = need_blocks;
        }

        if (mtp_dev > 0) {
            setenv("PF_MTP_DEV", std::to_string(mtp_dev).c_str(), 1);
        }
        engine e(model_path, ctx, 16, n_blocks, kv_cap_mb == INT_MIN ? -1 : kv_cap_mb, pc_dir, pc_disk_mb, pc_mem_mb,
                 pc_ram_mb, pc_vram_mb, device, layer_map, mtp_k);
        {
            const char * isa = cpu_isa_spec();
            if (e.cpu_mode) {
                fprintf(stderr, "[dev] backend=cpu (isa=%s, threads=%d), kv in host RAM\n", isa, cpu_thread_count());
            } else {
                fprintf(stderr, "[dev] backend=gpu (%s)\n",
                        e.q.get_device().get_info<sycl::info::device::name>().c_str());
            }
        }
        {
            const double kv_mb = (double)e.kv_bytes_total() / (1024.0 * 1024.0);
            const double cap_mb = (double)e.kv_bytes_cap() / (1024.0 * 1024.0);
            const double per_tok_kb =
                e.pool_blocks > 0 ? (double)e.kv_bytes_total() / (double)((size_t)e.pool_blocks * kBlockSize) / 1024.0
                                  : 0.0;
            char kvname[32];
            if (kv_k_dtype() == kv_v_dtype()) {
                snprintf(kvname, sizeof(kvname), "%s", kv_dtype_name(kv_k_dtype()));
            } else {
                snprintf(kvname, sizeof(kvname), "%s:%s", kv_dtype_name(kv_k_dtype()), kv_dtype_name(kv_v_dtype()));
            }
            fprintf(stderr,
                    "[ctx] max_seq=%d tokens%s, kv_blocks=%d (%d tokens), kv_pool=%.0f MB, "
                    "kv_cap=%d blocks (%.0f MB, %s), kv_type=%s (%.0f KB/token)\n",
                    e.max_seq, ctx_full ? " (full)" : (ctx_auto ? " (auto)" : ""), n_blocks, n_blocks * kBlockSize,
                    kv_mb, e.pool_cap, cap_mb, e.kv_virtual ? "virtual USM" : "fixed", kvname,
                    per_tok_kb);
            // device memory report (best effort: the free_memory aspect is not
            // implemented by every backend)
            try {
                if (e.q.get_device().has(sycl::aspect::ext_intel_free_memory)) {
                    const uint64_t free_b = e.q.get_device().get_info<sycl::ext::intel::info::device::free_memory>();
                    fprintf(stderr, "[ctx] device free memory %.0f MB, KV reservation %.0f MB%s\n",
                            (double)free_b / (1024.0 * 1024.0), cap_mb,
                            (double)free_b < cap_mb * 0.5 ? " (WARNING: KV cap exceeds half the free memory)" : "");
                }
            } catch (...) {
            }
        }
        if (cmd == "serve") {
            server_config cfg;
            cfg.host = host;
            cfg.port = port;
            cfg.mmproj_path = mmproj_path;
            cfg.audio_mmproj_path = audio_mmproj_path;
            cfg.max_video_frames = max_video_frames;
            cfg.max_video_side = max_video_side;
            return serve(e, cfg);
        }
        if (cmd == "gen") {
            gen_params gp;
            gp.max_tokens = max_tokens;
            gp.temperature = temp;
            gp.top_p = top_p;
            gp.top_k = top_k;
            utf8_stream_buffer ub;
            const bool dump_gen = getenv("PF_DUMP_GEN") != nullptr;
            auto emit = [&](int tok) {
                if (dump_gen) {
                    fprintf(stderr, "[gen] id=%d '%s'\n", tok, e.tk.token_piece(tok).c_str());
                }
                fputs(ub.push(e.tk.token_piece(tok)).c_str(), stdout);
                fflush(stdout);
                return true;
            };
            if (!image_paths.empty() || !video_paths.empty() || !audio_paths.empty()) {
                std::vector<mm_image> imgs;
                std::vector<mm_video> vids;
                std::vector<mm_audio> auds;
                std::vector<mm_media_ref> order;
                vision_model vm_owned;
                audio_model am_owned;
                vision_model & vm = vm_owned;
                audio_model & am = am_owned;
                bool am_loaded = false;
                image_preproc_cfg img_cfg;
                mm_video_fmt vfmt;
                vfmt.max_frames = max_video_frames;
                vfmt.max_side = max_video_side;
                audio_preproc_cfg acfg;
                bool vm_loaded = false;
                auto ensure_vm = [&]() {
                    if (!vm_loaded) {
                        if (mmproj_path.empty()) {
                            throw std::runtime_error("--image/--video requires --mmproj <mmproj.gguf>");
                        }
                        vm_owned.load(mmproj_path);
                        img_cfg.patch_size = vm.hp.patch_size;
                        img_cfg.merge = vm.hp.merge;
                        const int patch_area = img_cfg.patch_size * img_cfg.patch_size * img_cfg.merge * img_cfg.merge;
                        img_cfg.min_pixels = 8 * patch_area;
                        img_cfg.max_pixels = kMaxImgTokens * patch_area;
                        for (int c = 0; c < 3; c++) {
                            img_cfg.mean[c] = vm.hp.mean[c];
                            img_cfg.std[c] = vm.hp.std[c];
                        }
                        vm_loaded = true;
                    }
                };
                auto ensure_am = [&]() {
                    if (!am_loaded) {
                        if (audio_mmproj_path.empty()) {
                            throw std::runtime_error("--audio requires --audio-mmproj <audio-mmproj.gguf>");
                        }
                        am_owned.load(audio_mmproj_path);
                        acfg.sample_rate = am.hp.sample_rate;
                        acfg.n_fft = am.hp.n_fft;
                        acfg.hop = am.hp.hop;
                        acfg.n_mel = am.hp.n_mel;
                        acfg.f_min = am.hp.f_min;
                        acfg.f_max = am.hp.f_max;
                        am_loaded = true;
                    }
                };
                for (const std::string & p : image_paths) {
                    ensure_vm();
                    std::vector<uint8_t> rgb;
                    int w = 0, h = 0;
                    std::string err;
                    if (!mm_image_decode_file(p, rgb, w, h, &err)) {
                        throw std::runtime_error("image: " + err);
                    }
                    imgs.push_back(mm_image_preprocess(rgb.data(), w, h, img_cfg));
                    order.push_back({MM_KIND_IMAGE, (int)imgs.size() - 1});
                }
                for (const std::string & p : video_paths) {
                    ensure_vm();
                    mm_video vid;
                    std::string err;
                    if (!mm_video_decode_file(p, vfmt, vid, &err)) {
                        throw std::runtime_error("video: " + err);
                    }
                    vids.push_back(std::move(vid));
                    order.push_back({MM_KIND_VIDEO, (int)vids.size() - 1});
                }
                for (const std::string & p : audio_paths) {
                    ensure_am();
                    mm_audio aud;
                    std::string err;
                    std::vector<uint8_t> bytes;
                    if (!read_file(p, bytes)) {
                        throw std::runtime_error("audio: cannot read " + p);
                    }
                    if (!mm_audio_decode_mem(bytes.data(), bytes.size(), acfg, aud, &err)) {
                        // not a (readable) WAV: hand the file to the ffmpeg CLI
                        if (!mm_audio_decode_ffmpeg(p, aud, &err)) {
                            throw std::runtime_error("audio: " + err);
                        }
                    }
                    auds.push_back(std::move(aud));
                    order.push_back({MM_KIND_AUDIO, (int)auds.size() - 1});
                }
                chat_msg m;
                m.role = "user";
                chat_part tpart;
                tpart.kind = chat_part_kind::TEXT;
                tpart.text = prompt;
                for (const mm_media_ref & mr : order) {
                    chat_part cp;
                    cp.kind = mr.kind == MM_KIND_IMAGE      ? chat_part_kind::IMAGE
                               : mr.kind == MM_KIND_VIDEO   ? chat_part_kind::VIDEO
                                                            : chat_part_kind::AUDIO;
                    m.parts.push_back(std::move(cp));
                }
                m.parts.push_back(std::move(tpart));
                const std::string rendered =
                    render_chat(e.m.chat_template, {m}, /*add_generation_prompt=*/true, thinking);
                mm_prompt mp = mm_build_prompt_mixed_device(vm, am, e.q, e.tk, rendered, imgs, vids, auds, order,
                                                            e.m.hp.n_embd, e.d_img_embd, max_video_frames);
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
                toks = e.tk.encode(render_chat(e.m.chat_template, msgs, true, thinking));
            }
            // PF_DUMP_PROMPT: dump the exact prompt the model is conditioned on
            // (the rendered chat text and/or the token ids) - the fastest way to
            // tell a bad chat template from a bad forward pass.
            if (getenv("PF_DUMP_PROMPT")) {
                if (!raw) {
                    std::vector<chat_msg> msgs = {{"user", prompt}};
                    fprintf(stderr, "[prompt] %s\n", render_chat(e.m.chat_template, msgs, true, thinking).c_str());
                }
                fprintf(stderr, "[prompt ids]");
                for (int tkv : toks) {
                    fprintf(stderr, " %d", tkv);
                }
                fprintf(stderr, "\n");
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
