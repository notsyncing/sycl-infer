cmake_minimum_required(VERSION 3.20)
project(sycl-infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CTest) # opt-in model/GPU registrations below; default ctest stays hermetic

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# The SYCL device images are JIT-compiled by IGC at run time.  A Debug build
# (icpx defaults to -O0, which DPC++ forwards to the device backend as
# -cl-opt-disable) makes the kernels.cpp device image ~20x larger and its IGC
# translation take minutes: every GPU workload then looks like a hang.  Keep
# device code optimized even when the host build is Debug.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O2)
endif()

# The router is a plain host program: it needs neither SYCL nor oneDNN.  This
# option builds ONLY it, with the system compiler, so its container image does
# not need the oneAPI toolchain at all (and the build does not pay for the
# device pass).  See docs/design/16-router.md §11.
option(SI_ROUTER_ONLY "build only the router (no SYCL, no oneDNN, no GPU)" OFF)

if(NOT SI_ROUTER_ONLY)
    find_program(ICXX NAMES icpx PATHS /opt/intel/oneapi/compiler/latest/bin ENV PATH)
    if(NOT ICXX)
        message(FATAL_ERROR "icpx not found")
    endif()
    set(CMAKE_CXX_COMPILER ${ICXX})

    add_compile_options(-fsycl)
    add_link_options(-fsycl)
endif()

option(SYCL_INFER_AOT "AOT compile for the target GPU (avoids runtime JIT)" OFF)
# The ocloc target is device-specific and must agree with the device profile the
# binary will select at runtime (src/device/device_profile.h), so it is a BUILD-side
# mirror of a profile value.  The default stays `adl-p` because the primary dev
# box is the Iris Xe; the 2x A770 host must pass -DSYCL_INFER_AOT_DEVICE=acm-g10.
# Check `ocloc query` for any other SKU rather than inheriting either default.
set(SYCL_INFER_AOT_DEVICE "adl-p" CACHE STRING "ocloc target device for SYCL_INFER_AOT (adl-p = Iris Xe, acm-g10 = A770/DG2)")
set(SYCL_INFER_AOT_PROFILE "" CACHE STRING "device profile key baked into the binary; empty = auto-detect from the SYCL device name")
if(SYCL_INFER_AOT AND NOT SI_ROUTER_ONLY)
    add_compile_options(-fsycl-targets=spir64_gen -Xs "-device ${SYCL_INFER_AOT_DEVICE}")
    add_link_options(-fsycl-targets=spir64_gen -Xs "-device ${SYCL_INFER_AOT_DEVICE}")
    if(SYCL_INFER_AOT_PROFILE)
        # bake the profile in, so an AOT binary built for one card says so even
        # when it runs somewhere the name would mislead
        add_compile_definitions(SI_FORCE_DEVICE_PROFILE="${SYCL_INFER_AOT_PROFILE}")
        message(STATUS "sycl-infer: device profile baked in: ${SYCL_INFER_AOT_PROFILE}")
    endif()
    # AOT lowering (llvm-foreach -> ocloc/IGC) runs once per device image at the
    # final link and costs minutes of IGC work per image.  The default code split
    # (`auto`) merges a whole module into one image, which serialises that work;
    # `per_kernel` emits one image per kernel so they can be lowered in parallel
    # (and skipped individually if the ocloc cache is usable).
    add_link_options(-fsycl-device-code-split=per_kernel)
    # The post-link device step defaults to a single ocloc process even on a
    # many-core host; this is what makes an AOT build look serial.  Each job is a
    # separate IGC translation, so RAM use scales with the count.
    set(SYCL_INFER_AOT_JOBS "" CACHE STRING
        "Parallel ocloc/IGC jobs for the SYCL_INFER_AOT device-link step (default: nproc)")
    if(NOT SYCL_INFER_AOT_JOBS)
        include(ProcessorCount)
        ProcessorCount(SYCL_INFER_AOT_JOBS)
    endif()
    if(SYCL_INFER_AOT_JOBS LESS 1)
        set(SYCL_INFER_AOT_JOBS 1)
    endif()
    add_link_options(-fsycl-max-parallel-link-jobs=${SYCL_INFER_AOT_JOBS})
    # NOTE: `-g` (a Debug build) makes each device image several times larger and
    # slower to lower; configure with `-DCMAKE_BUILD_TYPE=Release` for AOT.
endif()
# ---------------------------------------------------------------- core library
# Everything from here to the router section needs SYCL; SI_ROUTER_ONLY skips
# it (the router links none of it).
if(NOT SI_ROUTER_ONLY)

add_library(sycl_infer_core STATIC
    third_party/unicode.cpp
    third_party/unicode-data.cpp
    # device: per-GPU tuning profiles + the registry that selects one
    src/device/device_registry.cpp
    src/device/profiles/arc_a770.cpp
    src/device/profiles/iris_xe.cpp
    # common: quantization formats and shared low-level helpers
    src/common/w8.cpp
    src/common/w4.cpp
    # common: host ISA detection (compiled without -fsycl, see below)
    src/common/cpu_isa.cpp
    # backend: the shared interface plus the GPU and CPU implementations
    src/backend/dnnl_gemm.cpp
    src/backend/gpu/gpu_backend.cpp
    src/backend/cpu/cpu_backend.cpp
    # cpu kernels: one translation unit per host kernel (compiled without -fsycl)
    src/backend/cpu/kernels/common.cpp
    src/backend/cpu/kernels/rmsnorm.cpp
    src/backend/cpu/kernels/embed.cpp
    src/backend/cpu/kernels/copy_row.cpp
    src/backend/cpu/kernels/gemv.cpp
    src/backend/cpu/kernels/qk_norm_rope.cpp
    src/backend/cpu/kernels/attn.cpp
    src/backend/cpu/kernels/conv.cpp
    src/backend/cpu/kernels/gdn.cpp
    src/backend/cpu/kernels/gated_norm.cpp
    src/backend/cpu/kernels/xq.cpp
    src/backend/cpu/kernels/dp4a.cpp
    src/backend/cpu/kernels/i8.cpp
    src/backend/cpu/kernels/moe.cpp
    src/backend/cpu/kernels/q4exp.cpp
    # gpu kernels: one translation unit per device kernel
    src/backend/gpu/kernels/kv_type.cpp
    src/backend/gpu/kernels/rmsnorm.cpp
    src/backend/gpu/kernels/embed.cpp
    src/backend/gpu/kernels/copy_row.cpp
    src/backend/gpu/kernels/gemv.cpp
    src/backend/gpu/kernels/w4_gemv.cpp
    src/backend/gpu/kernels/qk_norm_rope.cpp
    src/backend/gpu/kernels/attn.cpp
    src/backend/gpu/kernels/attn_xmx.cpp
    src/backend/gpu/kernels/conv.cpp
    src/backend/gpu/kernels/gdn.cpp
    src/backend/gpu/kernels/gated_norm.cpp
    src/backend/gpu/kernels/xq.cpp
    src/backend/gpu/kernels/mtp.cpp
    src/backend/gpu/kernels/mtp_argmax.cpp
    src/backend/gpu/kernels/dflash.cpp
    src/backend/gpu/kernels/moe.cpp
    src/backend/gpu/kernels/dp4a_common.cpp
    src/backend/gpu/kernels/dp4a_gemv.cpp
    src/backend/gpu/kernels/dp4a_gemm.cpp
    src/backend/gpu/kernels/vit.cpp
    src/backend/gpu/kernels/at.cpp
    src/backend/gpu/kernels/q4exp.cpp
    # model: GGUF loading, weights, tokenizer
    src/model/gguf.cpp
    src/model/model.cpp
    src/model/model_w8.cpp
    src/model/qwen35.cpp
    src/model/qwen35moe.cpp
    src/model/qwen4exp.cpp
    src/model/dflash.cpp
    src/model/tokenizer.cpp
    # mm: multimodal image/video/audio preprocessing + vision/audio encoders
    src/mm/image.cpp
    src/mm/video.cpp
    src/mm/audio.cpp
    src/mm/vision.cpp
    src/mm/audio_model.cpp
    src/mm/multimodal.cpp
    # engine: forward execution, graph capture, KV pool, prefix cache
    src/engine/engine.cpp
    src/engine/engine_graph.cpp
    src/engine/engine_mtp.cpp
    src/engine/engine_dflash.cpp
    src/engine/engine_q4exp.cpp
    src/engine/engine_qsa.cpp
    src/engine/engine_kvpool.cpp
    src/engine/moe_cache.cpp
    src/engine/engine_prefix_cache.cpp
    src/engine/pc_disk.cpp
    src/engine/pc_ram.cpp
    src/engine/sampler.cpp
    # server: chat template, continuous batching scheduler, HTTP API
    src/server/chat.cpp
    src/server/chat_template.cpp
    src/server/media_fetch.cpp
    src/server/request.cpp
    src/server/response.cpp
    src/server/response_parser.cpp
    src/server/scheduler.cpp
    src/server/control.cpp
    src/server/server.cpp
    src/server/url_fetch.cpp
)
target_include_directories(sycl_infer_core PUBLIC
    src                       # so "device/device_profile.h" resolves everywhere
    src/common
    src/backend
    src/backend/cpu
    src/backend/gpu/kernels
    src/model
    src/mm
    src/engine
    src/server
    third_party
)

# The CPU kernel library is plain host C++ with AVX2 / AVX-VNNI / AVX-512 target
# attributes; `-fno-sycl` (after the global -fsycl) keeps it out of the SYCL
# device pass, so <immintrin.h> and the intrinsics are compiled host-only.
set(SI_CPU_SOURCES
    src/common/cpu_isa.cpp
    src/backend/cpu/kernels/common.cpp
    src/backend/cpu/kernels/rmsnorm.cpp
    src/backend/cpu/kernels/embed.cpp
    src/backend/cpu/kernels/copy_row.cpp
    src/backend/cpu/kernels/gemv.cpp
    src/backend/cpu/kernels/qk_norm_rope.cpp
    src/backend/cpu/kernels/attn.cpp
    src/backend/cpu/kernels/conv.cpp
    src/backend/cpu/kernels/gdn.cpp
    src/backend/cpu/kernels/gated_norm.cpp
    src/backend/cpu/kernels/xq.cpp
    src/backend/cpu/kernels/dp4a.cpp
    src/backend/cpu/kernels/i8.cpp
    src/backend/cpu/kernels/moe.cpp
    src/backend/cpu/kernels/q4exp.cpp
)
if(NOT SYCL_INFER_AOT)
    set_source_files_properties(${SI_CPU_SOURCES} PROPERTIES COMPILE_OPTIONS "-fno-sycl")
endif()

# The vendored llama.cpp Unicode tables keep one intentionally-unused helper;
# silence the warning here instead of editing third_party (clangd reads this
# from the compilation database too).
set_source_files_properties(third_party/unicode.cpp PROPERTIES COMPILE_OPTIONS "-Wno-unused-function")

# ------------------------------------------------------- oneDNN (opt-in GEMM)
# PF_GEMM_DNNL=1 runs the mode-2 prefill GEMMs through oneDNN's GPU int8
# matmul; the default path does not use it.  Additive: only the include dir,
# the library and an rpath entry so the binaries find libdnnl.so.
set(DNNL_ROOT "/opt/intel/oneapi/dnnl/2026.0" CACHE PATH "oneDNN install prefix")
find_library(DNNL_LIBRARY NAMES dnnl PATHS "${DNNL_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT DNNL_LIBRARY)
    message(FATAL_ERROR "libdnnl not found under ${DNNL_ROOT}/lib")
endif()
target_include_directories(sycl_infer_core PRIVATE "${DNNL_ROOT}/include")
target_link_libraries(sycl_infer_core PUBLIC "${DNNL_LIBRARY}")

endif() # NOT SI_ROUTER_ONLY

# ------------------------------------------------- OpenSSL (remote https images)
# The HTTP server downloads `image_url` parts that point at http(s) URLs.  TLS
# needs cpp-httplib built with OpenSSL; without it plain http:// still works and
# https:// fails with an explicit error.  Found here so server.cpp sees the
# same macro everywhere it is used.  The router uses it too (its own upstream
# connections), and it is found outside the SYCL guard so the router-only
# build gets it as well.
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND AND NOT SI_ROUTER_ONLY)
    target_compile_definitions(sycl_infer_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(sycl_infer_core PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()

# ------------------------------------------------------------------- binaries
if(NOT SI_ROUTER_ONLY)
add_executable(sycl-infer src/main.cpp)
target_link_libraries(sycl-infer PRIVATE sycl_infer_core)
endif()

# --------------------------------------------------------------------- router
# The multi-model front end (docs/design/16-router.md).  It is a plain host
# C++17 program: no SYCL device pass, no oneDNN, and it does NOT link
# sycl_infer_core - so its container is tiny and its tests need neither a GPU
# nor a model.  `-fno-sycl` has to come after the global `-fsycl`, which is
# why it is set per source file (the same trick SI_CPU_SOURCES uses).
add_library(si_router_core STATIC
    src/router/config.cpp
    src/router/registry.cpp
    src/router/rpc_client.cpp
    src/router/supervisor.cpp
    src/router/proxy.cpp
    src/common/onnx_wire.cpp
)
if(NOT SI_ROUTER_ONLY)
    # The global -fsycl would drag the router through the device pass; this
    # puts it back on the host path (the same trick SI_CPU_SOURCES uses).
    set_source_files_properties(
        src/router/config.cpp
        src/router/registry.cpp
        src/router/rpc_client.cpp
        src/router/supervisor.cpp
        src/router/proxy.cpp
        src/common/onnx_wire.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-sycl")
endif()
target_include_directories(si_router_core PUBLIC src/router src/common third_party)
find_package(Threads REQUIRED)
target_link_libraries(si_router_core PUBLIC Threads::Threads)
if (OpenSSL_FOUND)
    target_compile_definitions(si_router_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(si_router_core PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()

add_executable(sycl-infer-router src/router/main.cpp)
if(NOT SI_ROUTER_ONLY)
    set_source_files_properties(src/router/main.cpp PROPERTIES COMPILE_OPTIONS "-fno-sycl")
    target_link_options(sycl-infer-router PRIVATE -fno-sycl)
endif()
target_link_libraries(sycl-infer-router PRIVATE si_router_core)

# ------------------------------------------------------ the ONNX model service
# The document-pipeline half of the engine: a service that runs the ONNX
# models (the layout detector, the OCR recogniser, the table segmenter) on
# the GPU and answers over HTTP.  The runtime is this project's own: an ONNX
# parser, a graph optimiser (constant folding and operator fusion) and a SYCL
# kernel library - all ahead-of-time compiled, so there is no runtime JIT.
#
# It is a SEPARATE binary (sycl-infer-onnx): the kernels are the only part
# that needs the device, and the service is useful without the LLM engine.
# The sources are NOT a separate tree: the graph/transforms live under
# src/model/onnx, the executor and kernels under src/backend/onnx, the
# service under src/server/onnx, and the wire protocol in src/common so the
# router can link it without SYCL or oneDNN.
option(SI_ONNX "build the ONNX model service (sycl-infer-onnx)" ON)
if(NOT SI_ROUTER_ONLY AND SI_ONNX)
    add_library(si_onnx_core STATIC
        src/model/onnx/proto.cpp
        src/model/onnx/graph.cpp
        src/model/onnx/transforms/pipeline.cpp
        src/model/onnx/transforms/transform_util.cpp
        src/model/onnx/transforms/conv_epilogue.cpp
        src/model/onnx/transforms/elementwise_activation.cpp
        src/model/onnx/transforms/matmul_bias.cpp
        src/model/onnx/transforms/mask_box.cpp
        src/backend/onnx/ops.cpp
        src/backend/onnx/ops_shape.cpp
        src/backend/onnx/ops_exec.cpp
        src/backend/onnx/kernels.cpp
        src/backend/onnx/conv_dnnl.cpp
        src/backend/onnx/exec.cpp
        src/backend/onnx/constant_fold.cpp
        src/common/onnx_wire.cpp
        src/server/onnx/service.cpp
        src/server/control.cpp
    )
    target_include_directories(si_onnx_core PUBLIC
        src
        src/model/onnx
        src/backend/onnx
        src/server/onnx
        src/server
        src/common
        src/device
        third_party)
    # The device profile: the kernels ask it for the work-group size the
    # card accepts.
    target_sources(si_onnx_core PRIVATE
        src/device/device_registry.cpp
        src/device/profiles/arc_a770.cpp
        src/device/profiles/iris_xe.cpp
    )
    find_package(Threads REQUIRED)
    target_link_libraries(si_onnx_core PUBLIC Threads::Threads)
    # oneDNN for the group-1 f32 convolution path (conv_dnnl.cpp): the same
    # library the engine's int8 GEMM uses, and the same include/lib search
    # (DNNL_ROOT above).
    target_include_directories(si_onnx_core PRIVATE "${DNNL_ROOT}/include")
    target_link_libraries(si_onnx_core PUBLIC "${DNNL_LIBRARY}")
    if(OpenSSL_FOUND)
        target_compile_definitions(si_onnx_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
        target_link_libraries(si_onnx_core PUBLIC OpenSSL::SSL OpenSSL::Crypto)
    endif()

    add_executable(sycl-infer-onnx src/server/onnx/main.cpp)
    target_link_libraries(sycl-infer-onnx PRIVATE si_onnx_core)
endif()

# ------------------------------------------------------------------- tests
# Tests mirror the src/ layout: tests/model, tests/engine, tests/kernels (one
# source file per kernel stage, next to the test_gpu_stages runner).
set(SI_TEST_INCLUDE_DIRS tests/common)

function(si_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE sycl_infer_core)
    target_include_directories(${name} PRIVATE ${SI_TEST_INCLUDE_DIRS})
endfunction()

# The router's tests link si_router_core, not sycl_infer_core: they must run
# without a GPU, without a model and without the SYCL runtime.
function(si_add_router_test name)
    add_executable(${name} ${ARGN})
    if(NOT SI_ROUTER_ONLY)
        set_source_files_properties(${ARGN} PROPERTIES COMPILE_OPTIONS "-fno-sycl")
        target_link_options(${name} PRIVATE -fno-sycl)
    endif()
    target_include_directories(${name} PRIVATE ${SI_TEST_INCLUDE_DIRS} third_party src/router)
    target_link_libraries(${name} PRIVATE si_router_core)
endfunction()

# dev/ benchmark harnesses (off by default: they are measurement tools, not
# part of the shipped build, and each one re-lowers the device image at link).
option(SI_DEV_BENCH "build the dev/ measurement harnesses" OFF)
if (SI_DEV_BENCH)
  si_add_test(bench_mtp dev/bench_mtp.cpp)
endif()

if(NOT SI_ROUTER_ONLY)

si_add_test(test_tokenizer   tests/model/test_tokenizer.cpp)
si_add_test(test_chat_template tests/model/test_chat_template.cpp)
si_add_test(test_response_parser tests/server/test_response_parser.cpp)
si_add_test(test_request_parse  tests/server/test_request_parse.cpp)
si_add_test(test_response_json   tests/server/test_response_json.cpp)
si_add_test(test_sse_cancel     tests/server/test_sse_cancel.cpp)
si_add_test(test_control         tests/server/test_control.cpp)
si_add_test(test_url_fetch     tests/server/test_url_fetch.cpp)
si_add_test(test_sampler tests/engine/test_sampler.cpp)
si_add_test(test_sched   tests/engine/test_sched.cpp)
si_add_test(test_27b_prefill tests/engine/test_27b_prefill.cpp)
si_add_test(test_multimodal  tests/mm/test_multimodal.cpp)
si_add_test(test_compare     tests/model/test_compare.cpp)
si_add_test(test_pc_disk     tests/backend/cpu/test_pc_disk.cpp)
si_add_test(test_pc_ram      tests/backend/cpu/test_pc_ram.cpp)
si_add_test(test_cpuref      tests/backend/cpu/test_cpuref.cpp)
si_add_test(test_pc_cpu      tests/backend/cpu/test_pc_cpu.cpp)
si_add_test(test_cpu_gemv    tests/backend/cpu/kernels/test_cpu_gemv.cpp)
si_add_test(test_cpu_gdn     tests/backend/cpu/kernels/test_cpu_gdn.cpp)
si_add_test(test_forward     tests/backend/gpu/test_forward.cpp)
si_add_test(test_gpu_vs_ref  tests/backend/gpu/test_gpu_vs_ref.cpp)
si_add_test(test_pc_gpu      tests/backend/gpu/test_pc_gpu.cpp)
si_add_test(test_pc_ram_gpu  tests/backend/gpu/test_pc_ram_gpu.cpp)
si_add_test(test_enc_own     tests/mm/test_enc_own.cpp)
si_add_test(test_dev_profile tests/backend/cpu/test_dev_profile.cpp)
si_add_test(test_dev_resolve tests/backend/gpu/test_dev_resolve.cpp)
si_add_test(test_gemv        tests/backend/gpu/kernels/test_gemv.cpp)
si_add_test(test_iq_dequant  tests/backend/gpu/kernels/test_iq_dequant.cpp)
si_add_test(test_gemv_stride tests/backend/gpu/kernels/test_gemv_stride.cpp)
si_add_test(test_quant_audit tests/backend/gpu/kernels/test_quant_audit.cpp)
si_add_test(test_w4          tests/model/test_w4.cpp)
si_add_test(test_hparams     tests/model/test_hparams.cpp)
si_add_test(test_f16conv     tests/model/test_f16conv.cpp)
si_add_test(test_yarn        tests/common/test_yarn.cpp)
si_add_test(test_w4_gemm     tests/backend/gpu/kernels/test_w4_gemm.cpp)
si_add_test(test_k5_gemv     tests/backend/gpu/kernels/test_k5_gemv.cpp)
si_add_test(test_w4_vs_i8    tests/backend/gpu/kernels/test_w4_vs_i8.cpp)
si_add_test(test_w4_vs_cpuref tests/engine/test_w4_vs_cpuref.cpp)
si_add_test(test_w4_topk     tests/engine/test_w4_topk.cpp)
si_add_test(test_decode_vs_prefill tests/engine/test_decode_vs_prefill.cpp)
si_add_test(test_dp4a        tests/backend/gpu/kernels/test_dp4a_gemm.cpp)
si_add_test(test_moe_kernels tests/backend/gpu/kernels/test_moe_kernels.cpp)
si_add_test(test_moe_ref tests/model/test_moe_ref.cpp)
si_add_test(test_dflash_kernels tests/backend/gpu/kernels/test_dflash_kernels.cpp)
si_add_test(test_spec        tests/engine/test_spec.cpp)

si_add_test(test_gpu_stages
    tests/backend/gpu/kernels/test_gpu_stages.cpp
    tests/backend/gpu/kernels/rmsnorm_stage.cpp
    tests/backend/gpu/kernels/embed_stage.cpp
    tests/backend/gpu/kernels/qk_norm_rope_stage.cpp
    tests/backend/gpu/kernels/attn_stage.cpp
    tests/backend/gpu/kernels/conv_stage.cpp
    tests/backend/gpu/kernels/gdn_stage.cpp
    tests/backend/gpu/kernels/gated_norm_stage.cpp
)

endif() # NOT SI_ROUTER_ONLY

# The router's tests: they need neither SYCL nor a GPU, so they exist in every
# build (including the router-only one the container image uses).
si_add_router_test(test_router_config    tests/router/test_router_config.cpp)
si_add_router_test(test_router_proxy     tests/router/test_router_proxy.cpp)
si_add_router_test(test_router_supervisor tests/router/test_router_supervisor.cpp)
si_add_router_test(test_router_control   tests/router/test_router_control.cpp)

if(BUILD_TESTING)
    if(NOT SI_ROUTER_ONLY)
        foreach(name IN ITEMS test_response_parser test_request_parse test_response_json test_sampler test_pc_disk test_pc_ram test_cpu_gdn test_sse_cancel test_control test_hparams test_f16conv test_yarn test_dev_profile)
            add_test(NAME ${name} COMMAND ${name})
            set_tests_properties(${name} PROPERTIES LABELS "hermetic;cpu")
        endforeach()
    endif()
    # The router's tests are hermetic in every build.
    foreach(name IN ITEMS test_router_config test_router_proxy test_router_supervisor test_router_control)
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES LABELS "hermetic;cpu")
    endforeach()
    if(NOT SI_ROUTER_ONLY)
        add_test(NAME test_video_hermetic COMMAND test_multimodal --video-only)
        set_tests_properties(test_video_hermetic PROPERTIES LABELS "hermetic;media")
    endif()

    # One binary, three processes: url_fetch_mode() caches its answer in a
    # static, so each policy needs its own process.  The test sets/clears the
    # env itself, so an ambient PF_MM_URL_FETCH cannot skew the default case.
    if(NOT SI_ROUTER_ONLY)
        foreach(mode default public private)
            add_test(NAME test_url_fetch_${mode} COMMAND test_url_fetch 18141 ${mode})
            set_tests_properties(test_url_fetch_${mode} PROPERTIES LABELS "hermetic;cpu")
        endforeach()
    endif()

    # Model-backed tests are explicit: a machine without the GGUF or GPU must
    # not claim a green run by silently skipping them.  Keep the 27B split out
    # of the default ctest selection (it requires both A770s and a draft GGUF).
    option(SI_REGISTER_GPU_TESTS "register GPU/model-backed tests with CTest" OFF)
    set(SI_TEST_08B_MODEL "" CACHE FILEPATH "0.8B GGUF for GPU tests")
    set(SI_TEST_27B_MODEL "" CACHE FILEPATH "27B GGUF for two-GPU tests")
    set(SI_TEST_DFLASH_MODEL "" CACHE FILEPATH "DFlash2 GGUF for speculative tests")
    if(SI_REGISTER_GPU_TESTS)
        if(NOT EXISTS "${SI_TEST_08B_MODEL}")
            message(FATAL_ERROR "SI_REGISTER_GPU_TESTS requires SI_TEST_08B_MODEL")
        endif()
        foreach(name IN ITEMS test_gpu_stages test_gpu_vs_ref test_forward test_decode_vs_prefill test_pc_gpu test_sched test_dev_resolve)
            add_test(NAME ${name} COMMAND ${name} "${SI_TEST_08B_MODEL}")
            set_tests_properties(${name} PROPERTIES LABELS "gpu;model-0.8b")
        endforeach()
        # Was a known failure with a WILL_FAIL waiver: a prefix-cache warm resume
        # on the CPU partition gave argmax 248046/198 with logits max|diff| ~3.2-3.8.
        # Root cause: cpu_backend::synchronize() was an empty override, so
        # engine::sync_all() was a no-op on the CPU path and the block
        # serialize/deserialize let an asynchronous queue::memcpy outlive the host
        # std::vector it was reading from - a heap use-after-free, reported by glibc
        # as a corrupted chunk at an unrelated free.  It now passes, so the waiver
        # is gone and it is an ordinary test like the rest.
        add_test(NAME test_pc_cpu COMMAND test_pc_cpu "${SI_TEST_08B_MODEL}")
        set_tests_properties(test_pc_cpu PROPERTIES LABELS "cpu;model-0.8b")
        if(SI_TEST_27B_MODEL)
            if(NOT EXISTS "${SI_TEST_27B_MODEL}")
                message(FATAL_ERROR "SI_TEST_27B_MODEL does not exist")
            endif()
            add_test(NAME test_27b_vs_ref COMMAND test_w4_vs_cpuref "${SI_TEST_27B_MODEL}")
            add_test(NAME test_27b_decode_vs_prefill COMMAND test_decode_vs_prefill "${SI_TEST_27B_MODEL}")
            set_tests_properties(test_27b_vs_ref test_27b_decode_vs_prefill
                PROPERTIES LABELS "gpu;model-27b" ENVIRONMENT "TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1")
            if(SI_TEST_DFLASH_MODEL)
                if(NOT EXISTS "${SI_TEST_DFLASH_MODEL}")
                    message(FATAL_ERROR "SI_TEST_DFLASH_MODEL does not exist")
                endif()
                add_test(NAME test_27b_spec COMMAND test_spec "${SI_TEST_27B_MODEL}" "${SI_TEST_DFLASH_MODEL}")
                set_tests_properties(test_27b_spec PROPERTIES LABELS "gpu;model-27b;spec"
                    ENVIRONMENT "TEST_LAYER_MAP=0-31:gpu.0,32-63:gpu.1")
            endif()
        endif()
    endif()
endif()
