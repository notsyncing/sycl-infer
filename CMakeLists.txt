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

find_program(ICXX NAMES icpx PATHS /opt/intel/oneapi/compiler/latest/bin ENV PATH)
if(NOT ICXX)
    message(FATAL_ERROR "icpx not found")
endif()
set(CMAKE_CXX_COMPILER ${ICXX})

add_compile_options(-fsycl)
add_link_options(-fsycl)

option(SYCL_INFER_AOT "AOT compile for the target GPU (avoids runtime JIT)" OFF)
# The ocloc target is device-specific and must agree with the device profile the
# binary will select at runtime (src/device/device_profile.h), so it is a BUILD-side
# mirror of a profile value.  The default stays `adl-p` because the primary dev
# box is the Iris Xe; the 2x A770 host must pass -DSYCL_INFER_AOT_DEVICE=acm-g10.
# Check `ocloc query` for any other SKU rather than inheriting either default.
set(SYCL_INFER_AOT_DEVICE "adl-p" CACHE STRING "ocloc target device for SYCL_INFER_AOT (adl-p = Iris Xe, acm-g10 = A770/DG2)")
set(SYCL_INFER_AOT_PROFILE "" CACHE STRING "device profile key baked into the binary; empty = auto-detect from the SYCL device name")
if(SYCL_INFER_AOT)
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
    src/backend/gpu/kernels/dp4a_common.cpp
    src/backend/gpu/kernels/dp4a_gemv.cpp
    src/backend/gpu/kernels/dp4a_gemm.cpp
    src/backend/gpu/kernels/vit.cpp
    src/backend/gpu/kernels/at.cpp
    # model: GGUF loading, weights, tokenizer
    src/model/gguf.cpp
    src/model/model.cpp
    src/model/model_w8.cpp
    src/model/qwen35.cpp
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
    src/engine/engine_kvpool.cpp
    src/engine/engine_prefix_cache.cpp
    src/engine/pc_disk.cpp
    src/engine/pc_ram.cpp
    src/engine/sampler.cpp
    # server: chat template, continuous batching scheduler, HTTP API
    src/server/chat.cpp
    src/server/chat_template.cpp
    src/server/response_parser.cpp
    src/server/scheduler.cpp
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

# ------------------------------------------------- OpenSSL (remote https images)
# The HTTP server downloads `image_url` parts that point at http(s) URLs.  TLS
# needs cpp-httplib built with OpenSSL; without it plain http:// still works and
# https:// fails with an explicit error.  Found here so server.cpp sees the
# same macro everywhere it is used.
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    target_compile_definitions(sycl_infer_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(sycl_infer_core PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()

# ------------------------------------------------------------------- binaries
add_executable(sycl-infer src/main.cpp)
target_link_libraries(sycl-infer PRIVATE sycl_infer_core)

# ------------------------------------------------------------------- tests
# Tests mirror the src/ layout: tests/model, tests/engine, tests/kernels (one
# source file per kernel stage, next to the test_gpu_stages runner).
set(SI_TEST_INCLUDE_DIRS tests/common)

function(si_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE sycl_infer_core)
    target_include_directories(${name} PRIVATE ${SI_TEST_INCLUDE_DIRS})
endfunction()

# dev/ benchmark harnesses (off by default: they are measurement tools, not
# part of the shipped build, and each one re-lowers the device image at link).
option(SI_DEV_BENCH "build the dev/ measurement harnesses" OFF)
if (SI_DEV_BENCH)
  si_add_test(bench_mtp dev/bench_mtp.cpp)
endif()

si_add_test(test_tokenizer   tests/model/test_tokenizer.cpp)
si_add_test(test_chat_template tests/model/test_chat_template.cpp)
si_add_test(test_response_parser tests/server/test_response_parser.cpp)
si_add_test(test_sse_cancel     tests/server/test_sse_cancel.cpp)
si_add_test(test_url_fetch     tests/server/test_url_fetch.cpp)
si_add_test(test_sampler tests/engine/test_sampler.cpp)
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
si_add_test(test_gemv        tests/backend/gpu/kernels/test_gemv.cpp)
si_add_test(test_iq_dequant  tests/backend/gpu/kernels/test_iq_dequant.cpp)
si_add_test(test_gemv_stride tests/backend/gpu/kernels/test_gemv_stride.cpp)
si_add_test(test_quant_audit tests/backend/gpu/kernels/test_quant_audit.cpp)
si_add_test(test_w4          tests/model/test_w4.cpp)
si_add_test(test_w4_gemm     tests/backend/gpu/kernels/test_w4_gemm.cpp)
si_add_test(test_k5_gemv     tests/backend/gpu/kernels/test_k5_gemv.cpp)
si_add_test(test_w4_vs_i8    tests/backend/gpu/kernels/test_w4_vs_i8.cpp)
si_add_test(test_w4_vs_cpuref tests/engine/test_w4_vs_cpuref.cpp)
si_add_test(test_w4_topk     tests/engine/test_w4_topk.cpp)
si_add_test(test_decode_vs_prefill tests/engine/test_decode_vs_prefill.cpp)
si_add_test(test_dp4a        tests/backend/gpu/kernels/test_dp4a_gemm.cpp)
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

if(BUILD_TESTING)
    foreach(name IN ITEMS test_response_parser test_sampler test_pc_disk test_pc_ram test_cpu_gdn test_sse_cancel)
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES LABELS "hermetic;cpu")
    endforeach()
    add_test(NAME test_video_hermetic COMMAND test_multimodal --video-only)
    set_tests_properties(test_video_hermetic PROPERTIES LABELS "hermetic;media")

    # One binary, three processes: url_fetch_mode() caches its answer in a
    # static, so each policy needs its own process.  The test sets/clears the
    # env itself, so an ambient PF_MM_URL_FETCH cannot skew the default case.
    foreach(mode default public private)
        add_test(NAME test_url_fetch_${mode} COMMAND test_url_fetch 18141 ${mode})
        set_tests_properties(test_url_fetch_${mode} PROPERTIES LABELS "hermetic;cpu")
    endforeach()

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
        foreach(name IN ITEMS test_gpu_stages test_gpu_vs_ref test_forward test_decode_vs_prefill test_pc_gpu)
            add_test(NAME ${name} COMMAND ${name} "${SI_TEST_08B_MODEL}")
            set_tests_properties(${name} PROPERTIES LABELS "gpu;model-0.8b")
        endforeach()
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
