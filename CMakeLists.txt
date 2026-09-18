cmake_minimum_required(VERSION 3.20)
project(sycl-infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

option(SYCL_INFER_AOT "AOT compile for Intel Iris Xe (adl-p)" OFF)
if(SYCL_INFER_AOT)
    add_compile_options(-fsycl-targets=spir64_gen -Xs "-device adl-p")
    add_link_options(-fsycl-targets=spir64_gen -Xs "-device adl-p")
endif()

# ---------------------------------------------------------------- core library
add_library(sycl_infer_core STATIC
    third_party/unicode.cpp
    third_party/unicode-data.cpp
    # common: quantization formats and shared low-level helpers
    src/common/w8.cpp
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
    src/backend/gpu/kernels/qk_norm_rope.cpp
    src/backend/gpu/kernels/attn.cpp
    src/backend/gpu/kernels/conv.cpp
    src/backend/gpu/kernels/gdn.cpp
    src/backend/gpu/kernels/gated_norm.cpp
    src/backend/gpu/kernels/xq.cpp
    src/backend/gpu/kernels/dp4a_common.cpp
    src/backend/gpu/kernels/dp4a_gemv.cpp
    src/backend/gpu/kernels/dp4a_gemm.cpp
    src/backend/gpu/kernels/vit.cpp
    # model: GGUF loading, weights, tokenizer
    src/model/gguf.cpp
    src/model/model.cpp
    src/model/model_w8.cpp
    src/model/qwen35.cpp
    src/model/tokenizer.cpp
    # mm: multimodal image preprocessing + vision encoder
    src/mm/image.cpp
    src/mm/vision.cpp
    src/mm/multimodal.cpp
    # engine: forward execution, graph capture, KV pool, prefix cache
    src/engine/engine.cpp
    src/engine/engine_graph.cpp
    src/engine/engine_kvpool.cpp
    src/engine/engine_prefix_cache.cpp
    src/engine/pc_disk.cpp
    src/engine/pc_ram.cpp
    src/engine/sampler.cpp
    # server: chat template, continuous batching scheduler, HTTP API
    src/server/chat.cpp
    src/server/chat_template.cpp
    src/server/scheduler.cpp
    src/server/server.cpp
)
target_include_directories(sycl_infer_core PUBLIC
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
set_source_files_properties(${SI_CPU_SOURCES} PROPERTIES COMPILE_OPTIONS "-fno-sycl")

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

si_add_test(test_tokenizer   tests/model/test_tokenizer.cpp)
si_add_test(test_chat_template tests/model/test_chat_template.cpp)
si_add_test(test_multimodal  tests/mm/test_multimodal.cpp)
si_add_test(test_compare     tests/model/test_compare.cpp)
si_add_test(test_pc_disk     tests/backend/cpu/test_pc_disk.cpp)
si_add_test(test_pc_ram      tests/backend/cpu/test_pc_ram.cpp)
si_add_test(test_cpuref      tests/backend/cpu/test_cpuref.cpp)
si_add_test(test_pc_cpu      tests/backend/cpu/test_pc_cpu.cpp)
si_add_test(test_cpu_gemv    tests/backend/cpu/kernels/test_cpu_gemv.cpp)
si_add_test(test_forward     tests/backend/gpu/test_forward.cpp)
si_add_test(test_gpu_vs_ref  tests/backend/gpu/test_gpu_vs_ref.cpp)
si_add_test(test_pc_gpu      tests/backend/gpu/test_pc_gpu.cpp)
si_add_test(test_pc_ram_gpu  tests/backend/gpu/test_pc_ram_gpu.cpp)
si_add_test(test_gemv        tests/backend/gpu/kernels/test_gemv.cpp)
si_add_test(test_dp4a        tests/backend/gpu/kernels/test_dp4a_gemm.cpp)

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
