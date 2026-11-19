#include "npu_runtime.h"

#include <dlfcn.h>
#include <level_zero/ze_api.h>
#include <ze_driver_npu_ext.h>
#include <ze_graph_ext.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace si {
namespace npu {

namespace {

struct ze_api {
    void * lib = nullptr;
    decltype(&zeInitDrivers) init_drivers = nullptr;
    decltype(&zeInit) init = nullptr;
    decltype(&zeDriverGet) driver_get = nullptr;
    decltype(&zeDriverGetProperties) driver_props = nullptr;
    decltype(&zeDriverGetExtensionProperties) driver_exts = nullptr;
    decltype(&zeDriverGetExtensionFunctionAddress) driver_ext_addr = nullptr;
    decltype(&zeDeviceGet) device_get = nullptr;
    decltype(&zeDeviceGetProperties) device_props = nullptr;
    decltype(&zeContextCreate) context_create = nullptr;
    decltype(&zeContextDestroy) context_destroy = nullptr;
    decltype(&zeDeviceGetCommandQueueGroupProperties) queue_groups = nullptr;
    decltype(&zeMemAllocHost) mem_alloc_host = nullptr;
    decltype(&zeMemFree) mem_free = nullptr;
    decltype(&zeCommandQueueCreate) queue_create = nullptr;
    decltype(&zeCommandQueueDestroy) queue_destroy = nullptr;
    decltype(&zeCommandQueueExecuteCommandLists) queue_execute = nullptr;
    decltype(&zeCommandListCreate) list_create = nullptr;
    decltype(&zeCommandListClose) list_close = nullptr;
    decltype(&zeCommandListDestroy) list_destroy = nullptr;
    decltype(&zeFenceCreate) fence_create = nullptr;
    decltype(&zeFenceDestroy) fence_destroy = nullptr;
    decltype(&zeFenceHostSynchronize) fence_sync = nullptr;
    decltype(&zeFenceReset) fence_reset = nullptr;
    decltype(&zeCommandListReset) list_reset = nullptr;
};

template <class T>
bool load_sym(void * lib, const char * name, T & dst) {
    void * p = dlsym(lib, name);
    if (!p) return false;
    std::memcpy(&dst, &p, sizeof(p));
    return true;
}

struct runtime {
    ze_api ze;
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    ze_context_handle_t context = nullptr;
    ze_graph_dditable_ext_t graph = {};
    uint32_t graph_version = 0;
    uint16_t compiler_major_v = 0;
    uint16_t compiler_minor_v = 0;
    uint32_t max_opset_v = 0;
    char name[ZE_MAX_DEVICE_NAME] = {};
    bool ok = false;
    std::string error;

    bool init() {
        if (ok || !error.empty()) return ok;
        ze.lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!ze.lib) {
            error = std::string("libze_loader.so.1: ") + dlerror();
            return false;
        }
        bool loaded = load_sym(ze.lib, "zeInitDrivers", ze.init_drivers) &&
                      load_sym(ze.lib, "zeDriverGetProperties", ze.driver_props) &&
                      load_sym(ze.lib, "zeDriverGetExtensionProperties", ze.driver_exts) &&
                      load_sym(ze.lib, "zeDriverGetExtensionFunctionAddress", ze.driver_ext_addr) &&
                      load_sym(ze.lib, "zeDeviceGet", ze.device_get) &&
                      load_sym(ze.lib, "zeDeviceGetProperties", ze.device_props) &&
                      load_sym(ze.lib, "zeContextCreate", ze.context_create) &&
                      load_sym(ze.lib, "zeContextDestroy", ze.context_destroy) &&
                      load_sym(ze.lib, "zeDeviceGetCommandQueueGroupProperties", ze.queue_groups) &&
                      load_sym(ze.lib, "zeMemAllocHost", ze.mem_alloc_host) &&
                      load_sym(ze.lib, "zeMemFree", ze.mem_free) &&
                      load_sym(ze.lib, "zeCommandQueueCreate", ze.queue_create) &&
                      load_sym(ze.lib, "zeCommandQueueDestroy", ze.queue_destroy) &&
                      load_sym(ze.lib, "zeCommandQueueExecuteCommandLists", ze.queue_execute) &&
                      load_sym(ze.lib, "zeCommandListCreate", ze.list_create) &&
                      load_sym(ze.lib, "zeCommandListClose", ze.list_close) &&
                      load_sym(ze.lib, "zeCommandListDestroy", ze.list_destroy) &&
                      load_sym(ze.lib, "zeFenceCreate", ze.fence_create) &&
                      load_sym(ze.lib, "zeFenceDestroy", ze.fence_destroy) &&
                      load_sym(ze.lib, "zeFenceHostSynchronize", ze.fence_sync) &&
                      load_sym(ze.lib, "zeFenceReset", ze.fence_reset) &&
                      load_sym(ze.lib, "zeCommandListReset", ze.list_reset);
        if (!loaded) {
            error = "Level Zero loader is missing a required core export";
            return false;
        }
        if (!ze.init_drivers) {
            if (!load_sym(ze.lib, "zeInit", ze.init) || !load_sym(ze.lib, "zeDriverGet", ze.driver_get)) {
                error = "Level Zero loader has neither zeInitDrivers nor zeInit/zeDriverGet";
                return false;
            }
        }
        ze_init_driver_type_desc_t idesc = {};
        idesc.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
        idesc.flags = ZE_INIT_DRIVER_TYPE_FLAG_NPU;
        uint32_t ndrv = 0;
        ze_result_t r;
        if (ze.init_drivers) {
            r = ze.init_drivers(&ndrv, nullptr, &idesc);
        } else {
            r = ze.init(ZE_INIT_FLAG_VPU_ONLY);
            if (r == ZE_RESULT_SUCCESS && ze.driver_get) r = ze.driver_get(&ndrv, nullptr);
        }
        if (r != ZE_RESULT_SUCCESS || ndrv == 0) {
            error = "no Level Zero NPU driver was found";
            return false;
        }
        std::vector<ze_driver_handle_t> drivers(ndrv);
        r = ze.init_drivers ? ze.init_drivers(&ndrv, drivers.data(), &idesc)
                            : ze.driver_get(&ndrv, drivers.data());
        if (r != ZE_RESULT_SUCCESS) {
            error = "zeInitDrivers(list) failed";
            return false;
        }
        for (ze_driver_handle_t d : drivers) {
            uint32_t ndev = 0;
            if (ze.device_get(d, &ndev, nullptr) != ZE_RESULT_SUCCESS || ndev == 0) continue;
            std::vector<ze_device_handle_t> devices(ndev);
            if (ze.device_get(d, &ndev, devices.data()) != ZE_RESULT_SUCCESS) continue;
            for (ze_device_handle_t dev : devices) {
                ze_device_properties_t p = {};
                p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                if (ze.device_props(dev, &p) != ZE_RESULT_SUCCESS) continue;
                if (p.type != ZE_DEVICE_TYPE_VPU) continue;
                driver = d;
                device = dev;
                std::memcpy(name, p.name, sizeof(name) - 1);
                break;
            }
            if (device) break;
        }
        if (!device) {
            error = "no Level Zero VPU device was enumerated";
            return false;
        }
        uint32_t next = 0;
        if (ze.driver_exts(driver, &next, nullptr) != ZE_RESULT_SUCCESS) {
            error = "zeDriverGetExtensionProperties failed";
            return false;
        }
        std::vector<ze_driver_extension_properties_t> exts(next);
        if (next && ze.driver_exts(driver, &next, exts.data()) != ZE_RESULT_SUCCESS) {
            error = "zeDriverGetExtensionProperties(list) failed";
            return false;
        }
        for (const auto & e : exts) {
            if (std::strcmp(e.name, ZE_GRAPH_EXT_NAME) == 0) graph_version = e.version;
        }
        if (!graph_version) {
            error = "NPU driver does not advertise ZE_extension_graph";
            return false;
        }
        void * raw = nullptr;
        if (ze.driver_ext_addr(driver, ZE_DRIVER_NPU_EXT_NAME, &raw) != ZE_RESULT_SUCCESS || !raw) {
            error = "ZE_extension_driver_npu is unavailable";
            return false;
        }
        ze_driver_npu_dditable_ext_t * ddi = static_cast<ze_driver_npu_dditable_ext_t *>(raw);
        if (!ddi->pfnGetExtension) {
            error = "NPU driver extension pfnGetExtension is missing";
            return false;
        }
        ze_driver_extension_npu_ext_t req = {};
        req.stype = ZE_STRUCTURE_TYPE_DRIVER_EXTENSION_NPU_EXT;
        req.name = ZE_GRAPH_EXT_NAME;
        req.version = graph_version;
        void * table = nullptr;
        req.ppFunctionAddress = &table;
        r = ddi->pfnGetExtension(driver, &req);
        if (r != ZE_RESULT_SUCCESS || !table) {
            error = "ZE_extension_graph table is unavailable";
            return false;
        }
        std::memcpy(&graph, table, sizeof(graph));
        if (!graph.pfnCreate2 || !graph.pfnDestroy || !graph.pfnDeviceGetGraphProperties2 ||
            !graph.pfnGetArgumentProperties3 || !graph.pfnSetArgumentValue || !graph.pfnAppendGraphExecute ||
            !graph.pfnGetProperties3) {
            error = "NPU graph extension is missing a required entry point";
            return false;
        }
        ze_device_graph_properties_2_t p = {};
        p.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES_2;
        r = graph.pfnDeviceGetGraphProperties2(device, &p);
        if (r != ZE_RESULT_SUCCESS) {
            error = "pfnDeviceGetGraphProperties2 failed";
            return false;
        }
        if ((p.graphFormatsSupported & ZE_GRAPH_FORMAT_NGRAPH_LITE) == 0) {
            error = "NPU does not accept NGRAPH_LITE graphs";
            return false;
        }
        compiler_major_v = p.compilerVersion.major;
        compiler_minor_v = p.compilerVersion.minor;
        max_opset_v = p.maxOVOpsetVersionSupported;
        ze_context_desc_t cd = {};
        cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        r = ze.context_create(driver, &cd, &context);
        if (r != ZE_RESULT_SUCCESS) {
            error = "zeContextCreate failed";
            return false;
        }
        ok = true;
        return true;
    }
};

runtime & rt() {
    static runtime r;
    return r;
}

std::mutex & rt_mutex() {
    static std::mutex m;
    return m;
}

ze_result_t find_compute_ordinal(uint32_t & ordinal, std::string & err) {
    runtime & r = rt();
    uint32_t n = 0;
    ze_result_t res = r.ze.queue_groups(r.device, &n, nullptr);
    if (res != ZE_RESULT_SUCCESS) {
        err = "zeDeviceGetCommandQueueGroupProperties failed";
        return res;
    }
    std::vector<ze_command_queue_group_properties_t> groups(n);
    for (auto & g : groups) g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    res = r.ze.queue_groups(r.device, &n, groups.data());
    if (res != ZE_RESULT_SUCCESS) return res;
    for (uint32_t i = 0; i < n; ++i) {
        if (groups[i].numQueues != 0 && (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0) {
            ordinal = i;
            return ZE_RESULT_SUCCESS;
        }
    }
    err = "NPU exposes no compute command queue group";
    return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

}  // namespace

struct graph {
    ze_graph_handle_t handle = nullptr;
    std::vector<void *> allocations;
    bool initialized = false;
    ze_command_queue_handle_t queue = nullptr;
    ze_command_list_handle_t list = nullptr;
    ze_fence_handle_t fence = nullptr;
    uint32_t ordinal = 0;
    ~graph() {
        runtime & r = rt();
        if (fence) r.ze.fence_destroy(fence);
        if (list) r.ze.list_destroy(list);
        if (queue) r.ze.queue_destroy(queue);
        if (handle) r.graph.pfnDestroy(handle);
        for (void * p : allocations) r.ze.mem_free(r.context, p);
    }
};

bool available() {
    std::lock_guard<std::mutex> lock(rt_mutex());
    return rt().init();
}

void * alloc_host(size_t bytes) {
    std::lock_guard<std::mutex> lock(rt_mutex());
    runtime & r = rt();
    if (!r.init()) return nullptr;
    ze_host_mem_alloc_desc_t d = {};
    d.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void * p = nullptr;
    if (r.ze.mem_alloc_host(r.context, &d, bytes, 64, &p) != ZE_RESULT_SUCCESS) return nullptr;
    return p;
}

void free_host(void * p) {
    if (!p) return;
    std::lock_guard<std::mutex> lock(rt_mutex());
    runtime & r = rt();
    if (r.ok) r.ze.mem_free(r.context, p);
}

const char * device_name() {
    std::lock_guard<std::mutex> lock(rt_mutex());
    if (!rt().init()) return "unavailable";
    return rt().name;
}

uint32_t graph_extension_version() {
    std::lock_guard<std::mutex> lock(rt_mutex());
    if (!rt().init()) return 0;
    return rt().graph_version;
}

uint16_t compiler_major() {
    std::lock_guard<std::mutex> lock(rt_mutex());
    if (!rt().init()) return 0;
    return rt().compiler_major_v;
}

uint16_t compiler_minor() {
    std::lock_guard<std::mutex> lock(rt_mutex());
    if (!rt().init()) return 0;
    return rt().compiler_minor_v;
}

uint32_t max_opset() {
    std::lock_guard<std::mutex> lock(rt_mutex());
    if (!rt().init()) return 0;
    return rt().max_opset_v;
}

static graph_ptr create(std::vector<uint8_t> input, ze_graph_format_t format, const char * flags, std::string & err) {
    std::lock_guard<std::mutex> lock(rt_mutex());
    runtime & r = rt();
    if (!r.init()) {
        err = r.error;
        return graph_ptr(nullptr, [](graph *) {});
    }
    ze_graph_desc_2_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_2;
    desc.format = format;
    desc.inputSize = input.size();
    desc.pInput = input.data();
    desc.pBuildFlags = flags;
    ze_graph_handle_t h = nullptr;
    ze_result_t res = r.graph.pfnCreate2(r.context, r.device, &desc, &h);
    if (res != ZE_RESULT_SUCCESS || !h) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "pfnCreate2 returned 0x%08x", static_cast<uint32_t>(res));
        err = buf;
        if (r.graph.pfnBuildLogGetString) {
            uint32_t n = 0;
            if (r.graph.pfnBuildLogGetString(h, &n, nullptr) == ZE_RESULT_SUCCESS && n > 1) {
                std::vector<char> log(n);
                if (r.graph.pfnBuildLogGetString(h, &n, log.data()) == ZE_RESULT_SUCCESS) {
                    err += std::string(": ") + log.data();
                }
            }
        }
        return graph_ptr(nullptr, [](graph *) {});
    }
    graph * g = new graph();
    g->handle = h;
    return graph_ptr(g, [](graph * p) { delete p; });
}

graph_ptr compile_ir(const std::string & xml, const void * weights, size_t weights_bytes, std::string & err) {
    std::unique_lock<std::mutex> lock(rt_mutex());
    runtime & r = rt();
    if (!r.init()) {
        err = r.error;
        return graph_ptr(nullptr, [](graph *) {});
    }
    std::vector<uint8_t> input;
    input.reserve(24 + xml.size() + weights_bytes);
    auto put16 = [&](uint16_t v) {
        input.push_back(static_cast<uint8_t>(v));
        input.push_back(static_cast<uint8_t>(v >> 8));
    };
    auto put32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) input.push_back(static_cast<uint8_t>(v >> (8 * i)));
    };
    auto put64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) input.push_back(static_cast<uint8_t>(v >> (8 * i)));
    };
    put16(r.compiler_major_v);
    put16(r.compiler_minor_v);
    put32(2);
    put64(xml.size());
    input.insert(input.end(), xml.begin(), xml.end());
    put64(weights_bytes);
    if (weights_bytes) {
        const uint8_t * p = static_cast<const uint8_t *>(weights);
        input.insert(input.end(), p, p + weights_bytes);
    }
    const char * flags = std::getenv("PF_NPU_BUILD_FLAGS");
    std::string build_flags = flags ? flags : "";
    lock.unlock();
    return create(std::move(input), ZE_GRAPH_FORMAT_NGRAPH_LITE, build_flags.c_str(), err);
}

graph_ptr import_blob(const void * blob, size_t bytes, std::string & err) {
    const uint8_t * p = static_cast<const uint8_t *>(blob);
    return create(std::vector<uint8_t>(p, p + bytes), ZE_GRAPH_FORMAT_NATIVE, "", err);
}

bool query_args(graph * g, std::vector<arg_desc> & args, std::string & err) {
    if (!g || !g->handle) {
        err = "invalid graph";
        return false;
    }
    runtime & r = rt();
    ze_graph_properties_3_t p = {};
    p.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3;
    ze_result_t res = r.graph.pfnGetProperties3(g->handle, &p);
    if (res != ZE_RESULT_SUCCESS) {
        err = "pfnGetProperties3 failed";
        return false;
    }
    args.clear();
    for (uint32_t i = 0; i < p.numGraphArgs; ++i) {
        ze_graph_argument_properties_3_t a = {};
        a.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;
        res = r.graph.pfnGetArgumentProperties3(g->handle, i, &a);
        if (res != ZE_RESULT_SUCCESS) {
            err = "pfnGetArgumentProperties3 failed";
            return false;
        }
        arg_desc d;
        d.name = a.name;
        d.kind = a.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT ? 0 : 1;
        d.precision = a.devicePrecision;
        size_t count = 1;
        uint32_t dims = a.dims_count > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE ? ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE
                                                                             : a.dims_count;
        for (uint32_t j = 0; j < dims; ++j) {
            d.dims.push_back(a.dims[j]);
            count *= a.dims[j] ? a.dims[j] : 1;
        }
        size_t esz = 4;
        switch (a.devicePrecision) {
            case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
            case ZE_GRAPH_ARGUMENT_PRECISION_UINT8: esz = 1; break;
            case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
            case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
            case ZE_GRAPH_ARGUMENT_PRECISION_UINT16: esz = 2; break;
            case ZE_GRAPH_ARGUMENT_PRECISION_INT64:
            case ZE_GRAPH_ARGUMENT_PRECISION_UINT64: esz = 8; break;
            default: break;
        }
        d.bytes = count * esz;
        args.push_back(std::move(d));
    }
    return true;
}

bool run(graph * g, const std::vector<const void *> & args, std::string & err) {
    if (!g || !g->handle) {
        err = "invalid graph";
        return false;
    }
    runtime & r = rt();
    if (!g->queue) {
        if (find_compute_ordinal(g->ordinal, err) != ZE_RESULT_SUCCESS) return false;
        ze_command_queue_desc_t qd = {};
        qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        qd.ordinal = g->ordinal;
        ze_command_list_desc_t ld = {};
        ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
        ld.commandQueueGroupOrdinal = g->ordinal;
        ze_fence_desc_t fd = {};
        fd.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
        ze_result_t cres = r.ze.queue_create(r.context, r.device, &qd, &g->queue);
        if (cres != ZE_RESULT_SUCCESS) {
            err = "zeCommandQueueCreate failed";
            return false;
        }
        cres = r.ze.list_create(r.context, r.device, &ld, &g->list);
        if (cres != ZE_RESULT_SUCCESS) {
            err = "zeCommandListCreate failed";
            return false;
        }
        cres = r.ze.fence_create(g->queue, &fd, &g->fence);
        if (cres != ZE_RESULT_SUCCESS) {
            err = "zeFenceCreate failed";
            return false;
        }
    }
    ze_result_t res;
    if (g->initialized && r.ze.list_reset) {
        res = r.ze.list_reset(g->list);
        if (res != ZE_RESULT_SUCCESS) {
            err = "zeCommandListReset failed";
            return false;
        }
    }
    if (g->initialized && r.ze.fence_reset) {
        res = r.ze.fence_reset(g->fence);
        if (res != ZE_RESULT_SUCCESS) {
            err = "zeFenceReset failed";
            return false;
        }
    }
    bool ok = true;
    for (uint32_t i = 0; i < args.size() && ok; ++i) {
        if (!args[i]) continue;
        res = r.graph.pfnSetArgumentValue(g->handle, i, args[i]);
        if (res != ZE_RESULT_SUCCESS) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "pfnSetArgumentValue(%u) returned 0x%08x", i, static_cast<uint32_t>(res));
            err = buf;
            ok = false;
        }
    }
    if (ok && !g->initialized) {
        ze_graph_properties_3_t p = {};
        p.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3;
        if (r.graph.pfnGetProperties3(g->handle, &p) == ZE_RESULT_SUCCESS &&
            p.initStageRequired == ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE) {
            res = r.graph.pfnAppendGraphInitialize(g->list, g->handle, nullptr, 0, nullptr);
            if (res != ZE_RESULT_SUCCESS) {
                err = "pfnAppendGraphInitialize failed";
                ok = false;
            }
        } else if (r.graph.pfnGraphInitialize) {
            res = r.graph.pfnGraphInitialize(g->handle);
            if (res != ZE_RESULT_SUCCESS) {
                err = "pfnGraphInitialize failed";
                ok = false;
            }
        }
    }
    if (ok) {
        res = r.graph.pfnAppendGraphExecute(g->list, g->handle, nullptr, nullptr, 0, nullptr);
        if (res != ZE_RESULT_SUCCESS) {
            err = "pfnAppendGraphExecute failed";
            ok = false;
        }
    }
    if (ok) {
        res = r.ze.list_close(g->list);
        if (res != ZE_RESULT_SUCCESS) {
            err = "zeCommandListClose failed";
            ok = false;
        }
    }
    if (ok) {
        res = r.ze.queue_execute(g->queue, 1, &g->list, g->fence);
        if (res != ZE_RESULT_SUCCESS) {
            err = "zeCommandQueueExecuteCommandLists failed";
            ok = false;
        }
    }
    if (ok) {
        res = r.ze.fence_sync(g->fence, 5000000000ull);
        if (res != ZE_RESULT_SUCCESS) {
            err = "zeFenceHostSynchronize failed";
            ok = false;
        } else {
            g->initialized = true;
        }
    }
    return ok;
}

bool export_blob(graph * g, std::vector<uint8_t> & blob, std::string & err) {
    if (!g || !g->handle) {
        err = "invalid graph";
        return false;
    }
    runtime & r = rt();
    size_t n = 0;
    ze_result_t res = r.graph.pfnGetNativeBinary(g->handle, &n, nullptr);
    if (res != ZE_RESULT_SUCCESS || n == 0) {
        err = "pfnGetNativeBinary failed";
        return false;
    }
    blob.resize(n);
    res = r.graph.pfnGetNativeBinary(g->handle, &n, blob.data());
    if (res != ZE_RESULT_SUCCESS) {
        err = "pfnGetNativeBinary(data) failed";
        return false;
    }
    blob.resize(n);
    return true;
}

}  // namespace npu
}  // namespace si
