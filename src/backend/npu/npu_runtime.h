#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace si {
namespace npu {

struct arg_desc {
    std::string name;
    int kind = 0;
    uint32_t precision = 0;
    std::vector<uint32_t> dims;
    size_t bytes = 0;
};

struct graph;
using graph_ptr = std::unique_ptr<graph, void (*)(graph *)>;

bool available();
void * alloc_host(size_t bytes);
void free_host(void * p);
const char * device_name();
uint32_t graph_extension_version();
uint16_t compiler_major();
uint16_t compiler_minor();
uint32_t max_opset();

graph_ptr compile_ir(const std::string & xml, const void * weights, size_t weights_bytes, std::string & err);
graph_ptr import_blob(const void * blob, size_t bytes, std::string & err);
bool query_args(graph * g, std::vector<arg_desc> & args, std::string & err);
bool run(graph * g, const std::vector<const void *> & args, std::string & err);
bool export_blob(graph * g, std::vector<uint8_t> & blob, std::string & err);

}  // namespace npu
}  // namespace si
