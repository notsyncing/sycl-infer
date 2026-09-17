# Third-party notices

sycl-infer itself is distributed under the MIT license (see [`LICENSE`](LICENSE)).
It bundles the following third-party components under `third_party/`; each is
used unmodified unless noted.  The full license text for every component is
reproduced in the files listed below.

| component | vendored as | license | license text |
|---|---|---|---|
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) v0.56.0 | `third_party/httplib.h` | MIT | [`third_party/LICENSE-httplib.txt`](third_party/LICENSE-httplib.txt) |
| [JSON for Modern C++](https://github.com/nlohmann/json) v3.12.0 | `third_party/json.hpp`, `third_party/nlohmann/json.hpp` | MIT | [`third_party/LICENSE-nlohmann-json.txt`](third_party/LICENSE-nlohmann-json.txt) |
| [minja](https://github.com/google/minja) | `third_party/minja/` | MIT | [`third_party/minja/LICENSE`](third_party/minja/LICENSE) |
| [stb_image](https://github.com/nothings/stb) v2.30 | `third_party/stb/stb_image.h` | public domain (MIT alternative) | header text (dual-licensed) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) Unicode tables | `third_party/unicode.{h,cpp}`, `third_party/unicode-data.{h,cpp}` | MIT | [`third_party/LICENSE-llama.cpp-unicode.txt`](third_party/LICENSE-llama.cpp-unicode.txt) |

## Notes

* `third_party/nlohmann/json.hpp` is a one-line include shim that forwards to
  `third_party/json.hpp`; it is covered by the nlohmann/json license above.
* The vendored `nlohmann/json` single header additionally embeds MIT-licensed
  code by Florian Loitsch (Grisu2) and Bjoern Hoehrmann (UTF-8 DFA decoder).
  Those notices are reproduced in
  [`third_party/LICENSE-nlohmann-json.txt`](third_party/LICENSE-nlohmann-json.txt).
* [`oneDNN`](https://github.com/oneapi-src/oneDNN) is an *optional external*
  build/run-time dependency (linked when `PF_GEMM_DNNL` is enabled); it is not
  vendored here.  It is distributed under the Apache License 2.0.
* The oneAPI DPC++/SYCL runtime and compiler are likewise external dependencies
  and are not vendored.

## Adding a new third-party component

1. Place the license text at `third_party/LICENSE-<component>.txt` (or keep the
   upstream `LICENSE` inside the component's own subdirectory).
2. Add a row to the table above with the upstream URL and license.
3. Keep the copyright notice intact in the vendored source files.
