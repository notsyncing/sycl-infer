# sycl-infer Python binding

This wheel carries the sycl-infer C library and its SYCL/oneDNN runtime
closure.  The host must provide the Intel GPU kernel driver (Level Zero / IGC).

```python
from sycl_infer import Model
with Model("model.gguf", device="gpu") as m:
    print(m.generate("Hello", max_tokens=32).text)
```

`python -m sycl_infer.doctor` prints the loaded library and ABI.  A source
checkout can use `SYCL_INFER_LIB=/path/to/libsycl_infer.so.1.0.0`.
