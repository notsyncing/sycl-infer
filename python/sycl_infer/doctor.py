"""Print the loaded sycl-infer runtime and its ABI.

    python -m sycl_infer.doctor
    python -m sycl_infer.doctor --model model.gguf --device cpu
"""
from __future__ import annotations

import argparse
import platform
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="sycl-infer runtime doctor")
    ap.add_argument("--model", help="optionally load a GGUF and print its info")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    args = ap.parse_args(argv)

    from . import _capi

    print(f"python      : {sys.version.split()[0]} ({platform.machine()})")
    try:
        lib = _capi.load()
    except Exception as exc:
        print(f"sycl-infer  : NOT LOADED ({exc})")
        print("hint        : set SYCL_INFER_LIB=/path/to/libsycl_infer.so.1.0.0")
        return 1
    print(f"sycl-infer  : {_capi.lib_path()}")
    print(f"version     : {lib.syi_version_string().decode()}")
    print(f"abi         : {lib.syi_abi_version()}")
    if args.model:
        try:
            from . import Model
            with Model(args.model, device=args.device, max_seq=128, max_slots=1,
                       kv_cap_mb=64, prefix_cache=False) as m:
                print(f"model       : {m.info}")
        except Exception as exc:
            print(f"model       : FAILED ({exc})")
            return 1
    print("status      : ok")
    print("note        : the Intel GPU driver/Level Zero loader is a system component;")
    print("              the wheel carries the SYCL/oneDNN runtime, not the kernel driver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
