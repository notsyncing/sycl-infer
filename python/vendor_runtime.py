#!/usr/bin/env python3
"""Populate python/sycl_infer/_lib with the engine and its runtime closure.

The wheel is meant to run on a machine that has the Intel GPU driver but not
the oneAPI toolkit.  This copies the DSOs the C API needs; the kernel driver
(Level Zero / IGC) is deliberately not bundled - it must match the host.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "sycl_infer" / "_lib"

# SONAME -> candidate directories (first existing wins)
LIBS = {
    "libsycl.so.9": ["/opt/intel/oneapi/2026.2/lib", "/opt/intel/oneapi/compiler/2026.2/lib", "/opt/intel/oneapi/2026.1/lib", "/opt/intel/oneapi/compiler/2026.1/lib"],
    "libsvml.so": ["/opt/intel/oneapi/compiler/2026.2/lib", "/opt/intel/oneapi/compiler/2026.1/lib", "/opt/intel/oneapi/2026.1/lib"],
    "libimf.so": ["/opt/intel/oneapi/compiler/2026.2/lib", "/opt/intel/oneapi/compiler/2026.1/lib", "/opt/intel/oneapi/2026.1/lib"],
    "libintlc.so.5": ["/opt/intel/oneapi/compiler/2026.2/lib", "/opt/intel/oneapi/compiler/2026.1/lib", "/opt/intel/oneapi/2026.1/lib"],
    "libirng.so": ["/opt/intel/oneapi/compiler/2026.2/lib", "/opt/intel/oneapi/compiler/2026.1/lib", "/opt/intel/oneapi/2026.1/lib"],
    "libdnnl.so.3": ["/opt/intel/oneapi/dnnl/2026.1/lib", "/opt/intel/oneapi/dnnl/2026.0/lib"],
    "libtbb.so.12": ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/opt/intel/oneapi/2026.1/lib", "/opt/intel/oneapi/tbb/2023.1/lib"],
    "libOpenCL.so.1": ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/opt/intel/oneapi/2026.1/lib", "/opt/intel/oneapi/tbb/2023.1/lib"],
    "libze_loader.so.1": ["/usr/lib/x86_64-linux-gnu", "/usr/lib64",
                           "/opt/intel/oneapi/2026.2/lib"],
}


def copy_one(name: str, dirs: list[str], dest: Path) -> bool:
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            real = p.resolve()
            shutil.copy2(real, dest / name)
            print(f"  {name:24s} <- {real}")
            return True
    print(f"  MISSING {name} (searched {', '.join(dirs)})", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(ROOT.parent / "build-rel2"),
                    help="directory holding libsycl_infer.so.1.0.0")
    args = ap.parse_args()
    build = Path(args.build)
    src = build / "libsycl_infer.so.1.0.0"
    if not src.exists():
        src = build / "libsycl_infer.so.1"
    if not src.exists():
        print(f"engine library not found under {build}", file=sys.stderr)
        return 1
    onnx_src = build / "libsycl_infer_onnx.so.1.0.0"
    if not onnx_src.exists():
        onnx_src = build / "libsycl_infer_onnx.so.1"
    if not onnx_src.exists():
        print(f"ONNX library not found under {build}", file=sys.stderr)
        return 1
    DEST.mkdir(parents=True, exist_ok=True)
    for old in DEST.glob("*.so*"):
        old.unlink()
    print(f"vendoring into {DEST}")
    real = src.resolve()
    shutil.copy2(real, DEST / "libsycl_infer.so.1.0.0")
    shutil.copy2(real, DEST / "libsycl_infer.so.1")
    print(f"  {'libsycl_infer.so':24s} <- {real}")
    onnx_real = onnx_src.resolve()
    shutil.copy2(onnx_real, DEST / "libsycl_infer_onnx.so.1.0.0")
    shutil.copy2(onnx_real, DEST / "libsycl_infer_onnx.so.1")
    print(f"  {'libsycl_infer_onnx.so':24s} <- {onnx_real}")
    ok = True
    for name, dirs in LIBS.items():
        ok = copy_one(name, dirs, DEST) and ok
    print("done" if ok else "done, with missing libraries (see above)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
