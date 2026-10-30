#!/usr/bin/env python3
"""Populate python/sycl_infer/_lib with the engine and its runtime closure.

The wheel is meant to run on a machine that has the Intel GPU driver but not
the oneAPI toolkit.  This copies the DSOs the C API needs; the kernel driver
(Level Zero / IGC) is deliberately not bundled - it must match the host.

When ``libsycl_infer_onnx`` was built in the same tree it is vendored too, so
the single wheel serves both the LLM C ABI and RapidDoc's in-process ONNX
executor.
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "sycl_infer" / "_lib"
ONEAPI = "/opt/intel/oneapi"


def _dirs(*patterns: str) -> list[str]:
    """Expand versioned oneAPI paths, newest version first."""
    out: list[str] = []
    for pattern in patterns:
        out.extend(sorted(glob.glob(pattern), reverse=True))
    return out


# SONAME -> candidate directories (first existing wins)
LIBS = {
    "libsycl.so.9": _dirs(f"{ONEAPI}/compiler/*/lib", f"{ONEAPI}/*/lib"),
    "libsvml.so": _dirs(f"{ONEAPI}/compiler/*/lib"),
    "libimf.so": _dirs(f"{ONEAPI}/compiler/*/lib"),
    "libintlc.so.5": _dirs(f"{ONEAPI}/compiler/*/lib"),
    "libirng.so": _dirs(f"{ONEAPI}/compiler/*/lib"),
    "libdnnl.so.3": _dirs(f"{ONEAPI}/dnnl/*/lib"),
    "libtbb.so.12": ["/usr/lib/x86_64-linux-gnu", "/usr/lib64",
                     *_dirs(f"{ONEAPI}/tbb/*/lib")],
    "libOpenCL.so.1": ["/usr/lib/x86_64-linux-gnu", "/usr/lib64",
                       *_dirs(f"{ONEAPI}/*/lib")],
    "libze_loader.so.1": ["/usr/lib/x86_64-linux-gnu", "/usr/lib64",
                           *_dirs(f"{ONEAPI}/*/lib")],
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
    ap.add_argument("--dnnl-root", default="",
                    help="oneDNN prefix whose libdnnl.so.3 must be bundled "
                         "(default: newest /opt/intel/oneapi/dnnl/*)")
    args = ap.parse_args()
    build = Path(args.build)
    libs = dict(LIBS)
    if args.dnnl_root:
        libs["libdnnl.so.3"] = [str(Path(args.dnnl_root) / "lib"),
                                *LIBS["libdnnl.so.3"]]
    src = build / "libsycl_infer.so.1.0.0"
    if not src.exists():
        src = build / "libsycl_infer.so.1"
    if not src.exists():
        print(f"engine library not found under {build}", file=sys.stderr)
        return 1
    DEST.mkdir(parents=True, exist_ok=True)
    for old in DEST.glob("*.so*"):
        old.unlink()
    print(f"vendoring into {DEST}")
    real = src.resolve()
    shutil.copy2(real, DEST / "libsycl_infer.so.1.0.0")
    shutil.copy2(real, DEST / "libsycl_infer.so.1")
    print(f"  {'libsycl_infer.so':24s} <- {real}")
    onnx = build / "libsycl_infer_onnx.so.1.0.0"
    if not onnx.exists():
        onnx = build / "libsycl_infer_onnx.so.1"
    if not onnx.exists():
        print(f"ONNX library not found under {build}", file=sys.stderr)
        return 1
    onnx_real = onnx.resolve()
    shutil.copy2(onnx_real, DEST / "libsycl_infer_onnx.so.1.0.0")
    shutil.copy2(onnx_real, DEST / "libsycl_infer_onnx.so.1")
    print(f"  {'libsycl_infer_onnx.so':24s} <- {onnx_real}")
    ok = True
    for name, dirs in libs.items():
        ok = copy_one(name, dirs, DEST) and ok
    print("done" if ok else "done, with missing libraries (see above)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
