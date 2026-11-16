#!/usr/bin/env python3
"""Kernel unit-test coverage gate.

Every kernel entry point declared in src/backend/gpu/kernels/kernels.h or
src/backend/cpu/cpu_types.h must have a row in ci/kernel_test_map.txt naming a
test that exercises it, or an explicit `!integration` row with a reason and a
test that covers it end to end.

A unit row is only accepted when the named test file exists and contains the
launcher's identifier: a test that merely includes the header does not count.
The gate also rejects stale rows (a kernel that no longer exists) and rows for
a test that is not registered in CMakeLists.txt.
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_HDR = os.path.join(ROOT, "src/backend/gpu/kernels/kernels.h")
CPU_HDR = os.path.join(ROOT, "src/backend/cpu/cpu_types.h")
MAP = os.path.join(ROOT, "ci/kernel_test_map.txt")
CMAKE = os.path.join(ROOT, "CMakeLists.txt")

EXTRA_GPU = {
    "gemv_mix_q8_f32_plan",
    "moe_grp_tile_rows",
    "kv_store_sum_slot",
    "kv_store_sum_ptr",
}
EXTRA_CPU = {"cpu_pc_guard"}


def declared_kernels():
    gpu_src = open(GPU_HDR, encoding="utf-8").read()
    gpu = set(re.findall(r"^\s*(?:void|bool|sycl::event|float|int|int64_t\s*\*)\s+(\w+)\s*\(", gpu_src, re.M))
    gpu = {n for n in gpu if n.endswith("_launch")} | EXTRA_GPU
    cpu_src = open(CPU_HDR, encoding="utf-8").read()
    cpu = set(re.findall(r"^\s*(?:void|bool|float|int|int32_t)\s+(cpu_\w+|q5_1_sb_pack)\s*\(", cpu_src, re.M))
    cpu |= EXTRA_CPU
    return gpu, cpu


def cmake_text():
    return open(CMAKE, encoding="utf-8").read()


def parse_map():
    rows = {}
    errors = []
    if not os.path.exists(MAP):
        return rows, [f"missing map: {os.path.relpath(MAP, ROOT)}"]
    for ln, line in enumerate(open(MAP, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            continue
        if "=" not in line:
            errors.append(f"{ln}: expected `kernel = tests`")
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if name in rows:
            errors.append(f"{ln}: duplicate row for {name}")
        rows[name] = (value, ln)
    return rows, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every row by tree")
    args = ap.parse_args()

    gpu, cpu = declared_kernels()
    rows, errors = parse_map()
    registered = cmake_text()

    def test_registered(path):
        return path in registered

    unit = integration = 0
    for tree, names in (("gpu", gpu), ("cpu", cpu)):
        for name in sorted(names):
            if name not in rows:
                errors.append(f"{name}: no row in ci/kernel_test_map.txt")
                continue
            value, ln = rows[name]
            if value.startswith("!integration"):
                m = re.match(r"!integration:\s*(.+?)\s*\|\s*(.+)$", value)
                if not m:
                    errors.append(f"{ln}: {name}: !integration needs `reason | test path`")
                    continue
                reason, paths = m.group(1).strip(), [p.strip() for p in m.group(2).split(",")]
                if len(reason) < 8:
                    errors.append(f"{ln}: {name}: integration reason is too short")
                    continue
                for p in paths:
                    if not os.path.exists(os.path.join(ROOT, p)):
                        errors.append(f"{ln}: {name}: integration test does not exist: {p}")
                    elif not test_registered(p):
                        errors.append(f"{ln}: {name}: integration test is not in CMakeLists: {p}")
                integration += 1
                if args.list:
                    print(f"{tree:3} {name:34} !integration: {reason} | {', '.join(paths)}")
                continue
            paths = [p.strip() for p in value.split(",") if p.strip()]
            if not paths:
                errors.append(f"{ln}: {name}: empty test list")
                continue
            ok = True
            for p in paths:
                full = os.path.join(ROOT, p)
                if not os.path.exists(full):
                    errors.append(f"{ln}: {name}: test does not exist: {p}")
                    ok = False
                    continue
                if not test_registered(p):
                    errors.append(f"{ln}: {name}: test is not in CMakeLists: {p}")
                    ok = False
                if name not in open(full, encoding="utf-8", errors="ignore").read():
                    errors.append(f"{ln}: {name}: {p} does not mention the launcher")
                    ok = False
            if ok:
                unit += 1
            if args.list:
                print(f"{tree:3} {name:34} {', '.join(paths)}")

    # stale rows
    known = gpu | cpu
    for name, (_, ln) in rows.items():
        if name not in known:
            errors.append(f"{ln}: {name}: row for a kernel that is not declared")

    total = len(gpu) + len(cpu)
    print(f"[kernel-coverage] {total} declared kernels: {unit} unit-tested, "
          f"{integration} integration-only, {total - unit - integration} missing")
    if errors:
        print("[kernel-coverage] FAIL")
        for e in errors:
            print("  " + e)
        return 1
    print("[kernel-coverage] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
