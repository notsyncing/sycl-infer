"""Setuptools shim: the wheel is Python-only but carries native DSOs, so it
must not be tagged ``none-any``.  A release build on an old glibc (or in a
manylinux_2_28 container) will produce the corresponding manylinux tag; this
source checkout produces ``linux_x86_64``, which is honest for the host.
"""
import os
import sysconfig
from pathlib import Path

from setuptools import setup

try:
    # setuptools >= 70 vendors the command here; the old wheel package is a
    # fallback for builds that still drive the standalone bdist_wheel.
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - old setuptools
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        return "py3", "none", plat


BASE_VERSION = Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()
VERSION = os.environ.get("SYCL_INFER_VERSION", BASE_VERSION)

setup(version=VERSION, cmdclass={"bdist_wheel": bdist_wheel})
