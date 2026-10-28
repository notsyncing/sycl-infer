"""Setuptools shim: the wheel is Python-only but carries native DSOs, so it
must not be tagged ``none-any``.  A release build on an old glibc (or in a
manylinux_2_28 container) will produce the corresponding manylinux tag; this
source checkout produces ``linux_x86_64``, which is honest for the host.
"""
import sysconfig

from setuptools import setup

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False
            self.python_tag = "py3"
            self.abi_tag = "none"
            self.plat_name = sysconfig.get_platform().replace("-", "_").replace(".", "_")

        def get_tag(self):
            return "py3", "none", self.plat_name

    setup(cmdclass={"bdist_wheel": bdist_wheel})
except Exception:
    setup()
