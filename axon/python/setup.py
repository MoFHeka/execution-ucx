import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BazelExtension(Extension):
    def __init__(self, name, target):
        # We don't rely on setuptools to compile any C++ sources.
        super().__init__(name, sources=[])
        self.target = target


class BazelBuildExt(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, BazelExtension):
            super().build_extension(ext)
            return

        # Ensure bazel is available
        try:
            subprocess.check_call(
                ["bazel", "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            raise RuntimeError("Bazel must be installed to build this extension.")

        # Determine workspace root (parent of axon/python)
        ext_dir = Path(__file__).resolve().parent
        workspace_root = ext_dir.parent.parent

        # Targets to build: the shared object and the UCX libs
        targets = [
            "//axon/python:axon_python_lib",
            "//axon/python:copy_ucx_core_libs",
            "//axon/python:copy_ucx_transport_libs",
        ]

        # Run Bazel build
        print(f"Building {ext.target} and UCX libs with Bazel...")
        subprocess.check_call(["bazel", "build"] + targets, cwd=workspace_root)

        # Bazel puts the outputs in bazel-bin/axon/python/axon/
        bazel_bin = workspace_root / "bazel-bin" / "axon" / "python" / "axon"

        # Target directory in build_lib
        # ext.name is 'axon.axon' (translates to path `axon/axon.so`)
        ext_dest_path = Path(self.get_ext_fullpath(ext.name))
        ext_dest_dir = ext_dest_path.parent
        ext_dest_dir.mkdir(parents=True, exist_ok=True)

        # In develop mode, pip install -e . might call this, but typically pip install -e .
        # puts an egg-link and uses the source directory. Setuptools handles that, but for extension
        # it compiles it to inplace if setup.py build_ext --inplace is used.
        # For editable install, PEP 660 is used by setuptools.

        # Also copy to the source tree directly for editable installs
        source_ext_dir = ext_dir / "axon"

        # Copy _axon.so to setuptools build dir
        axon_so_src = bazel_bin / "_axon.so"
        if axon_so_src.exists():
            print(f"Copying {axon_so_src} to {ext_dest_path}")
            if ext_dest_path.exists():
                ext_dest_path.unlink()
            shutil.copy2(axon_so_src, ext_dest_path)
        else:
            raise RuntimeError(f"Expected {axon_so_src} not found after bazel build.")

        # Copy libs/ directory
        libs_src_dir = bazel_bin / "libs"
        libs_dest_dir = ext_dest_dir / "libs"

        if libs_src_dir.exists() and libs_src_dir.is_dir():
            # Create dest dirs if not exists
            libs_dest_dir.mkdir(parents=True, exist_ok=True)

            print(f"Copying UCX libraries from {libs_src_dir} to {libs_dest_dir}")
            for so_file in libs_src_dir.glob("*.so*"):
                # Copy to setuptools build dir
                dest_file = libs_dest_dir / so_file.name
                if dest_file.exists():
                    dest_file.unlink()
                shutil.copy2(so_file, dest_file)


setup(
    ext_modules=[
        BazelExtension("axon._axon", "//axon/python:axon_python_lib"),
    ],
    cmdclass={"build_ext": BazelBuildExt},
    packages=["axon"],
    # Include __init__.py, device.py, and all .so files in libs/
    package_data={"axon": ["*.py", "libs/*.so*"]},
)
