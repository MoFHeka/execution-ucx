import re
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_VERSIONED_SO = re.compile(r"\.so\.\d+\.\d+")


class BazelExtension(Extension):
    def __init__(self, name, bazel_target):
        super().__init__(name, sources=[])
        self.bazel_target = bazel_target


class BazelBuildExt(build_ext):
    """Build C extensions via Bazel and place symlinks inplace.

    We override run() entirely to avoid setuptools' internal copy_file
    machinery, which would dereference our symlinks into actual files.
    """

    def run(self):
        try:
            subprocess.check_call(
                ["bazel", "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            raise RuntimeError("Bazel must be installed to build this extension.")

        setup_dir = Path(__file__).resolve().parent
        workspace_root = setup_dir.parent.parent

        try:
            bazel_bin_out = subprocess.check_output(
                ["bazel", "info", "bazel-bin"], cwd=workspace_root, text=True
            ).strip()
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to query 'bazel info bazel-bin': {e}") from e

        bazel_bin = Path(bazel_bin_out)
        bazel_axon = bazel_bin / "axon" / "python" / "axon"

        targets = [
            "//axon/python:axon/_axon.so",
            "//axon/python:copy_ucx_core_libs",
            "//axon/python:copy_ucx_transport_libs",
        ]
        print(f"Building with Bazel: {' '.join(targets)}")
        subprocess.check_call(["bazel", "build"] + targets, cwd=workspace_root)

        axon_so_src = bazel_axon / "_axon.so"
        if not axon_so_src.exists():
            raise RuntimeError(f"{axon_so_src} not found after bazel build.")

        pkg_dir = setup_dir / "axon"

        # Use the ABI-tagged name so multiple Python versions can coexist.
        # get_ext_fullpath returns a path relative to CWD (= setup_dir here).
        inplace_so = setup_dir / self.get_ext_fullpath(self.extensions[0].name)
        inplace_so.parent.mkdir(parents=True, exist_ok=True)
        _symlink(axon_so_src, inplace_so)

        # UCX core libs → axon/libs/  (resolved via RPATH $ORIGIN/libs)
        _symlink_dir(bazel_axon / "libs", pkg_dir / "libs")

        # UCX transport plugins → axon/libs/ucx/  (loaded by UCX via dlopen)
        _symlink_dir(bazel_axon / "ucx_modules", pkg_dir / "libs" / "ucx")

    def build_extension(self, ext):
        # Handled in run(); nothing to do here.
        pass


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _symlink_dir(src: Path, dst: Path) -> None:
    """Symlink only the most relevant .so file to prevent bloat.

    For a given library (e.g., libucm), prefers libucm.so.0 over libucm.so,
    and completely ignores fully versioned files like libucm.so.0.0.0.
    """
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)

    groups = {}
    for entry in src.iterdir():
        if not (entry.is_file() or entry.is_symlink()):
            continue
        base = entry.name.split(".so")[0]
        groups.setdefault(base, []).append(entry)

    for base, entries in groups.items():
        # Ignore .so.X.Y.Z
        valid = [e for e in entries if not _VERSIONED_SO.search(e.name)]
        if not valid:
            continue

        # Prefer .so.X over .so
        so_x = [e for e in valid if re.search(r"\.so\.\d+$", e.name)]
        best = so_x[0] if so_x else valid[0]

        _symlink(best, dst / best.name)


setup(
    ext_modules=[
        BazelExtension("axon._axon", "//axon/python:axon/_axon.so"),
    ],
    cmdclass={"build_ext": BazelBuildExt},
    packages=["axon"],
    package_data={"axon": ["*.py", "libs/*.so*", "libs/ucx/*.so*"]},
)
