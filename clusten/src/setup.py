#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
#
import os
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _prepend_env_path(var_name: str, new_path: Path) -> None:
    current = os.environ.get(var_name, "")
    os.environ[var_name] = f"{new_path}:{current}" if current else str(new_path)


def _nvcc_version(cuda_home: Path) -> str | None:
    try:
        output = subprocess.check_output([cuda_home / "bin" / "nvcc", "--version"], text=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    for line in output.splitlines():
        if "release" in line:
            # nvcc reports versions like "Cuda compilation tools, release 11.7, V11.7.64"
            return line.split("release", 1)[1].split(",", 1)[0].strip()
    return None


def configure_cuda_home(default_home: str = "/usr/local/cuda-11.7") -> None:
    """Ensure the build uses a CUDA toolkit compatible with the installed PyTorch."""
    import torch

    requested = os.environ.get("AFF_CUDA_HOME")
    torch_cuda = getattr(torch.version, "cuda", None)
    search_order: list[str] = []
    if requested:
        search_order.append(requested)
    if os.environ.get("CUDA_HOME"):
        search_order.append(os.environ["CUDA_HOME"])
    if torch_cuda:
        search_order.append(f"/usr/local/cuda-{torch_cuda}")
    search_order.extend([default_home, "/usr/local/cuda"])

    chosen = None
    for path in dict.fromkeys(search_order):
        candidate = Path(path)
        if candidate.exists():
            chosen = candidate
            break

    if not chosen:
        print(
            "[clusten] Warning: No usable CUDA toolkit directory found. "
            "Falling back to whatever nvcc Torch discovers."
        )
        return

    nvcc_version = _nvcc_version(chosen)
    if torch_cuda and nvcc_version and not nvcc_version.startswith(torch_cuda):
        print(
            f"[clusten] Warning: Torch expects CUDA {torch_cuda}, "
            f"but nvcc under {chosen} reports {nvcc_version}. Continuing anyway."
        )

    os.environ["CUDA_HOME"] = str(chosen)
    _prepend_env_path("PATH", chosen / "bin")
    _prepend_env_path("LD_LIBRARY_PATH", chosen / "lib64")
    print(f"[clusten] Using CUDA toolkit at {chosen}")


def configure_arch_list(default_arch: str = "8.6") -> None:
    """Set TORCH_CUDA_ARCH_LIST to a toolkit-compatible value if the user did not specify one."""
    override = os.environ.get("AFF_TORCH_CUDA_ARCH_LIST")
    if override:
        os.environ["TORCH_CUDA_ARCH_LIST"] = override
        print(f"[clusten] Using custom TORCH_CUDA_ARCH_LIST={override}")
        return

    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        # Respect explicit user configuration.
        return

    os.environ["TORCH_CUDA_ARCH_LIST"] = default_arch
    print(f"[clusten] TORCH_CUDA_ARCH_LIST not set; defaulting to {default_arch}")


configure_cuda_home()
configure_arch_list()


setup(
    name='clustencuda',
    version='0.1',
    author='Ziwen Chen',
    author_email='chenziw@oregonstate.edu',
    description='Cluster Attention CUDA Kernel',
    ext_modules=[
        CUDAExtension('clustenqk_cuda', [
            'clustenqk_cuda.cpp',
            'clustenqk_cuda_kernel.cu',
        ]),
        CUDAExtension('clustenav_cuda', [
            'clustenav_cuda.cpp',
            'clustenav_cuda_kernel.cu',
        ]),
        CUDAExtension('clustenwf_cuda', [
            'clustenwf_cuda.cpp',
            'clustenwf_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
