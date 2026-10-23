#!/usr/bin/env bash
# Build the host extension and package the headers used by runtime DeepJIT.
# Run with CWD = the clean DeepGEMM checkout, with recursive submodules ready.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/lib/env.sh
source "${SCRIPT_DIR}/lib/env.sh"

export_extra_env

# DeepJIT includes elfutils/libdwfl.h, and DeepGEMM requires C++20 <format>.
# Keep these prerequisites local to this builder so other wheels retain
# their existing toolchain fingerprints.
maybe_sudo apt-get update
maybe_sudo apt-get install -y --no-install-recommends g++-13 libdw-dev
export CC=gcc-13 CXX=g++-13
pip install -q packaging wheel setuptools

# setup.py uses the installed torch ABI directly (no force-ABI flag).
python - <<'PY'
import os
import torch

want = os.environ["CXX11_ABI"].upper() == "TRUE"
assert torch.compiled_with_cxx11_abi() == want, "torch CXX11 ABI does not match the matrix"
print("torch", torch.__version__, "cuda", torch.version.cuda, "cxx11abi", want)
PY

# Build from the pinned source, retaining +<commit> in both the wheel name
# and METADATA. GPU kernels are compiled at runtime; TORCH_CUDA_ARCH_LIST
# does not select their architectures at wheel-build time.
export DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=1 DG_SKIP_CUDA_BUILD=0
# setup.py persists these variables into the package if present. Never ship
# a CI machine's cache path or diagnostic settings as runtime defaults.
unset DG_JIT_CACHE_DIR DG_JIT_PRINT_COMPILER_COMMAND DG_JIT_CPP_STANDARD
python setup.py bdist_wheel --dist-dir=dist

# A CPU-only CI runner cannot launch GPU kernels. Check the package payload
# here; cold-cache numerical validation belongs on Hopper/Blackwell hosts.
python - <<'PY'
import subprocess
import zipfile
from pathlib import Path
from email.parser import BytesParser

revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
wheels = list(Path("dist").glob("deep_gemm-*.whl"))
assert len(wheels) == 1, f"Expected one DeepGEMM wheel, got {wheels}"
with zipfile.ZipFile(wheels[0]) as wheel:
    names = wheel.namelist()
    metadata = next(n for n in names if n.endswith(".dist-info/METADATA"))
    version = BytesParser().parsebytes(wheel.read(metadata))["Version"]
    assert version.endswith("+" + revision), f"Source revision missing from wheel: {version}"
    assert any(n.startswith("deep_gemm/_C.") and n.endswith(".so") for n in names)
    for directory in ("deep_gemm", "cutlass", "cute"):
        assert any(n.startswith(f"deep_gemm/include/{directory}/") for n in names), directory
print("Verified", wheels[0])
PY

echo "Built wheels:"
ls -al dist
