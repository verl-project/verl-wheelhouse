#!/usr/bin/env bash
# Builds the TransformerEngine PyTorch wheel using the exact env vars/flags
# verl's Dockerfiles use:
#
#   export NVTE_FRAMEWORK=pytorch && \
#     MAX_JOBS=<n> NVTE_BUILD_THREADS_PER_JOB=4 \
#     pip3 install --resume-retries 999 --no-build-isolation \
#     git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_VERSION}
#
# swapping `pip install` for `pip wheel --no-deps -w dist` against the local
# pinned checkout (`.`) so a distributable artifact is produced. Run with
# CWD = the TransformerEngine submodule checkout. Requires cuDNN (installed
# by the calling workflow, see versions.yaml's requires_cudnn: true) and NCCL
# (installed here via install_nccl).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/lib/env.sh
source "${SCRIPT_DIR}/lib/env.sh"
# shellcheck source=ci/build_scripts/lib/nccl.sh
source "${SCRIPT_DIR}/lib/nccl.sh"
# shellcheck source=ci/build_scripts/lib/arch.sh
source "${SCRIPT_DIR}/lib/arch.sh"

export_extra_env

install_nccl

# Both verl Dockerfiles pre-install these before building TransformerEngine.
pip install -q pybind11 nvidia-mathdx ninja wheel packaging

export NVTE_FRAMEWORK=pytorch

# Emit a clean release version (e.g. "2.16.1") in the built wheel's filename.
# Without this, TransformerEngine's build_tools/te_version.py appends the short
# git commit as a PEP 440 local-version segment (e.g. "2.16.1+c9877beb"), since
# _build.yml checks the submodule out at a detached commit rather than a tag.
# NVTE_NO_LOCAL_VERSION suppresses only that local segment - unlike
# NVTE_RELEASE_BUILD, which would also drop the PyTorch C++ extension and the
# framework runtime deps that this single-wheel build needs.
export NVTE_NO_LOCAL_VERSION=1

export NVTE_CUDA_ARCHS
NVTE_CUDA_ARCHS="$(arch_list_strip_dots "${TORCH_CUDA_ARCH_LIST}")"

mkdir -p dist
MAX_JOBS="${MAX_JOBS}" pip wheel -v \
  --no-build-isolation \
  --no-deps \
  -w dist \
  --resume-retries 999 \
  .

echo "Built wheels:"
ls -al dist
