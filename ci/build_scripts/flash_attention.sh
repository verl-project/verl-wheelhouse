#!/usr/bin/env bash
# Builds the flash-attention wheel, mirroring the build step in
# flash-attention's own .github/workflows/_build.yml. Run with CWD = the
# flash-attention submodule checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/lib/env.sh
source "${SCRIPT_DIR}/lib/env.sh"
# shellcheck source=ci/build_scripts/lib/arch.sh
source "${SCRIPT_DIR}/lib/arch.sh"

export_extra_env

pip install -q ninja packaging wheel psutil setuptools

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI="${CXX11_ABI}"
export FLASH_ATTN_CUDA_ARCHS
FLASH_ATTN_CUDA_ARCHS="$(arch_list_strip_dots "${TORCH_CUDA_ARCH_LIST}")"

# Limit parallel nvcc invocations on newer CUDA toolkits — mirrors
# flash-attention/.github/workflows/_build.yml. With MAX_JOBS>1 on GitHub's
# 7 GB runners, nvcc 12.9+ routinely OOM-kills the build (exit 137/143).
cuda_compact="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr -d '.')"
if [ "${cuda_compact}" = "129" ] || [ "${cuda_compact}" = "130" ] || [ "${cuda_compact}" = "132" ]; then
  export MAX_JOBS=1
else
  export MAX_JOBS="${MAX_JOBS}"
fi
export NVCC_THREADS="${NVCC_THREADS:-2}"

python setup.py bdist_wheel --dist-dir=dist

echo "Built wheels:"
ls -al dist
