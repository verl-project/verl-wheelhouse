#!/usr/bin/env bash
# Builds the apex wheel using the exact flags verl's Dockerfiles use
# (docker/Dockerfile.stable.vllm / docker/Dockerfile.stable.sglang):
#
#   MAX_JOBS=<n> pip install -v --disable-pip-version-check --no-build-isolation \
#     --config-settings "--build-option=--cpp_ext" \
#     --config-settings "--build-option=--cuda_ext" \
#     git+https://github.com/NVIDIA/apex.git
#
# swapping `pip install` for `pip wheel --no-deps -w dist` against the local
# pinned checkout (`.`) so a distributable artifact is produced instead of
# installing in place. Run with CWD = the apex submodule checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/lib/env.sh
source "${SCRIPT_DIR}/lib/env.sh"

export_extra_env

# apex has a hard `nvcc == torch.version.cuda` check (setup.py's
# check_cuda_torch_binary_vs_bare_metal), so the installed torch's CUDA
# build tag must match the toolkit installed by the workflow.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

mkdir -p dist
MAX_JOBS="${MAX_JOBS}" pip wheel -v \
  --no-build-isolation \
  --no-deps \
  -w dist \
  --disable-pip-version-check \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" \
  .

echo "Built wheels:"
ls -al dist
