#!/usr/bin/env bash
# install_nccl, used only by the transformer_engine builder. Sourced, not
# executed. Part of transformer-engine's build-input fingerprint alone: a
# change here must not rebuild components that never link NCCL.
__lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/lib/env.sh
source "${__lib_dir}/env.sh"
unset __lib_dir

# ---------------------------------------------------------------------------
# install_nccl: TransformerEngine's common/util/logging.h always includes
# nccl.h, and NCCL EP (enabled when NVTE_CUDA_ARCHS contains arch >= 90)
# links libnccl at build time. Mirrors TE's own wheel Dockerfile
# (libnccl2 + libnccl-dev) and vllm/docker/Dockerfile's CUDA-matched pin.
# Expects CUDA_VERSION to be exported and the NVIDIA CUDA apt repo to already
# be configured (the cuda-toolkit install step does this).
# ---------------------------------------------------------------------------
install_nccl() {
  local cuda_short nccl_ver
  cuda_short="$(echo "${CUDA_VERSION}" | cut -d. -f1,2)"

  echo "::group::Install NCCL (+cuda${cuda_short})"
  maybe_sudo apt-get update
  nccl_ver="$(
    apt-cache madison libnccl-dev 2>/dev/null \
      | grep "+cuda${cuda_short}" \
      | head -1 \
      | awk -F'|' '{gsub(/^ +| +$/, "", $2); print $2}'
  )"
  if [ -z "${nccl_ver}" ]; then
    echo "::error::No libnccl-dev package found for +cuda${cuda_short}" >&2
    exit 1
  fi
  maybe_sudo apt-get install -y --no-install-recommends --allow-change-held-packages \
    "libnccl-dev=${nccl_ver}" "libnccl2=${nccl_ver}"
  echo "Installed libnccl-dev=${nccl_ver} libnccl2=${nccl_ver}"
  echo "::endgroup::"
}
