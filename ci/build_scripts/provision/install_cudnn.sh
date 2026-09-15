#!/usr/bin/env bash
# Provision cuDNN before building components that set requires_cudnn: true
# (currently transformer-engine). Invoked by the build-toolchain composite
# action, not by a builder, so it enters the build-input fingerprint only for
# components with requires_cudnn set (see build_input_fingerprint in
# ci/generate_matrix.py). Mirrors the cuDNN network-repo install both verl
# Dockerfiles run before building TransformerEngine. Expects CUDA_VERSION to be
# exported. This file must be *sourced* (it defines install_cudnn), then the
# caller invokes `install_cudnn`.
set -euo pipefail
__provision_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/lib/env.sh
source "${__provision_dir}/../lib/env.sh"
unset __provision_dir

install_cudnn() {
  local cuda_major arch
  cuda_major="$(echo "${CUDA_VERSION}" | cut -d. -f1)"
  arch="$(uname -m)"
  if [ "${arch}" = "aarch64" ]; then
    arch="sbsa"
  fi

  echo "::group::Install cuDNN ${cuda_major}"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${arch}/cuda-keyring_1.1-1_all.deb"
  maybe_sudo dpkg -i cuda-keyring_1.1-1_all.deb
  rm -f cuda-keyring_1.1-1_all.deb
  maybe_sudo apt-get update
  maybe_sudo apt-get install -y --allow-downgrades --allow-change-held-packages \
    "cudnn9-cuda-${cuda_major}"
  echo "::endgroup::"
}
