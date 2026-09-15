#!/usr/bin/env bash
# RDMA/InfiniBand headers and NVSHMEM provisioning, used only by the deep-ep
# builder. Sourced, not executed. Part of deep-ep's build-input fingerprint
# alone: a change here must not rebuild components that never link IB/NVSHMEM.
__lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/lib/env.sh
source "${__lib_dir}/env.sh"
unset __lib_dir

# ---------------------------------------------------------------------------
# install_rdma_devel: userspace RDMA/InfiniBand headers and libraries.
#
# deep-ep's csrc/kernels/configs.cuh includes <infiniband/mlx5dv.h> for its
# IBGDA path, so every translation unit that pulls configs.cuh in - the .cpp
# as well as the .cu files - fails to compile without these. GitHub's runner
# images don't carry them; verl's docker/Dockerfile.uv.cu130 apt-installs the
# same set in its base stage (its "RDMA/IB" superset of the sglang/vllm
# basic stages).
#
# libibverbs-dev is the one that actually matters (it ships
# /usr/include/infiniband/mlx5dv.h alongside verbs.h); librdmacm-dev and
# ibverbs-providers come along to mirror that image, the latter supplying the
# libmlx5 provider itself.
# ---------------------------------------------------------------------------
install_rdma_devel() {
  echo "::group::Install RDMA/InfiniBand development packages"
  maybe_sudo apt-get update
  maybe_sudo apt-get install -y --no-install-recommends \
    libibverbs-dev \
    librdmacm-dev \
    ibverbs-providers
  echo "::endgroup::"
}

# ---------------------------------------------------------------------------
# install_nvshmem: install NVIDIA's NVSHMEM pip package at the same absolute
# path verl's runtime image puts it, and export NVSHMEM_DIR pointing there.
#
# deep-ep links NVSHMEM and bakes `-Wl,-rpath,${NVSHMEM_DIR}/lib` into
# deep_ep_cpp.so, so a *prebuilt* wheel only resolves libnvshmem at run time if
# that same directory exists on the target machine. verl's
# docker/Dockerfile.uv.cu130 deliberately installs nvidia-nvshmem-cu<major> as
# a system pip package under /usr/local/lib/python<X.Y>/dist-packages (not into
# a venv), so this targets that exact path instead of the runner's own
# site-packages. Keep NVSHMEM_VERSION (extra_env in versions.yaml) in sync with
# that Dockerfile's NVSHMEM_VERSION arg.
#
# Setting NVSHMEM_DIR also makes deep-ep's setup.py link the unversioned
# `-l:libnvshmem_host.so`, which the pip package does not ship - hence the
# symlink, mirroring the one that Dockerfile creates. Only the build needs it:
# what gets recorded in the .so is the library's SONAME
# (libnvshmem_host.so.<major>), which the pip package does ship.
# ---------------------------------------------------------------------------
install_nvshmem() {
  local version="${NVSHMEM_VERSION:?NVSHMEM_VERSION must be set (see extra_env in versions.yaml)}"
  local cuda_major site_dir host_lib
  local host_libs=()
  cuda_major="$(echo "${CUDA_VERSION}" | cut -d. -f1)"
  site_dir="/usr/local/lib/python${PYTHON_VERSION}/dist-packages"

  echo "::group::Install nvidia-nvshmem-cu${cuda_major}==${version} into ${site_dir}"
  maybe_sudo pip install --upgrade --target "${site_dir}" \
    "nvidia-nvshmem-cu${cuda_major}==${version}"

  export NVSHMEM_DIR="${site_dir}/nvidia/nvshmem"
  host_libs=("${NVSHMEM_DIR}"/lib/libnvshmem_host.so.*)
  if [ ! -e "${host_libs[0]}" ]; then
    echo "::error::No libnvshmem_host.so.* under ${NVSHMEM_DIR}/lib after installing the wheel" >&2
    return 1
  fi
  host_lib="$(basename "${host_libs[0]}")"
  maybe_sudo ln -sf "${host_lib}" "${NVSHMEM_DIR}/lib/libnvshmem_host.so"

  echo "NVSHMEM_DIR=${NVSHMEM_DIR} (build-time symlink -> ${host_lib})"
  echo "::endgroup::"
}
