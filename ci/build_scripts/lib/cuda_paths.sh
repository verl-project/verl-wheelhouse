#!/usr/bin/env bash
# CUDA toolkit include / link path setup shared by the deep-ep and flash-mla
# builders. Sourced, not executed. Part of the build-input fingerprint only
# for the builders that source it (see build_input_fingerprint in
# ci/generate_matrix.py, which follows each builder's `source` directives).

# ---------------------------------------------------------------------------
# ensure_cuda_cccl_include_path: put CUDA 13's CCCL headers on CPATH.
#
# CUDA 13 moved libcu++ / cub / thrust out of the toolkit's top-level include/
# and into include/cccl/. nvcc finds them there on its own, but the *host*
# compiler does not, so a `#include <cuda/std/...>` reached from a .cpp
# translation unit (deep-ep's csrc/deep_ep.cpp, FlashMLA's cutlass headers)
# fails to resolve without help. verl's own Dockerfiles do the same thing -
# `export CPATH=/usr/local/cuda/targets/<target>/include/cccl:$CPATH` in
# docker/Dockerfile.stable.vllm / .sglang, a /usr/local/cuda-cccl symlink on
# CPATH in docker/Dockerfile.uv.cu130. A no-op warning on toolkits that
# predate the move (their headers are already on the default search path).
# ---------------------------------------------------------------------------
ensure_cuda_cccl_include_path() {
  local cuda_home target include_dir
  cuda_home="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
  case "$(uname -m)" in
    aarch64) target="sbsa-linux" ;;
    *) target="x86_64-linux" ;;
  esac

  for include_dir in \
    "${cuda_home}/targets/${target}/include/cccl" \
    "${cuda_home}/include/cccl"; do
    if [ -d "${include_dir}" ]; then
      export CPATH="${include_dir}${CPATH:+:${CPATH}}"
      echo "CCCL headers: added ${include_dir} to CPATH"
      return 0
    fi
  done

  echo "::warning::No cccl include directory found under ${cuda_home}; leaving CPATH unchanged."
}

# ---------------------------------------------------------------------------
# ensure_cuda_stub_library_path: put the toolkit's libcuda.so stub on the link
# path.
#
# deep-ep's hybrid_ep extension declares libraries=["cuda"], i.e. it links the
# CUDA driver API. Build machines have no NVIDIA driver, so that resolves
# against the toolkit's stub instead - harmless, because what ends up recorded
# in the .so is the real SONAME libcuda.so.1, which the GPU host provides at
# run time. torch's cpp_extension only ever adds $CUDA_HOME/lib64 to the link
# path, never lib64/stubs, so point the linker there via LIBRARY_PATH (link-time
# only: unlike an -rpath it leaves nothing behind in the built object). The
# stub itself comes from the driver-dev sub-package installed by the build
# toolchain action.
# ---------------------------------------------------------------------------
ensure_cuda_stub_library_path() {
  local cuda_home target stub_dir
  cuda_home="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
  case "$(uname -m)" in
    aarch64) target="sbsa-linux" ;;
    *) target="x86_64-linux" ;;
  esac

  for stub_dir in \
    "${cuda_home}/lib64/stubs" \
    "${cuda_home}/targets/${target}/lib/stubs"; do
    if [ -e "${stub_dir}/libcuda.so" ]; then
      export LIBRARY_PATH="${stub_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
      echo "CUDA driver stub: added ${stub_dir} to LIBRARY_PATH"
      return 0
    fi
  done

  echo "::error::No libcuda.so stub under ${cuda_home}; is the cuda-toolkit driver-dev sub-package installed?" >&2
  return 1
}
