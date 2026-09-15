#!/usr/bin/env bash
# flashinfer index/download helpers, used only by the flashinfer builder.
# Sourced, not executed. Part of only flashinfer's build-input fingerprint.

# ---------------------------------------------------------------------------
# flashinfer_jit_cache_cu_index: map a CUDA toolkit version to the
# flashinfer.ai jit-cache wheel index suffix (e.g. 13.0.2 -> cu130). Cubin
# wheels are fetched from the CUDA-agnostic https://flashinfer.ai/whl index
# instead. Mirrors sglang/docker/Dockerfile and vllm/docker/Dockerfile.
# ---------------------------------------------------------------------------
flashinfer_jit_cache_cu_index() {
  echo "cu$(echo "$1" | cut -d. -f1,2 | tr -d '.')"
}

# ---------------------------------------------------------------------------
# download_prebuilt_wheel: pip download a prebuilt wheel (wheels only, no deps)
# with retry logic, writing the .whl into dest_dir. Used to vendor wheels this
# repo intentionally does not build from source - currently flashinfer's
# companion wheels (flashinfer-cubin / -jit-cache from flashinfer.ai;
# jit-cache is ~1.2 GB and that index can be flaky).
# --only-binary=:all: guarantees we rehost the upstream binary wheel and never
# silently fall back to building an sdist.
# ---------------------------------------------------------------------------
download_prebuilt_wheel() {
  local package="$1"
  local version="$2"
  local index_url="$3"
  local dest_dir="$4"
  local attempt max_attempts=5

  for attempt in $(seq 1 "${max_attempts}"); do
    if pip download "${package}==${version}" \
      --index-url "${index_url}" \
      --no-deps \
      --only-binary=:all: \
      -d "${dest_dir}"; then
      echo "Downloaded ${package}==${version} from ${index_url}"
      return 0
    fi
    if [ "${attempt}" -lt "${max_attempts}" ]; then
      echo "::warning::Attempt ${attempt}/${max_attempts} to download ${package} failed; retrying in 10s..."
      sleep 10
    fi
  done

  echo "::error::Failed to download ${package}==${version} from ${index_url} after ${max_attempts} attempts" >&2
  return 1
}
