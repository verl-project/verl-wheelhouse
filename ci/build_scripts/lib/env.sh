#!/usr/bin/env bash
# Build-environment helpers every component builder sources: privilege
# escalation (maybe_sudo) and EXTRA_ENV -> exported variables.
#
# Because every builder sources this file it is part of every wheel's
# build-input fingerprint (see build_input_fingerprint in
# ci/generate_matrix.py). Keep it wheel-output-relevant only - CI plumbing
# (disk cleanup, caching, uploads) does not belong here.
#
# This file must be *sourced*, not executed. The calling workflow exports the
# matrix fields (CUDA_VERSION, TORCH_CUDA_ARCH_LIST, MAX_JOBS, EXTRA_ENV, ...)
# before the builder runs (see _build.yml's "Build wheel" step).

# ---------------------------------------------------------------------------
# maybe_sudo: run a privileged command through sudo only where that is both
# needed and possible. GitHub-hosted runners build as an unprivileged user
# with passwordless sudo; the self-hosted machine (apex, TransformerEngine)
# builds as root in a container that has no sudo binary at all.
# ---------------------------------------------------------------------------
maybe_sudo() {
  if [ "$(id -u)" -eq 0 ] || ! command -v sudo >/dev/null 2>&1; then
    "$@"
  else
    sudo "$@"
  fi
}

# ---------------------------------------------------------------------------
# export_extra_env: EXTRA_ENV is exported as a JSON object string by the
# workflow (e.g. '{"NVTE_BUILD_THREADS_PER_JOB": "4"}'); turn its entries
# into real exported environment variables.
# ---------------------------------------------------------------------------
export_extra_env() {
  if [ -z "${EXTRA_ENV:-}" ] || [ "${EXTRA_ENV}" = "{}" ]; then
    return 0
  fi
  local key value
  while IFS='=' read -r key value; do
    [ -n "${key}" ] || continue
    export "${key}=${value}"
    echo "Exported ${key}=${value} (from extra_env)"
  done < <(echo "${EXTRA_ENV}" | jq -r 'to_entries[] | "\(.key)=\(.value)"')
}
