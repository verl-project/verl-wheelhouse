#!/usr/bin/env bash
# CI-only disk cleanup run before a CUDA build on standard GitHub-hosted
# runners, mirroring flash-attention's own _build.yml. Deliberately kept OUT of
# ci/build_scripts/ (and thus out of the build-input fingerprint): freeing disk
# changes nothing about the wheel a build produces, so editing it must never
# trigger a rebuild. It is not called on the persistent self-hosted machine,
# where these paths belong to the host rather than a throwaway VM.
#
# Must be *sourced*, then the caller invokes `free_disk_space`.

# GitHub-hosted runners build as an unprivileged user with passwordless sudo;
# inlined here (rather than sourced from build_scripts/lib/env.sh) so this
# CI-only helper stays fully disjoint from the wheel build inputs.
_runner_maybe_sudo() {
  if [ "$(id -u)" -eq 0 ] || ! command -v sudo >/dev/null 2>&1; then
    "$@"
  else
    sudo "$@"
  fi
}

free_disk_space() {
  echo "::group::Free up disk space"
  _runner_maybe_sudo rm -rf /usr/share/dotnet || true
  _runner_maybe_sudo rm -rf /opt/ghc || true
  _runner_maybe_sudo rm -rf /opt/hostedtoolcache/CodeQL || true
  _runner_maybe_sudo rm -rf /usr/local/lib/android || true
  echo "::endgroup::"
}
