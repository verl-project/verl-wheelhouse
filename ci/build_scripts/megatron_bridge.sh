#!/usr/bin/env bash
# Builds the megatron-bridge wheel. NVIDIA-NeMo/Megatron-Bridge is a
# pure-Python library (declarative pyproject.toml with setuptools.build_meta,
# no setup.py and no ext_modules), so this produces a portable
# megatron_bridge-<ver>-py3-none-any.whl - there is nothing CUDA-specific to
# compile here. The reusable workflow still installs CUDA/torch before calling
# this script, but the wheel build itself only needs the setuptools backend.
#
# Two build details worth noting:
#   - pyproject's build-system.requires lists torch/pybind11, but with no
#     setup.py/ext_modules nothing is actually compiled against them. Building
#     with --no-build-isolation + the pinned backend avoids re-downloading a
#     multi-GB torch into an isolated env (torch is already installed by
#     _build.yml). --no-deps keeps us from pulling the heavy runtime
#     dependency tree (megatron-core, transformers, ...) just to package the
#     pure-Python sources.
#   - the version comes from megatron.bridge.package_info.__version__, which
#     appends "+<git-sha>" unless NO_VCS_VERSION=1. We set it so the wheel gets
#     a clean PEP 440 version (e.g. 0.5.2) with no local-version segment, so it
#     stays a valid, `pip install megatron-bridge==<ver>`-able release asset.
# Run with CWD = the Megatron-Bridge submodule checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/lib/env.sh
source "${SCRIPT_DIR}/lib/env.sh"

export_extra_env

# Emit a clean, non-local version: skips the package_info.py `git rev-parse`
# that would otherwise tack a "+<sha>" local segment onto __version__.
export NO_VCS_VERSION=1

# Build-backend deps only (mirrors pyproject's build-system.requires, minus
# torch which _build.yml already installed). setuptools is pinned <80 to match
# upstream's own pin and keep the `version = {attr = ...}` resolution stable.
pip install -q "setuptools<80.0.0" pybind11 wheel

mkdir -p dist
pip wheel -v \
  --no-build-isolation \
  --no-deps \
  -w dist \
  .

echo "Built wheels:"
ls -al dist
