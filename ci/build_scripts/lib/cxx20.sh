#!/usr/bin/env bash
# Upgrade hardcoded -std=c++17 flags in a checkout's setup.py to -std=c++20
# when building against PyTorch >= 2.14, whose ATen/torch headers require C++20.
#
# Idempotent and mtime-preserving: when a resumed build already carries the
# rewritten setup.py, the file is left untouched so Ninja/distutils keeps its
# cached objects. Only builders that source this file include it in their
# build-input fingerprint (see ci/generate_matrix.py).

upgrade_setup_py_to_cxx20_for_torch214() {
  local setup_file="${1:-setup.py}"
  python3 - "${setup_file}" <<'PY'
import os
import re
import sys
from pathlib import Path

torch_ver = os.environ.get("TORCH_VERSION", "").strip()
if not torch_ver:
    import torch

    torch_ver = torch.__version__

m = re.match(r"(\d+)\.(\d+)", torch_ver)
if not m or (int(m.group(1)), int(m.group(2))) < (2, 14):
    sys.exit(0)

setup_path = Path(sys.argv[1])
text = setup_path.read_text(encoding="utf-8")
updated = text.replace("-std=c++17", "-std=c++20")
if updated == text:
    print(f"{setup_path} already uses C++20 (or has no -std=c++17); skipping.")
    sys.exit(0)

setup_path.write_text(updated, encoding="utf-8")
print(f"Upgraded -std=c++17 -> -std=c++20 in {setup_path} for torch {torch_ver}.")
PY
}
