"""Tests for ci/build_scripts/lib/cxx20.sh (PyTorch >= 2.14 C++20 upgrade)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
CXX20_LIB = REPO_ROOT / "ci" / "build_scripts" / "lib" / "cxx20.sh"

if str(CI_DIR) not in sys.path:
    sys.path.insert(0, str(CI_DIR))

import generate_matrix  # noqa: E402


class Cxx20UpgradeTests(unittest.TestCase):
    def _run_upgrade(self, setup_py: Path, torch_version: str) -> subprocess.CompletedProcess[str]:
        lib_posix = CXX20_LIB.as_posix()
        setup_posix = setup_py.as_posix()
        env = dict(os.environ)
        env["TORCH_VERSION"] = torch_version
        return subprocess.run(
            [
                "bash",
                "-euo",
                "pipefail",
                "-c",
                f'source "{lib_posix}" && upgrade_setup_py_to_cxx20_for_torch214 "{setup_posix}"',
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_cxx20_lib_scoped_to_flash_attention_and_deep_ep(self) -> None:
        versions = generate_matrix.load_versions()
        fp_paths = {
            comp: {
                entry["path"]
                for entry in generate_matrix.build_input_fingerprint(versions, comp, "x86_64")
            }
            for comp in ("flash-attention", "deep-ep", "apex", "transformer-engine")
        }
        self.assertIn("ci/build_scripts/lib/cxx20.sh", fp_paths["flash-attention"])
        self.assertIn("ci/build_scripts/lib/cxx20.sh", fp_paths["deep-ep"])
        self.assertNotIn("ci/build_scripts/lib/cxx20.sh", fp_paths["apex"])
        self.assertNotIn("ci/build_scripts/lib/cxx20.sh", fp_paths["transformer-engine"])

    def test_noop_for_torch_2_13(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            setup_py = Path(tmp) / "setup.py"
            original = 'cxx_flags = ["-O3", "-std=c++17"]\n'
            setup_py.write_text(original, encoding="utf-8")
            os.utime(setup_py, (1_700_000_000, 1_700_000_000))

            self._run_upgrade(setup_py, "2.13.0")

            self.assertEqual(setup_py.read_text(encoding="utf-8"), original)
            self.assertEqual(int(setup_py.stat().st_mtime), 1_700_000_000)

    def test_upgrades_and_preserves_mtime_on_resume_for_torch_2_14(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            setup_py = Path(tmp) / "setup.py"
            setup_py.write_text(
                'cxx_flags = ["-O3", "-std=c++17"]\nnvcc_flags = ["-std=c++17"]\n',
                encoding="utf-8",
            )

            self._run_upgrade(setup_py, "2.14.0")
            self.assertEqual(
                setup_py.read_text(encoding="utf-8"),
                'cxx_flags = ["-O3", "-std=c++20"]\nnvcc_flags = ["-std=c++20"]\n',
            )

            # Simulate resumed build from cache: second invocation must be a no-op
            # that does not bump setup.py's mtime.
            os.utime(setup_py, (1_700_000_123, 1_700_000_123))
            r2 = self._run_upgrade(setup_py, "2.14.0")
            self.assertIn("skipping", r2.stdout)
            self.assertEqual(int(setup_py.stat().st_mtime), 1_700_000_123)


if __name__ == "__main__":
    unittest.main()
