#!/usr/bin/env python3
"""Unit tests for ci/runner/resumable_cache.py and patch-resume idempotency."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ci" / "runner"))
sys.path.insert(0, str(REPO_ROOT / "ci"))

import generate_matrix
import resumable_cache


class CacheKeyPrefixTests(unittest.TestCase):
    def _base_kwargs(self, repo_root: Path) -> dict:
        return {
            "repo_root": repo_root,
            "component": "flash-attention",
            "ref": "v2.8.3",
            "builder": "flash_attention",
            "arch": "x86_64",
            "cuda_version": "13.2.2",
            "python_version": "3.12",
            "torch_version": "2.12.1",
            "torch_vision_version": "0.27.1",
            "cxx11_abi": "TRUE",
            "torch_cuda_arch_list": "8.0;9.0;10.0;12.0",
            "requires_cudnn": "false",
            "extra_env": '{"FLASH_ATTN_LOCAL_VERSION": "cute_split.d51a4a1"}',
        }

    def _make_fake_repo(self, root: Path) -> None:
        (root / ".github" / "actions" / "build-toolchain").mkdir(parents=True)
        (root / ".github" / "actions" / "build-toolchain" / "action.yml").write_text(
            "name: toolchain\n", encoding="utf-8"
        )
        (root / "ci" / "build_scripts" / "lib").mkdir(parents=True)
        (root / "ci" / "build_scripts" / "lib" / "env.sh").write_text(
            "export_extra_env() { :; }\n", encoding="utf-8"
        )
        (root / "ci" / "build_scripts" / "flash_attention.sh").write_text(
            'source "${SCRIPT_DIR}/lib/env.sh"\n'
            'patch_file="${SCRIPT_DIR}/../patches/flash_attention_exclude_cute.patch"\n',
            encoding="utf-8",
        )
        (root / "ci" / "build_scripts" / "apex.sh").write_text(
            'source "${SCRIPT_DIR}/lib/env.sh"\n', encoding="utf-8"
        )
        (root / "ci" / "patches").mkdir(parents=True)
        (root / "ci" / "patches" / "flash_attention_exclude_cute.patch").write_text(
            "diff --git a/setup.py b/setup.py\n", encoding="utf-8"
        )
        (root / "ci" / "patches" / "enable_deep_ep_sm80.py").write_text(
            "# deep_ep patch\n", encoding="utf-8"
        )

    def test_key_is_deterministic_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_fake_repo(root)
            k1 = resumable_cache.compute_cache_key_prefix(**self._base_kwargs(root))
            k2 = resumable_cache.compute_cache_key_prefix(**self._base_kwargs(root))
            self.assertEqual(k1, k2)
            self.assertNotIn(",", k1)
            self.assertLess(len(k1), 200)
            self.assertTrue(
                k1.startswith("build-cache-flash-attention-x86_64-13.2.2-3.12-2.12.1-")
            )

    def test_key_changes_when_build_affecting_inputs_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_fake_repo(root)
            baseline = resumable_cache.compute_cache_key_prefix(**self._base_kwargs(root))

            for field, new_val in [
                ("ref", "v2.8.4"),
                ("torch_cuda_arch_list", "8.0;9.0"),
                ("extra_env", "{}"),
                ("cxx11_abi", "FALSE"),
                ("requires_cudnn", "true"),
            ]:
                kw = self._base_kwargs(root)
                kw[field] = new_val
                self.assertNotEqual(
                    baseline,
                    resumable_cache.compute_cache_key_prefix(**kw),
                    msg=f"changing {field} must change cache key prefix",
                )

    def test_key_changes_when_builder_or_referenced_patch_changes_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_fake_repo(root)
            k0 = resumable_cache.compute_cache_key_prefix(**self._base_kwargs(root))

            # Unrelated builder/patch edit must NOT invalidate flash-attention cache.
            (root / "ci" / "build_scripts" / "apex.sh").write_text(
                "# unrelated change\n", encoding="utf-8"
            )
            (root / "ci" / "patches" / "enable_deep_ep_sm80.py").write_text(
                "# unrelated patch change\n", encoding="utf-8"
            )
            k_after_unrelated = resumable_cache.compute_cache_key_prefix(
                **self._base_kwargs(root)
            )
            self.assertEqual(k0, k_after_unrelated)

            # Editing referenced patch MUST invalidate flash-attention cache.
            (root / "ci" / "patches" / "flash_attention_exclude_cute.patch").write_text(
                "# updated patch\n", encoding="utf-8"
            )
            k_after_patch = resumable_cache.compute_cache_key_prefix(
                **self._base_kwargs(root)
            )
            self.assertNotEqual(k0, k_after_patch)

            # Editing sourced lib/env.sh MUST invalidate flash-attention cache.
            (root / "ci" / "build_scripts" / "lib" / "env.sh").write_text(
                "# updated env\n", encoding="utf-8"
            )
            k_after_lib = resumable_cache.compute_cache_key_prefix(
                **self._base_kwargs(root)
            )
            self.assertNotEqual(k_after_patch, k_after_lib)


class PrepareResumedBuildTests(unittest.TestCase):
    def test_prunes_empty_objects_and_clamps_toolchain_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_root = root / "checkout"
            obj_dir = build_root / "build" / "temp.linux-x86_64-cpython-312"
            obj_dir.mkdir(parents=True)

            # Simulate Attempt 1 objects (old mtimes) + one 0-byte object from SIGKILL.
            t_old_obj = time.time() - 7200  # 2h ago
            t_new_obj = time.time() - 3600  # 1h ago
            obj1 = obj_dir / "a.o"
            obj2 = obj_dir / "b.o"
            empty_obj = obj_dir / "truncated.o"
            obj1.write_bytes(b"ELF-obj-1")
            obj2.write_bytes(b"ELF-obj-2")
            empty_obj.write_bytes(b"")
            os.utime(obj1, (t_old_obj, t_old_obj))
            os.utime(obj2, (t_new_obj, t_new_obj))
            os.utime(empty_obj, (t_new_obj, t_new_obj))

            # Simulate fresh runner toolchain headers installed "now" (> obj mtime).
            site_dir = root / "site-packages"
            (site_dir / "torch" / "include").mkdir(parents=True)
            header = site_dir / "torch" / "include" / "extension.h"
            py_file = site_dir / "torch" / "__init__.py"
            header.write_text("#pragma once\n", encoding="utf-8")
            py_file.write_text("# python\n", encoding="utf-8")
            py_mtime_before = py_file.stat().st_mtime

            result = resumable_cache.prepare_resumed_build(
                build_root,
                toolchain_dirs=[site_dir, build_root],
                include_default_toolchain_dirs=False,
            )

            self.assertEqual(result["pruned_empty"], 1)
            self.assertFalse(empty_obj.exists())
            self.assertEqual(result["clamped_files"], 1)

            # Cached object mtimes must NOT be altered (preserves .ninja_deps invariant).
            self.assertAlmostEqual(obj1.stat().st_mtime, t_old_obj, places=2)
            self.assertAlmostEqual(obj2.stat().st_mtime, t_new_obj, places=2)

            # External header must now be strictly older than oldest cached object.
            self.assertLess(header.stat().st_mtime, obj1.stat().st_mtime)
            # .py files must be untouched so __pycache__ stays valid.
            self.assertAlmostEqual(py_file.stat().st_mtime, py_mtime_before, places=2)

    def test_no_compiled_artifacts_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_root = root / "checkout"
            build_root.mkdir()
            site_dir = root / "site-packages"
            site_dir.mkdir()
            header = site_dir / "foo.h"
            header.write_text("int x;\n", encoding="utf-8")
            before = header.stat().st_mtime

            result = resumable_cache.prepare_resumed_build(
                build_root,
                toolchain_dirs=[site_dir],
                include_default_toolchain_dirs=False,
            )
            self.assertIsNone(result["oldest_mtime"])
            self.assertEqual(result["clamped_files"], 0)
            self.assertAlmostEqual(header.stat().st_mtime, before, places=2)

    def test_permission_error_is_handled_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_root = root / "checkout"
            obj_dir = build_root / "build"
            obj_dir.mkdir(parents=True)
            obj = obj_dir / "a.o"
            obj.write_bytes(b"data")
            t_old = time.time() - 3600
            os.utime(obj, (t_old, t_old))

            sys_dir = root / "usr_include"
            sys_dir.mkdir()
            hdr = sys_dir / "nccl.h"
            hdr.write_text("#define NCCL 1\n", encoding="utf-8")

            with patch.object(os, "utime", side_effect=PermissionError("read-only")):
                result = resumable_cache.prepare_resumed_build(
                    build_root,
                    toolchain_dirs=[sys_dir],
                    include_default_toolchain_dirs=False,
                )
            self.assertEqual(result["permission_skipped"], 1)
            self.assertEqual(result["clamped_files"], 0)


class FlashAttentionPatchIdempotencyTests(unittest.TestCase):
    def test_flash_attention_patch_is_idempotent_across_resume(self) -> None:
        patch_path = REPO_ROOT / "ci" / "patches" / "flash_attention_exclude_cute.patch"
        self.assertTrue(patch_path.is_file())

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            # Construct a minimal setup.py matching lines 600-615 of flash-attention v2.8.3.
            lines = [f"# line {i}\n" for i in range(1, 603)]
            lines.extend(
                [
                    "    packages=find_packages(\n",
                    "        exclude=(\n",
                    '            "docs",\n',
                    '            "benchmarks",\n',
                    '            "flash_attn.egg-info",\n',
                    "        )\n",
                    "    ),\n",
                    '    author="Tri Dao",\n',
                ]
            )
            setup_py = repo / "setup.py"
            setup_py.write_text("".join(lines), encoding="utf-8")

            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "CI"], cwd=repo, check=True)
            subprocess.run(["git", "add", "setup.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

            apply_cmd = (
                f'patch_file="{patch_path}"\n'
                'if git apply --reverse --check "${patch_file}" >/dev/null 2>&1; then\n'
                '  echo "already-applied"\n'
                "else\n"
                '  git apply "${patch_file}"\n'
                "fi\n"
            )

            # Attempt 1: cold checkout -> applies patch.
            r1 = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", apply_cmd],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertNotIn("already-applied", r1.stdout)
            self.assertIn('"flash_attn.cute"', setup_py.read_text(encoding="utf-8"))

            # Backdate setup.py to simulate unpacked build-cache.tar mtime.
            old_ts = time.time() - 1800
            os.utime(setup_py, (old_ts, old_ts))

            # Attempt 2: resumed checkout -> skips cleanly without touching setup.py mtime.
            r2 = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", apply_cmd],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("already-applied", r2.stdout)
            self.assertAlmostEqual(setup_py.stat().st_mtime, old_ts, places=2)


class FingerprintIsolationTests(unittest.TestCase):
    def test_resumable_cache_helper_not_in_wheel_fingerprints(self) -> None:
        versions = generate_matrix.load_versions()
        for component in generate_matrix.component_names(versions):
            for arch in generate_matrix.component_arches(versions, component):
                fp = generate_matrix.build_input_fingerprint(versions, component, arch)
                paths = {entry["path"] for entry in fp}
                self.assertNotIn("ci/runner/resumable_cache.py", paths)
                self.assertNotIn(".github/workflows/_build.yml", paths)


if __name__ == "__main__":
    unittest.main()
