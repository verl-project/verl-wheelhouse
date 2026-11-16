#!/usr/bin/env python3
"""Helpers for resumable multi-attempt builds across GitHub Actions retries.

This module intentionally lives under ci/runner/ (pure CI orchestration,
excluded from ci/generate_matrix.py's wheel-input fingerprints) and uses only
the Python standard library so it can run before PyYAML or extra packages are
installed.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import posixpath
import re
import site
import sys
import sysconfig
from collections.abc import Iterable
from pathlib import Path

_SOURCE_DIRECTIVE_RE = re.compile(
    r'^\s*(?:source|\.)\s+["\']?\$\{(?:SCRIPT_DIR|__lib_dir|__provision_dir)\}/([^"\'\s]+)["\']?',
    re.MULTILINE,
)

TOOLCHAIN_ACTION_REL = Path(".github/actions/build-toolchain/action.yml")
BUILD_SCRIPTS_REL = Path("ci/build_scripts")
CUDNN_PROVISION_REL = Path("ci/build_scripts/provision/install_cudnn.sh")
PATCHES_REL = Path("ci/patches")

COMPILED_ARTIFACT_DIRS = (
    Path("build"),
    Path("3rdparty/nccl-extensions/build"),
)


def _is_compiled_artifact(path: Path) -> bool:
    name = path.name
    return name.endswith((".o", ".a", ".so")) or ".so." in name


def _sourced_script_closure(repo_root: Path, root_rels: list[Path]) -> list[Path]:
    closure: list[Path] = []
    seen: set[str] = set()
    stack: list[Path] = list(root_rels)
    while stack:
        rel = stack.pop()
        key = rel.as_posix()
        if key in seen:
            continue
        seen.add(key)
        full_path = repo_root / rel
        if not full_path.is_file():
            continue
        closure.append(rel)
        text = full_path.read_text(encoding="utf-8")
        for dep in _SOURCE_DIRECTIVE_RE.findall(text):
            target = Path(posixpath.normpath(posixpath.join(rel.parent.as_posix(), dep)))
            if target.as_posix() not in seen:
                stack.append(target)
    return closure


def _referenced_patches(repo_root: Path, script_rels: Iterable[Path]) -> list[Path]:
    patches_dir = repo_root / PATCHES_REL
    if not patches_dir.is_dir():
        return []
    patch_files = [p for p in patches_dir.iterdir() if p.is_file()]
    combined_text = "\n".join(
        (repo_root / rel).read_text(encoding="utf-8")
        for rel in script_rels
        if (repo_root / rel).is_file()
    )
    referenced: list[Path] = []
    for patch_path in patch_files:
        if patch_path.name in combined_text:
            referenced.append(PATCHES_REL / patch_path.name)
    return referenced


def _sanitize_cache_segment(value: str) -> str:
    # GitHub Actions cache keys forbid commas and benefit from concise tokens.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "none"


def compute_cache_key_prefix(
    repo_root: Path,
    *,
    component: str,
    ref: str,
    builder: str,
    arch: str,
    cuda_version: str,
    python_version: str,
    torch_version: str,
    torch_vision_version: str = "",
    cxx11_abi: str = "",
    torch_cuda_arch_list: str = "",
    requires_cudnn: str = "false",
    extra_env: str = "{}",
) -> str:
    """Compute a config-scoped cache key prefix for resumable build caches.

    Because build-cache.tar archives the full submodule checkout (including
    patched files and .git), restoring a cache across different refs, arch lists,
    extra_env, or patch/builder changes would overwrite the checked-out source
    or reuse incompatible object files. Hashing those build-affecting inputs
    into the prefix isolates incompatible caches while still allowing retries
    (same run_id, incrementing run_attempt) and re-runs of the identical config
    to resume.
    """
    roots: list[Path] = []
    if builder:
        roots.append(BUILD_SCRIPTS_REL / f"{builder}.sh")
    if str(requires_cudnn).lower() == "true":
        roots.append(CUDNN_PROVISION_REL)

    script_closure = _sourced_script_closure(repo_root, roots)
    input_files: list[Path] = [TOOLCHAIN_ACTION_REL]
    input_files.extend(script_closure)
    input_files.extend(_referenced_patches(repo_root, script_closure))

    file_hashes: dict[str, str] = {}
    for rel in sorted(set(input_files), key=lambda p: p.as_posix()):
        full_path = repo_root / rel
        if full_path.is_file():
            file_hashes[rel.as_posix()] = hashlib.sha256(
                full_path.read_bytes()
            ).hexdigest()

    payload = {
        "builder": builder,
        "cxx11_abi": cxx11_abi,
        "extra_env": extra_env,
        "file_hashes": file_hashes,
        "ref": ref,
        "requires_cudnn": str(requires_cudnn).lower(),
        "torch_cuda_arch_list": torch_cuda_arch_list,
        "torch_vision_version": torch_vision_version,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]

    return "-".join(
        [
            "build-cache",
            _sanitize_cache_segment(component),
            _sanitize_cache_segment(arch),
            _sanitize_cache_segment(cuda_version),
            _sanitize_cache_segment(python_version),
            _sanitize_cache_segment(torch_version),
            digest,
        ]
    )


def default_toolchain_dirs() -> list[Path]:
    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
    except AttributeError:
        pass
    for key in ("include", "platinclude"):
        val = sysconfig.get_path(key)
        if val:
            candidates.append(val)
    candidates.append(
        f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"
    )
    candidates.extend(sorted(glob.glob("/usr/local/cuda*")))
    candidates.append("/usr/include")

    seen: set[Path] = set()
    dirs: list[Path] = []
    for raw in candidates:
        p = Path(raw)
        if not p.is_dir():
            continue
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            dirs.append(resolved)
    return dirs


def prepare_resumed_build(
    build_root: Path,
    toolchain_dirs: Iterable[Path] | None = None,
    *,
    include_default_toolchain_dirs: bool = True,
) -> dict[str, object]:
    """Prepare an unpacked build-cache checkout for incremental compilation.

    1. Delete any 0-byte compiled artifacts (.o/.a/.so) that an abrupt OOM/kill
       might have left behind.
    2. Find the oldest remaining compiled artifact timestamp T_oldest.
    3. Clamp external toolchain headers/libraries installed on this fresh runner
       to <= (T_oldest - 60s) when their mtime is newer.

    Why clamp toolchain headers instead of touching cached .o files?
    Fresh runners install torch (and pybind11/cudnn_frontend/nvshmem) with the
    current wall-clock mtime, which is newer than the .o files restored from
    build-cache.tar. Ninja reads those header paths from .o.d / .ninja_deps and
    rebuilds every translation unit from scratch. Conversely, touching .o files
    without updating .ninja_deps breaks CMake/Ninja builds that use `deps = gcc`,
    because Ninja rejects .ninja_deps entries when `output.mtime > deps.mtime`.
    Clamping newly provisioned toolchain headers preserves both invariants.
    """
    build_root = build_root.resolve()
    pruned_empty = 0
    oldest_mtime: float | None = None

    for rel_dir in COMPILED_ARTIFACT_DIRS:
        artifact_root = build_root / rel_dir
        if not artifact_root.is_dir():
            continue
        for dirpath, _, filenames in os.walk(artifact_root, followlinks=False):
            for name in filenames:
                path = Path(dirpath) / name
                if path.is_symlink() or not _is_compiled_artifact(path):
                    continue
                try:
                    st = path.stat()
                except FileNotFoundError:
                    continue
                if st.st_size == 0:
                    path.unlink(missing_ok=True)
                    pruned_empty += 1
                    continue
                if oldest_mtime is None or st.st_mtime < oldest_mtime:
                    oldest_mtime = st.st_mtime

    if oldest_mtime is None:
        return {
            "pruned_empty": pruned_empty,
            "oldest_mtime": None,
            "clamp_ts": None,
            "clamped_files": 0,
            "permission_skipped": 0,
        }

    clamp_ts = oldest_mtime - 60.0
    search_dirs: list[Path] = []
    if include_default_toolchain_dirs:
        search_dirs.extend(default_toolchain_dirs())
    if toolchain_dirs is not None:
        for d in toolchain_dirs:
            if d.is_dir():
                resolved = d.resolve()
                if resolved not in search_dirs:
                    search_dirs.append(resolved)

    clamped_files = 0
    permission_skipped = 0
    for root_dir in search_dirs:
        # Never touch files inside the build checkout itself.
        if root_dir == build_root or build_root in root_dir.parents:
            continue
        for dirpath, _, filenames in os.walk(root_dir, followlinks=False):
            for name in filenames:
                if name.endswith((".py", ".pyc")):
                    continue
                path = Path(dirpath) / name
                if path.is_symlink():
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                if st.st_mtime > clamp_ts:
                    try:
                        os.utime(path, (clamp_ts, clamp_ts))
                        clamped_files += 1
                    except PermissionError:
                        permission_skipped += 1
                    except OSError:
                        continue

    return {
        "pruned_empty": pruned_empty,
        "oldest_mtime": oldest_mtime,
        "clamp_ts": clamp_ts,
        "clamped_files": clamped_files,
        "permission_skipped": permission_skipped,
    }


def _cmd_key(args: argparse.Namespace) -> int:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    prefix = compute_cache_key_prefix(
        repo_root,
        component=os.environ["COMPONENT"],
        ref=os.environ["REF"],
        builder=os.environ["BUILDER"],
        arch=os.environ["TARGET_ARCH"],
        cuda_version=os.environ["CUDA_VERSION"],
        python_version=os.environ["PYTHON_VERSION"],
        torch_version=os.environ["TORCH_VERSION"],
        torch_vision_version=os.environ.get("TORCH_VISION_VERSION", ""),
        cxx11_abi=os.environ.get("CXX11_ABI", ""),
        torch_cuda_arch_list=os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        requires_cudnn=os.environ.get("REQUIRES_CUDNN", "false"),
        extra_env=os.environ.get("EXTRA_ENV", "{}"),
    )
    if args.github_output:
        github_output = Path(os.environ["GITHUB_OUTPUT"])
        with github_output.open("a", encoding="utf-8") as fh:
            fh.write(f"key_prefix={prefix}\n")
    else:
        print(prefix)
    return 0


def _cmd_prepare_resumed(args: argparse.Namespace) -> int:
    extra_dirs = [Path(d) for d in args.extra_toolchain_dir]
    result = prepare_resumed_build(
        Path(args.build_root),
        toolchain_dirs=extra_dirs,
        include_default_toolchain_dirs=not args.skip_default_toolchain_dirs,
    )
    print(
        "Resumable build cache prepared: "
        f"pruned_empty={result['pruned_empty']} "
        f"oldest_mtime={result['oldest_mtime']} "
        f"clamp_ts={result['clamp_ts']} "
        f"clamped_files={result['clamped_files']} "
        f"permission_skipped={result['permission_skipped']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    key_parser = subparsers.add_parser("key", help="Compute resumable cache key prefix")
    key_parser.add_argument("--repo-root", default=None)
    key_parser.add_argument("--github-output", action="store_true")
    key_parser.set_defaults(func=_cmd_key)

    prep_parser = subparsers.add_parser(
        "prepare-resumed",
        help="Prune empty artifacts and clamp toolchain mtimes for incremental resume",
    )
    prep_parser.add_argument("--build-root", default=".")
    prep_parser.add_argument("--extra-toolchain-dir", action="append", default=[])
    prep_parser.add_argument("--skip-default-toolchain-dirs", action="store_true")
    prep_parser.set_defaults(func=_cmd_prepare_resumed)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
