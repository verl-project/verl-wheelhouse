#!/usr/bin/env python3
"""Recreate no-`-cu` release tags as compatibility aliases after a tag migration.

The release-tag scheme is append-only: frozen uv.lock files pin full
/releases/download/<tag>/<file> URLs (see release_tag in ci/generate_matrix.py),
so deleting or renaming a tag turns those locks into 404s. When a tag has
already been removed (or migrated to a `-cu<maj><min>` keyed tag), this tool
recreates the old tag as a small "compat alias" release that serves the exact
wheels the canonical release has, without touching any index:

    apex-master-py3.12-torch2.13.0        (compat alias, not indexed)
        copies wheels from apex-master-py3.12-torch2.13.0-cu130

The alias title carries no "cu.. py.. torch.." segment, so ci/build_index.py
leaves it out of every simple-index world (no duplicate hrefs) while the
pinned direct download URLs resolve again. Bytes are copied unchanged.

Usage (dry run unless --apply):
    python ci/restore_compat_tags.py --repo owner/repo \
        --alias apex-master-py3.12-torch2.13.0 \
        --canonical apex-master-py3.12-torch2.13.0-cu130 \
        [--alias <old> --canonical <new> ...] [--apply]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

COMPAT_TITLE = "[compat alias, not indexed] {alias} - canonical: {canonical}"

NOTES = (
    "Compatibility alias for frozen locks that still pin this tag.\n\n"
    "Wheels are the original build copied from the canonical cuda-keyed "
    "release (see the title). Deliberately NOT indexed: the title carries no "
    "`cu.. py.. torch..` segment, so ci/build_index.py leaves it out of every "
    "simple-index world. Use the cuda-keyed tag / per-world index for new "
    "resolves; this tag only keeps already-pinned direct download URLs alive.\n"
)


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def gh_json(*args: str) -> object:
    return json.loads(run(["gh", *args]).stdout)


def wheel_names(repo: str, tag: str) -> List[str]:
    release = gh_json("api", f"repos/{repo}/releases/tags/{tag}")
    return [a["name"] for a in release.get("assets", []) if a["name"].endswith(".whl")]  # type: ignore[union-attr]


def release_exists(repo: str, tag: str) -> bool:
    return subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo], capture_output=True
    ).returncode == 0


def plan(repo: str, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, List[str]]]:
    todo = []
    for alias, canonical in pairs:
        if release_exists(repo, alias):
            print(f"SKIP (exists): {alias}")
            continue
        wheels = wheel_names(repo, canonical)
        if not wheels:
            print(f"SKIP (canonical {canonical!r} has no wheels): {alias}")
            continue
        todo.append((alias, canonical, wheels))
        print(f"CREATE {alias} <- {canonical}")
        for w in wheels:
            print(f"    {w}")
    return todo


def apply(repo: str, todo: List[Tuple[str, str, List[str]]]) -> None:
    for alias, canonical, wheels in todo:
        with tempfile.TemporaryDirectory(prefix="compat-") as tmp:
            d = Path(tmp)
            for w in wheels:
                url = f"https://github.com/{repo}/releases/download/{canonical}/{w}"
                print(f"  download {w}", flush=True)
                run(["curl", "-fsSL", "--retry", "4", "-o", str(d / w), url])
            notes = d / "notes.md"
            notes.write_text(NOTES)
            run(
                [
                    "gh", "release", "create", alias, "--repo", repo,
                    "--title", COMPAT_TITLE.format(alias=alias, canonical=canonical),
                    "--notes-file", str(notes), "--target", "main",
                ]
            )
            run(
                ["gh", "release", "upload", alias, "--repo", repo, "--clobber",
                 *[str(d / w) for w in wheels]]
            )
        print(f"CREATED {alias} with {len(wheels)} wheel(s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--alias", action="append", default=[], help="old tag to recreate")
    parser.add_argument("--canonical", action="append", default=[],
                        help="canonical tag to copy wheels from (one per --alias, in order)")
    parser.add_argument("--apply", action="store_true", help="actually create releases")
    args = parser.parse_args()

    if not args.alias or len(args.alias) != len(args.canonical):
        parser.error("provide matching --alias/--canonical pairs (same count)")

    todo = plan(args.repo, list(zip(args.alias, args.canonical)))
    if not args.apply:
        print(f"\nDry run: would recreate {len(todo)} compat release(s). Re-run with --apply.")
        return
    apply(args.repo, todo)
    print(f"Done: recreated {len(todo)} compat release(s).")


if __name__ == "__main__":
    main()
