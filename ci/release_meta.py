#!/usr/bin/env python3
"""Compute GitHub Release metadata (tag/title/notes) for one or every
component in versions.yaml.

Each (component, Python version) pair gets its own persistent GitHub
Release, keyed by the component's currently-pinned ref:

    tag:   "<component>-<ref>"          e.g. "transformer-engine-v2.16.1"
           (the legacy interpreter - LEGACY_RELEASE_PYTHON, the only one
           the wheelhouse shipped before per-Python releases - keeps this
           bare tag so releases already on GitHub keep their identity and
           their skip detection)
           "<component>-<ref>-pyX.Y"    e.g. "apex-master-py3.11"
           (every other interpreter gets its own suffixed release, so
           adding a Python version creates releases instead of changing
           the existing ones)
    title: "<component> <ref> - [<arch> ]cu<cuda> py<python> torch<torch>[; ...]"
           (one segment per versions.yaml build_matrix row of that
           component *for that Python only*, with the x86_64 arch left
           implicit) e.g.
           "apex master - cu13.0.2 py3.11 torch2.11.0; aarch64 cu13.0.2 py3.11 torch2.11.0"
    notes: human-readable pin plus a hidden JSON snapshot of each wheel's
           build config (CUDA/Python/Torch, torch_cuda_arch_list, extra_env,
           builder, ...). Skip detection requires that snapshot to match
           versions.yaml exactly; title and wheel filename stay free of GPU arch.

Rebuilding the same ref re-uploads (--clobber) wheels onto the same
per-Python release; bumping a component's ref in versions.yaml starts a
brand new set of releases (new tags), leaving the previous one attached
to the old ref as a historical record. This module reuses
generate_matrix.py's helpers so the tag computed here always matches the
"release_tag" field _build.yml is given for that matrix row.

With no --python, the running interpreter's major.minor selects the
release: _build.yml runs this after setup-python has put the matrix row's
Python on PATH, so the correct per-Python metadata is computed without
another workflow input. --python X.Y selects one interpreter explicitly;
--python all emits one entry per (component, interpreter the component
opts into).

Usage:
    python ci/release_meta.py --component apex
    python ci/release_meta.py --component apex --python 3.11
    python ci/release_meta.py --component all --python all
    python ci/release_meta.py --component apex --github-output
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from typing import Any, Dict, List

from generate_matrix import (
    BUILD_MANIFEST_FILENAME,
    component_build_manifest,
    component_combos_for_python,
    component_names,
    component_python_versions,
    format_release_notes,
    get_component,
    load_versions,
    matrix_python_versions,
    release_tag,
    release_title,
)


def runtime_python() -> str:
    """Major.minor of the running interpreter (setup-python's Python in CI)."""
    return ".".join(platform.python_version().split(".")[:2])


def component_release_meta(
    versions: Dict[str, Any], component: str, python: str
) -> Dict[str, Any]:
    cfg = get_component(versions, component)
    ref = str(cfg["ref"])
    combos = component_combos_for_python(versions, component, python)
    if not combos:
        available = ", ".join(component_python_versions(versions, component))
        raise SystemExit(
            f"Component {component!r} has no build_matrix rows for Python {python}. "
            f"It builds for: {available or '(nothing)'}."
        )
    return {
        "component": component,
        "python": str(python),
        "ref": ref,
        "tag": release_tag(component, ref, python),
        "title": release_title(ref, component, combos),
        "notes": format_release_notes(versions, component, python),
        # Machine-readable build snapshot uploaded as a release asset
        # (BUILD_MANIFEST_FILENAME); the notes above embed the same JSON as a
        # fallback for older skip-detection code.
        "manifest": component_build_manifest(versions, component, python),
        "manifest_asset": BUILD_MANIFEST_FILENAME,
    }


def release_meta_entries(
    versions: Dict[str, Any], components: List[str], python: str
) -> List[Dict[str, Any]]:
    """One metadata record per (component, Python) pair.

    With python == "all" every component contributes one record per
    interpreter it opts into; otherwise components that don't opt into that
    interpreter are skipped (an explicitly named such component is rejected
    by main() before reaching here).
    """
    if python == "all":
        return [
            component_release_meta(versions, name, py)
            for name in components
            for py in component_python_versions(versions, name)
        ]
    return [
        component_release_meta(versions, name, python)
        for name in components
        if python in component_python_versions(versions, name)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--component",
        default="all",
        help="Component name from versions.yaml, or 'all' (default) for every component.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Python version the release is for (e.g. 3.11), or 'all' for every "
            "interpreter each component opts into. Defaults to the running "
            "interpreter's major.minor (CI: the matrix row's setup-python)."
        ),
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Also write matrix=<json> to $GITHUB_OUTPUT, for use in a workflow step.",
    )
    args = parser.parse_args()

    versions = load_versions()

    if args.component == "all":
        components = component_names(versions)
        explicit = False
    else:
        get_component(versions, args.component)  # validates, raises a clear SystemExit if unknown
        components = [args.component]
        explicit = True

    python = args.python if args.python is not None else runtime_python()
    if python != "all":
        known = matrix_python_versions(versions)
        if python not in known:
            raise SystemExit(
                f"Python {python!r} is not in the build_matrix. Known versions: "
                f"{', '.join(known)}. Pass --python X.Y or 'all'."
            )
        if explicit and python not in component_python_versions(versions, args.component):
            available = ", ".join(component_python_versions(versions, args.component))
            raise SystemExit(
                f"Component {args.component!r} does not build for Python {python}. "
                f"It builds for: {available or '(nothing)'}."
            )

    entries = release_meta_entries(versions, components, python)
    payload = json.dumps(entries)
    print(payload)

    if args.github_output:
        github_output = os.environ.get("GITHUB_OUTPUT")
        if not github_output:
            raise SystemExit("--github-output requires $GITHUB_OUTPUT to be set")
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={payload}\n")


if __name__ == "__main__":
    main()
