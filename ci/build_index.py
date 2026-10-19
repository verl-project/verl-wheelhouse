#!/usr/bin/env python3
"""Build static PEP 503 "simple" package indexes from this repo's GitHub
Release assets, for deployment to GitHub Pages.

Every .whl attached to any GitHub Release in the repo is listed under its
(PEP 503-normalized) distribution name, linking straight back to the release
asset's own download URL - GitHub Pages only ever serves the tiny index
HTML, never the wheel bytes themselves.

One index is published per CUDA/torch world the releases cover:

    /cu130/torch2.13/simple/   every wheel built against cu13.0.2 + torch 2.13
    /cu130/torch2.11/simple/   ... and the ones still on torch 2.11
    /simple/                   alias for the build_matrix's current default

The split exists because a wheel filename records neither CUDA nor torch,
so two torch builds of the same component produce byte-different wheels with
identical names and versions. A single index could not hold both, and a
consumer resolving from it would have no way to ask for the build matching
its own torch; a per-world index URL makes that choice explicit and
unambiguous - the same shape download.pytorch.org and flashinfer.ai use.

Which world a release belongs to is read from its title, which
ci/release_meta.py generates with a "cu<cuda> py<python> torch<torch>"
segment per build.

Exception: a small set of genuinely CUDA/torch-independent distributions -
pure-Python `py3-none-any` wheels with no compiled or data payload - are
declared `world_agnostic: true` in versions.yaml (currently megatron-bridge).
Every release of such a distribution is mirrored into EVERY world, so a
consumer pinned to an older version keeps resolving it after the default
world moves (e.g. megatron-bridge 0.5.2 stays installable from the cu132
index even though it was first published under cu130). Different versions
have different filenames, so they coexist on one page; the newest is the
default and an exact `==<old>` pin still finds its wheel. This must not be
used for wheels that merely *name* py3-none-any but carry CUDA payloads
(flashinfer-cubin/jit-cache are CUDA-specific data despite the tag).

Usage:
    python ci/build_index.py --repo <owner>/<repo> --out-dir _site

Requires the `gh` CLI to be authenticated (e.g. via $GITHUB_TOKEN) and on
PATH, which is already true on GitHub-hosted runners.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, TypedDict

from generate_matrix import (
    combo_arch,
    combo_cuda,
    combo_python,
    combo_torch,
    component_combos,
    load_versions,
)

# Title prefix on hand-created compatibility-alias releases (see
# ci/restore_compat_tags.py): they exist only to keep frozen direct-download
# URLs alive after a tag-scheme migration, are deliberately not indexed, and
# must not spam the "no CUDA/torch segment" warning on every publish.
COMPAT_ALIAS_PREFIX = "[compat alias"


class WheelAsset(TypedDict):
    name: str
    url: str
    worlds: List[str]
    # GitHub release tag the asset lives on (for deterministic dedup tie-breaks).
    tag: str
    # Asset content digest exactly as the GitHub API reports it, e.g.
    # "sha256:90c5…" (None when GitHub has none). Used for PEP 503 hash
    # fragments and to detect same-version wheels whose bytes differ.
    digest: "str | None"


# "cu13.0.2 py3.12 torch2.13.0" segments of a release title (see
# ci/release_meta.py's release_title). One per build the release covers;
# under the current tag scheme they all agree on CUDA and torch and differ
# only by arch, but parsing every segment keeps older multi-world titles
# (and hand-made ones) from landing in the wrong index.
_TITLE_SEGMENT_RE = re.compile(r"cu(?P<cuda>[\w.]+)\s+py(?P<python>[\w.]+)\s+torch(?P<torch>[\w.+]+)")


def normalize(name: str) -> str:
    """PEP 503 project-name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_distribution(filename: str) -> str:
    """PEP 503-normalized distribution name from a wheel filename."""
    return normalize(filename.split("-")[0])


def is_universal_wheel(filename: str) -> bool:
    """True for a py3-none-any (interpreter/ABI/platform-independent) wheel."""
    return filename[: -len(".whl")].rsplit("-", 1)[-1] == "any"


def wheel_version(filename: str) -> str:
    """Distribution version from a wheel filename: 'megatron_bridge-0.6.2-...' -> '0.6.2'."""
    stem = filename[: -len(".whl")]
    parts = stem.split("-")
    # name-version-pythontag-abitag-platformtag; name/version carry no '-'.
    return parts[1] if len(parts) >= 2 else ""


def wheel_tags(filename: str) -> "tuple[str, str, str]":
    """(python-tag, abi-tag, platform-tag) from a wheel filename.

    e.g. 'apex-0.1-cp312-cp312-linux_x86_64.whl' -> ('cp312', 'cp312', 'linux_x86_64')
         'flashinfer_jit_cache-0.6.13+cu130-cp39-abi3-manylinux_2_28_x86_64.whl'
            -> ('cp39', 'abi3', 'manylinux_2_28_x86_64')
         'megatron_bridge-0.6.2-py3-none-any.whl' -> ('py3', 'none', 'any')
    """
    stem = filename[: -len(".whl")]
    parts = stem.split("-")
    platform = parts[-1]
    abi = parts[-2] if len(parts) >= 2 else "none"
    pytag = parts[-3] if len(parts) >= 3 else "py3"
    return pytag, abi, platform


def wheel_supports(filename: str, python: str, arch: str) -> bool:
    """Whether a wheel can install on the given CPython version and CPU arch.

    Used by the coverage gate. Platform is an exact arch match or the 'any'
    universal tag. Python matches on an exact cp<XY> tag, a forward-compatible
    stable-ABI 'abi3' wheel (cp39-abi3 runs on 3.11/3.12), or a universal
    'py3' wheel.
    """
    pytag, abi, platform = wheel_tags(filename)
    if platform != "any" and arch not in platform:
        return False
    cp = f"cp{str(python).replace('.', '')}"
    return pytag == "py3" or abi == "abi3" or pytag == cp


def hash_fragment(digest: "str | None") -> str:
    """PEP 503 '#sha256=<hex>' href fragment from a GitHub asset digest.

    GitHub reports digests as 'sha256:<hex>'; only sha256 is emitted (pip's
    --require-hashes / uv honor it). Wheels without a recorded digest get no
    fragment rather than a guessed one.
    """
    if digest and digest.startswith("sha256:"):
        return f"#{digest}"
    return ""


def dedupe_universal(world: str, assets: List[WheelAsset], agnostic: set) -> List[WheelAsset]:
    """Collapse mirrored same-version universal wheels to one href per world.

    A world-agnostic distribution is mirrored into every world, so a single
    version can be reachable from several releases (e.g. a wheel rebuilt for a
    newer torch world, plus the original). Their filenames are identical, so a
    simple index lists duplicate hrefs for one version. We keep exactly one,
    choosing deterministically:

      1. the release whose title declares THIS world (the wheel's real home);
      2. otherwise the lexicographically greatest release tag (reproducible,
         and newer torch/cuda tags sort ahead).

    Wheels with differing digests are byte-different despite the identical
    name/version; that is reported loudly instead of being hidden.
    """
    groups: "dict[tuple[str, str], List[WheelAsset]]" = defaultdict(list)
    passthrough: List[WheelAsset] = []
    for a in assets:
        dist = wheel_distribution(a["name"])
        if is_universal_wheel(a["name"]) and dist in agnostic:
            groups[(dist, wheel_version(a["name"]))].append(a)
        else:
            passthrough.append(a)
    out = list(passthrough)
    for (_dist, _ver), group in groups.items():
        chosen = sorted(group, key=lambda g: (world in g["worlds"], g.get("tag", "")))[-1]
        out.append(chosen)
    return out


def warn_universal_digest_collisions(assets: List[WheelAsset], agnostic: set) -> None:
    """Report once per (dist, version) when same-named universal wheels differ
    in digest across releases (zip-timestamp-only rebuilds differ here too -
    the index picks one deterministically via dedupe_universal)."""
    seen: "dict[tuple[str, str], dict[str, str]]" = defaultdict(dict)
    for a in assets:
        if not (is_universal_wheel(a["name"]) and wheel_distribution(a["name"]) in agnostic):
            continue
        d = a.get("digest")
        if d:
            seen[(wheel_distribution(a["name"]), wheel_version(a["name"]))][a.get("tag", "?")] = d
    for (dist, ver), by_tag in seen.items():
        if len(set(by_tag.values())) > 1:
            detail = ", ".join(f"{tag}={digest}" for tag, digest in sorted(by_tag.items()))
            print(
                f"warning: universal wheel {dist}=={ver} ships from {len(by_tag)} releases "
                f"with DIFFERENT digests (indexing one per world): {detail}",
                file=sys.stderr,
            )


def world_agnostic_distributions(versions: Dict[str, object]) -> set:
    """Normalized distributions mirrored into every index world.

    Sourced from `world_agnostic: true` components in versions.yaml and their
    `wheel_packages`. Only their py3-none-any wheels are actually mirrored
    (see assign_worlds); a compiled/data wheel is never promoted even if the
    flag is set, so flashinfer-style data wheels stay world-bound.
    """
    agnostic: set = set()
    for cfg in versions.get("components", {}).values():  # type: ignore[union-attr]
        if cfg.get("world_agnostic") is True:  # type: ignore[union-attr]
            agnostic.update(normalize(p) for p in cfg.get("wheel_packages", []))  # type: ignore[union-attr]
    return agnostic


def assign_worlds(asset: WheelAsset, all_worlds: List[str], agnostic: set) -> List[str]:
    """Index worlds one asset belongs to.

    World-agnostic py3-none-any wheels go to every world; every other wheel
    stays on the world(s) its release title declares.
    """
    if is_universal_wheel(asset["name"]) and wheel_distribution(asset["name"]) in agnostic:
        return list(all_worlds)
    return asset["worlds"]


def cuda_tag(cuda: str) -> str:
    """CUDA version as an index path segment: "13.0.2" -> "cu130"."""
    major_minor = "".join(str(cuda).split(".")[:2])
    return f"cu{major_minor}"


def torch_tag(torch: str) -> str:
    """Torch version as an index path segment: "2.13.0" -> "torch2.13".

    Patch releases keep libtorch's ABI, so wheels are interchangeable within
    a major.minor - the same granularity download.pytorch.org and
    flashinfer.ai use for their index paths.
    """
    major_minor = ".".join(str(torch).split(".")[:2])
    return f"torch{major_minor}"


def world_path(cuda: str, torch: str) -> str:
    """Index subtree one (CUDA, torch) pair is published under."""
    return f"{cuda_tag(cuda)}/{torch_tag(torch)}"


def release_worlds(title: str) -> List[str]:
    """The CUDA/torch worlds a release's title says its wheels were built for."""
    return list(
        dict.fromkeys(
            world_path(match["cuda"], match["torch"])
            for match in _TITLE_SEGMENT_RE.finditer(title or "")
        )
    )


def _gh_json(*args: str) -> object:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def list_wheel_assets(repo: str) -> List[WheelAsset]:
    """Enumerate every .whl asset across every release in the repo.

    A release whose title carries no parseable "cu.. py.. torch.." segment
    cannot be placed in a world, so its wheels are reported and left out of
    every index rather than guessed into one: an unlisted wheel is a visible
    404, a misfiled one installs and then fails at import with an ABI error.
    """
    releases = _gh_json(
        "release", "list", "--repo", repo, "--json", "tagName,name", "--limit", "1000"
    )
    assets: List[WheelAsset] = []
    for release in releases:  # type: ignore[union-attr]
        tag = release["tagName"]
        title = release.get("name") or ""
        worlds = release_worlds(title)
        compat_alias = title.lstrip().startswith(COMPAT_ALIAS_PREFIX)
        view = _gh_json("release", "view", tag, "--repo", repo, "--json", "assets")
        wheels = [
            asset
            for asset in view.get("assets", [])  # type: ignore[union-attr]
            if asset["name"].endswith(".whl")
        ]
        if wheels and not worlds:
            if compat_alias:
                # Deliberate URL-only compatibility shim; skip quietly so its
                # presence doesn't dilute the fail-closed warning below.
                continue
            print(
                f"Release {tag!r} has {len(wheels)} wheel(s) but no CUDA/torch segment "
                f"in its title ({title!r}); leaving them out of the index.",
                file=sys.stderr,
            )
            continue
        for asset in wheels:
            assets.append(
                {
                    "name": asset["name"],
                    "url": asset["url"],
                    "worlds": worlds,
                    "tag": tag,
                    "digest": asset.get("digest"),
                }
            )
    return assets


def group_by_world(
    assets: List[WheelAsset],
    all_worlds: List[str],
    agnostic: set,
) -> Dict[str, List[WheelAsset]]:
    by_world: Dict[str, List[WheelAsset]] = defaultdict(list)
    grouped: Dict[str, List[WheelAsset]] = defaultdict(list)
    for asset in assets:
        for world in assign_worlds(asset, all_worlds, agnostic):
            grouped[world].append(asset)
    # One href per (agnostic-distribution, version) per world even when several
    # mirrored releases ship the same universal filename.
    for world, world_assets in grouped.items():
        by_world[world] = dedupe_universal(world, world_assets, agnostic)
    return by_world


def group_by_package(assets: List[WheelAsset]) -> Dict[str, List[WheelAsset]]:
    by_package: Dict[str, List[WheelAsset]] = defaultdict(list)
    for asset in assets:
        # Per the wheel filename spec, the distribution name is everything
        # before the first "-" (version/build/tag segments follow).
        distribution = asset["name"].split("-")[0]
        by_package[normalize(distribution)].append(asset)
    return by_package


def render_page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "  <head>\n"
        '    <meta charset="utf-8">\n'
        f"    <title>{html.escape(title)}</title>\n"
        "  </head>\n"
        "  <body>\n"
        f"{body}\n"
        "  </body>\n"
        "</html>\n"
    )


def write_simple_tree(assets: List[WheelAsset], simple_dir: Path, label: str) -> Dict[str, List[WheelAsset]]:
    """Write one PEP 503 "simple" tree (root listing + one page per package)."""
    by_package = group_by_package(assets)
    simple_dir.mkdir(parents=True, exist_ok=True)

    for package, package_assets in by_package.items():
        pkg_dir = simple_dir / package
        pkg_dir.mkdir(parents=True, exist_ok=True)
        links = "\n".join(
            f'    <a href="{html.escape(a["url"] + hash_fragment(a.get("digest")))}">'
            f"{html.escape(a['name'])}</a><br/>"
            for a in sorted(package_assets, key=lambda a: a["name"])
        )
        (pkg_dir / "index.html").write_text(render_page(package, links))

    root_links = "\n".join(
        f'    <a href="{html.escape(pkg)}/">{html.escape(pkg)}</a><br/>'
        for pkg in sorted(by_package)
    )
    (simple_dir / "index.html").write_text(render_page(label, root_links))
    return by_package


def all_declared_worlds(assets: List[WheelAsset]) -> List[str]:
    """Every world any release title declares, in first-seen order."""
    return list(dict.fromkeys(world for asset in assets for world in asset["worlds"]))


def build_index(
    assets: List[WheelAsset],
    out_dir: Path,
    default_world: str,
    agnostic: set,
) -> Dict[str, List[WheelAsset]]:
    """Publish one index per CUDA/torch world, plus the default-world alias."""
    declared = all_declared_worlds(assets)
    by_world = group_by_world(assets, declared, agnostic)

    for world, world_assets in by_world.items():
        write_simple_tree(
            world_assets,
            out_dir / Path(world) / "simple",
            f"verl-wheelhouse {world} simple index",
        )

    # /simple/ keeps serving the world versions.yaml currently builds, so URLs
    # published before the split (and anything that just wants "what the
    # wheelhouse builds today") keep working. Pin the world path instead to
    # stay on one torch across a future bump.
    write_simple_tree(
        by_world.get(default_world, []),
        out_dir / "simple",
        f"verl-wheelhouse simple index ({default_world})",
    )

    worlds_list = "\n".join(
        f'      <li><code>{html.escape(world)}/simple/</code> - '
        f"{len(group_by_package(by_world[world]))} package(s)"
        f"{' (also served at <code>/simple/</code>)' if world == default_world else ''}</li>"
        for world in sorted(by_world)
    )
    landing_body = (
        "    <h1>verl-wheelhouse</h1>\n"
        "    <p>Prebuilt CUDA wheels for apex, TransformerEngine, flash-attention, "
        "flashinfer, Megatron-Bridge, DeepEP and FlashMLA, built by "
        '<a href="https://github.com/verl-project/verl">verl</a>\'s wheelhouse CI.</p>\n'
        "    <p>Wheels are ABI-bound to the CUDA and torch they were built "
        "against, and their filenames record neither, so each combination has "
        "its own index. Pick the one matching your environment:</p>\n"
        "    <ul>\n"
        f"{worlds_list}\n"
        "    </ul>\n"
        "    <p>Install with:</p>\n"
        "    <pre>pip install --extra-index-url &lt;this-pages-url&gt;/"
        f"{html.escape(default_world)}/simple/ &lt;package&gt;</pre>\n"
        f'    <p><a href="{html.escape(default_world)}/simple/">Browse the current '
        "index</a></p>\n"
    )
    (out_dir / "index.html").write_text(render_page("verl-wheelhouse", landing_body))

    return by_world


def default_world(versions: Dict[str, object]) -> str:
    """The world /simple/ aliases.

    Explicit is best: versions.yaml's `index.default_world` {cuda,torch} names
    it, so reordering build_matrix rows can't silently move /simple to another
    ABI. Falls back to the first build_matrix row for older versions files.
    """
    index = versions.get("index", {})  # type: ignore[union-attr]
    explicit = (index or {}).get("default_world") if isinstance(index, dict) else None
    if isinstance(explicit, dict) and explicit.get("cuda") and explicit.get("torch"):
        return world_path(str(explicit["cuda"]), str(explicit["torch"]))
    matrix = versions["build_matrix"]  # type: ignore[index]
    first = matrix[0]  # type: ignore[index]
    return world_path(str(first["cuda"]), str(first["torch"]))


def coverage_gaps(
    versions: Dict[str, object],
    by_world: Dict[str, List[WheelAsset]],
) -> List[Dict[str, str]]:
    """Expected (world, package, python, arch) cells with no installable wheel.

    The expected matrix is derived exactly like the build matrix is: each
    component's component_combos() after its arches/python_versions/
    cuda_versions filters, crossed with its wheel_packages. A cell is covered
    when some wheel indexed in that world matches the package and supports the
    combo's CPython + arch (native cp tag, forward-compatible abi3, or a
    universal py3-none-any wheel). This turns a silently partial world - one
    failed arch build with the others green - into a hard publish failure.
    """
    gaps: List[Dict[str, str]] = []
    components = versions.get("components", {})  # type: ignore[union-attr]
    for component, cfg in components.items():  # type: ignore[union-attr]
        packages = [normalize(p) for p in cfg.get("wheel_packages", [])]  # type: ignore[union-attr]
        if not packages:
            continue
        for combo in component_combos(versions, component):
            arch, python = combo_arch(combo), combo_python(combo)
            world = world_path(combo_cuda(combo), combo_torch(combo))
            present = by_world.get(world, [])
            for package in packages:
                covered = any(
                    wheel_distribution(a["name"]) == package
                    and wheel_supports(a["name"], python, arch)
                    for a in present
                )
                if not covered:
                    gaps.append(
                        {"world": world, "component": str(component), "package": package,
                         "python": str(python), "arch": arch}
                    )
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. verl-project/verl-wheelhouse")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--default-world",
        default=None,
        help=(
            "Index subtree to also serve at /simple/, e.g. cu130/torch2.13. "
            "Defaults to versions.yaml's index.default_world, else the first "
            "build_matrix row."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Publish even when the expected (component, python, arch) coverage "
            "matrix has gaps. Default is fail-closed: a partial world blocks the "
            "publish instead of silently serving an index with a missing wheel."
        ),
    )
    args = parser.parse_args()

    versions = load_versions()
    world = args.default_world or default_world(versions)
    agnostic = world_agnostic_distributions(versions)
    if agnostic:
        print(f"World-agnostic distributions (mirrored into every world): "
              f"{', '.join(sorted(agnostic))}", file=sys.stderr)

    assets = list_wheel_assets(args.repo)
    if not assets:
        print("No .whl release assets found yet; publishing an empty index.", file=sys.stderr)

    warn_universal_digest_collisions(assets, agnostic)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_world = build_index(assets, args.out_dir, world, agnostic)
    if world not in by_world:
        print(
            f"No release wheels for the default world {world!r}; /simple/ is empty "
            "(worlds found: " + (", ".join(sorted(by_world)) or "none") + ").",
            file=sys.stderr,
        )

    gaps = coverage_gaps(versions, by_world)
    if gaps:
        for g in gaps:
            print(
                f"coverage gap: {g['world']} {g['package']} has no wheel for "
                f"python {g['python']} / {g['arch']} (component {g['component']})",
                file=sys.stderr,
            )
        if not args.allow_partial:
            print(
                f"FAIL: {len(gaps)} expected wheel(s) missing. Rebuild the failed "
                "job(s), or rerun with --allow-partial to publish anyway.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"warning: publishing despite {len(gaps)} coverage gap(s) "
            "(--allow-partial set).",
            file=sys.stderr,
        )

    summary = ", ".join(
        f"{name} ({len(group_by_package(world_assets))} package(s))"
        for name, world_assets in sorted(by_world.items())
    )
    print(f"Indexed {len(assets)} wheel(s) across {len(by_world)} world(s): {summary or 'none'}.")


if __name__ == "__main__":
    main()
