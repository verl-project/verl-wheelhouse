#!/usr/bin/env bash
# Wheel post-processing shared by the deep-ep and flash-mla builders.
# Sourced, not executed. Part of only those builders' build-input fingerprint.

# ---------------------------------------------------------------------------
# strip_wheel_local_version: rewrite the wheels in a dist dir to drop the PEP
# 440 local version segment their build appended (deep-ep and flash-mla both
# tack "+<short git sha>" onto their own version).
#
# GitHub's release-asset upload rewrites "+" to "." in the stored filename (it
# arrives URL-encoded as a space), so deep_ep-1.2.1+3f601f7-...whl would land
# on the release as deep_ep-1.2.1.3f601f7-...whl - a filename whose version no
# longer matches the METADATA inside, which pip and uv both reject. Neither
# project has an opt-out env var (there is no equivalent of TransformerEngine's
# NVTE_NO_LOCAL_VERSION), so the segment is stripped from the built artifact
# here instead. Nothing is lost: the exact commit a wheel came from is what the
# component's release tag and title record.
#
# Requires the `wheel` package, which every caller installs anyway.
# ---------------------------------------------------------------------------
strip_wheel_local_version() {
  local dist_dir="${1:-dist}"
  local wheel unpack_dir pkg_dir clean_pkg_dir meta_dir version clean_version

  for wheel in "${dist_dir}"/*.whl; do
    [ -e "${wheel}" ] || continue
    case "$(basename "${wheel}")" in
      *+*) ;;
      *) continue ;;
    esac

    unpack_dir="$(mktemp -d)"
    python -m wheel unpack --dest "${unpack_dir}" "${wheel}"
    pkg_dir="$(find "${unpack_dir}" -mindepth 1 -maxdepth 1 -type d)"
    meta_dir="$(find "${pkg_dir}" -mindepth 1 -maxdepth 1 -type d -name '*.dist-info')"

    version="$(sed -n 's/^Version: //p' "${meta_dir}/METADATA" | head -1)"
    clean_version="${version%%+*}"
    sed "s/^Version: .*/Version: ${clean_version}/" "${meta_dir}/METADATA" > "${meta_dir}/METADATA.rewritten"
    mv "${meta_dir}/METADATA.rewritten" "${meta_dir}/METADATA"

    # The version is spelled out again in the unpacked root's name and in the
    # .dist-info (plus .data, for wheels that have one) directories inside it.
    # Substitute on each basename in turn - a path-wide substitution would
    # rewrite the parent directory's copy of the version instead. `wheel pack`
    # reads the name and version back out of .dist-info to build the new
    # filename, and regenerates RECORD itself.
    clean_pkg_dir="${unpack_dir}/$(basename "${pkg_dir}" | sed "s/${version}\$/${clean_version}/")"
    mv "${pkg_dir}" "${clean_pkg_dir}"
    for meta_dir in "${clean_pkg_dir}"/*.dist-info "${clean_pkg_dir}"/*.data; do
      [ -d "${meta_dir}" ] || continue
      mv "${meta_dir}" "${clean_pkg_dir}/$(basename "${meta_dir}" | sed "s/${version}\./${clean_version}./")"
    done

    echo "Repacking $(basename "${wheel}") without its local version segment (${version} -> ${clean_version})"
    rm -f "${wheel}"
    python -m wheel pack --dest-dir "${dist_dir}" "${clean_pkg_dir}"
    rm -rf "${unpack_dir}"
  done
}
