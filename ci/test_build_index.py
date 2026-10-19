#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import build_index


def asset(name, worlds, url=None):
    return {"name": name, "url": url or f"https://example.invalid/{name}", "worlds": worlds}


# Two builds of the same component/ref against different torch versions: the
# wheel filename is byte-identical, which is the whole reason the index has to
# be split by world.
APEX_213 = asset("apex-0.1-cp312-cp312-linux_x86_64.whl", ["cu130/torch2.13"])
APEX_211 = asset(
    "apex-0.1-cp312-cp312-linux_x86_64.whl",
    ["cu130/torch2.11"],
    url="https://example.invalid/old/apex-0.1-cp312-cp312-linux_x86_64.whl",
)
BRIDGE_052 = asset("megatron_bridge-0.5.2-py3-none-any.whl", ["cu130/torch2.13"])
BRIDGE_052_211 = asset(
    "megatron_bridge-0.5.2-py3-none-any.whl",
    ["cu130/torch2.11"],
    url="https://example.invalid/old/megatron_bridge-0.5.2-py3-none-any.whl",
)
BRIDGE_062 = asset("megatron_bridge-0.6.2-py3-none-any.whl", ["cu132/torch2.13"])
BRIDGE = BRIDGE_052
AGNOSTIC = {"megatron-bridge"}


class WorldParsingTests(unittest.TestCase):
    def test_title_segments_become_one_world(self) -> None:
        title = (
            "flash-attention v2.8.3 - cu13.0.2 py3.11 torch2.13.0; "
            "aarch64 cu13.0.2 py3.11 torch2.13.0"
        )
        self.assertEqual(["cu130/torch2.13"], build_index.release_worlds(title))

    def test_patch_releases_share_a_world(self) -> None:
        """libtorch keeps its ABI across patch releases, so 2.13.0 and 2.13.1
        wheels are interchangeable and belong in one index."""
        self.assertEqual(
            build_index.release_worlds("demo v1 - cu13.0.2 py3.12 torch2.13.0"),
            build_index.release_worlds("demo v1 - cu13.0.2 py3.12 torch2.13.1"),
        )

    def test_differing_torch_minor_is_a_different_world(self) -> None:
        self.assertNotEqual(
            build_index.release_worlds("demo v1 - cu13.0.2 py3.12 torch2.13.0"),
            build_index.release_worlds("demo v1 - cu13.0.2 py3.12 torch2.11.0"),
        )

    def test_unparseable_title_yields_no_world(self) -> None:
        self.assertEqual([], build_index.release_worlds("hand-made release"))

    def test_tags_drop_dots_for_cuda_and_keep_torch_minor(self) -> None:
        self.assertEqual("cu130", build_index.cuda_tag("13.0.2"))
        self.assertEqual("torch2.13", build_index.torch_tag("2.13.0"))


class IndexLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def build(self, assets, default="cu130/torch2.13", agnostic=None):
        return build_index.build_index(assets, self.out, default, agnostic or set())

    def test_each_world_gets_its_own_tree(self) -> None:
        self.build([APEX_213, APEX_211, BRIDGE])
        for world in ("cu130/torch2.13", "cu130/torch2.11"):
            self.assertTrue((self.out / world / "simple" / "apex" / "index.html").is_file())
        self.assertTrue(
            (self.out / "cu130/torch2.13/simple/megatron-bridge/index.html").is_file()
        )
        # megatron-bridge only ever built against torch 2.13 here, so it must
        # not appear in the 2.11 index.
        self.assertFalse(
            (self.out / "cu130/torch2.11/simple/megatron-bridge").exists()
        )

    def test_same_filename_in_two_worlds_keeps_its_own_url(self) -> None:
        self.build([APEX_213, APEX_211])
        new = (self.out / "cu130/torch2.13/simple/apex/index.html").read_text()
        old = (self.out / "cu130/torch2.11/simple/apex/index.html").read_text()
        self.assertIn(APEX_213["url"], new)
        self.assertNotIn(APEX_211["url"], new)
        self.assertIn(APEX_211["url"], old)
        self.assertNotIn(APEX_213["url"], old)

    def test_default_world_is_also_served_at_simple(self) -> None:
        self.build([APEX_213, APEX_211])
        default = (self.out / "simple" / "apex" / "index.html").read_text()
        self.assertIn(APEX_213["url"], default)
        self.assertNotIn(APEX_211["url"], default)

    def test_default_world_without_wheels_leaves_simple_empty(self) -> None:
        by_world = self.build([APEX_211], default="cu130/torch2.13")
        self.assertEqual(["cu130/torch2.11"], sorted(by_world))
        self.assertTrue((self.out / "simple" / "index.html").is_file())
        self.assertFalse((self.out / "simple" / "apex").exists())

    def test_landing_page_lists_every_world(self) -> None:
        self.build([APEX_213, APEX_211])
        landing = (self.out / "index.html").read_text()
        self.assertIn("cu130/torch2.13/simple/", landing)
        self.assertIn("cu130/torch2.11/simple/", landing)


class WorldAgnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def _page(self, world, package="megatron-bridge"):
        return (self.out / world / "simple" / package / "index.html").read_text()

    def test_agnostic_old_version_is_mirrored_into_the_new_world(self) -> None:
        # The 0.5.2 wheel was published under cu130; once the default world is
        # cu132 it must still be installable from the cu132 index and /simple.
        by_world = build_index.build_index(
            [BRIDGE_052_211, BRIDGE_052, BRIDGE_062, APEX_213],
            self.out,
            "cu132/torch2.13",
            AGNOSTIC,
        )
        for world in ("cu130/torch2.11", "cu130/torch2.13", "cu132/torch2.13"):
            self.assertIn(world, by_world)
            page = self._page(world)
            self.assertIn("megatron_bridge-0.5.2-py3-none-any.whl", page)
        # The current and default worlds carry BOTH versions: newest is the
        # default, an exact ==0.5.2 pin still resolves.
        cu132 = self._page("cu132/torch2.13")
        self.assertIn("megatron_bridge-0.5.2-py3-none-any.whl", cu132)
        self.assertIn("megatron_bridge-0.6.2-py3-none-any.whl", cu132)
        simple = (self.out / "simple" / "megatron-bridge" / "index.html").read_text()
        self.assertIn("megatron_bridge-0.5.2-py3-none-any.whl", simple)
        self.assertIn("megatron_bridge-0.6.2-py3-none-any.whl", simple)

    def test_compiled_wheels_are_not_cross_worlded(self) -> None:
        build_index.build_index(
            [APEX_213, APEX_211, BRIDGE_052, BRIDGE_062],
            self.out,
            "cu132/torch2.13",
            AGNOSTIC,
        )
        apex_213 = self._page("cu130/torch2.13", "apex")
        self.assertIn(APEX_213["url"], apex_213)
        self.assertNotIn(APEX_211["url"], apex_213)
        # apex never built for cu132, so the flag must not create an apex page.
        self.assertFalse((self.out / "cu132/torch2.13/simple/apex").exists())

    def test_non_universal_wheel_is_never_promoted(self) -> None:
        # Even if a flagged distribution ships a platform wheel, it stays bound
        # to its title worlds (guards against a flashinfer-cubin-style misuse).
        compiled = asset("megatron_bridge-9.9-cp312-cp312-linux_x86_64.whl", ["cu130/torch2.13"])
        build_index.build_index(
            [compiled, BRIDGE_062], self.out, "cu132/torch2.13", AGNOSTIC
        )
        # The platform wheel is not promoted; the universal 0.6.2 is mirrored.
        cu132 = self._page("cu132/torch2.13")
        self.assertNotIn("linux_x86_64", cu132)
        self.assertIn("megatron_bridge-0.6.2-py3-none-any.whl", cu132)

    def test_assign_worlds_helper(self) -> None:
        worlds = ["cu130/torch2.11", "cu130/torch2.13", "cu132/torch2.13"]
        self.assertEqual(
            worlds,
            build_index.assign_worlds(BRIDGE_052, worlds, AGNOSTIC),
        )
        self.assertEqual(
            ["cu130/torch2.13"],
            build_index.assign_worlds(APEX_213, worlds, AGNOSTIC),
        )
        # Without the flag the universal wheel keeps its declared worlds.
        self.assertEqual(["cu130/torch2.13"], build_index.assign_worlds(BRIDGE_052, worlds, set()))

    def test_is_universal_wheel(self) -> None:
        self.assertTrue(build_index.is_universal_wheel("megatron_bridge-0.5.2-py3-none-any.whl"))
        self.assertFalse(build_index.is_universal_wheel("apex-0.1-cp312-cp312-linux_x86_64.whl"))

    def test_world_agnostic_distributions_from_versions(self) -> None:
        versions = {
            "components": {
                "megatron-bridge": {
                    "wheel_packages": ["megatron-bridge"],
                    "world_agnostic": True,
                },
                "flashinfer": {
                    "wheel_packages": ["flashinfer-python", "flashinfer-cubin"],
                    # data wheels must not be promoted even if someone flipped this
                },
                "apex": {"wheel_packages": ["apex"]},
            }
        }
        self.assertEqual({"megatron-bridge"}, build_index.world_agnostic_distributions(versions))


class DefaultWorldTests(unittest.TestCase):
    def test_default_world_comes_from_the_first_matrix_row(self) -> None:
        versions = {
            "build_matrix": [
                {"cuda": "13.0.2", "python": "3.11", "torch": "2.13.0"},
                {"cuda": "13.0.2", "python": "3.12", "torch": "2.13.0"},
            ]
        }
        self.assertEqual("cu130/torch2.13", build_index.default_world(versions))

    def test_explicit_default_world_overrides_matrix_order(self) -> None:
        # Reordering the matrix must not move /simple: the explicit field wins.
        versions = {
            "index": {"default_world": {"cuda": "13.2.2", "torch": "2.13.0"}},
            "build_matrix": [
                {"cuda": "13.0.2", "python": "3.11", "torch": "2.11.0"},
                {"cuda": "13.2.2", "python": "3.12", "torch": "2.13.0"},
            ],
        }
        self.assertEqual("cu132/torch2.13", build_index.default_world(versions))


class WheelTagTests(unittest.TestCase):
    def test_wheel_version(self) -> None:
        self.assertEqual("0.6.2", build_index.wheel_version("megatron_bridge-0.6.2-py3-none-any.whl"))
        self.assertEqual(
            "0.6.13+cu130",
            build_index.wheel_version("flashinfer_jit_cache-0.6.13+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"),
        )

    def test_stable_abi3_wheel_supports_newer_cpython(self) -> None:
        f = "flashinfer_jit_cache-0.6.13+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"
        self.assertTrue(build_index.wheel_supports(f, "3.12", "x86_64"))
        self.assertTrue(build_index.wheel_supports(f, "3.11", "x86_64"))
        self.assertFalse(build_index.wheel_supports(f, "3.12", "aarch64"))

    def test_universal_and_native_support(self) -> None:
        any_w = "megatron_bridge-0.6.2-py3-none-any.whl"
        self.assertTrue(build_index.wheel_supports(any_w, "3.11", "x86_64"))
        self.assertTrue(build_index.wheel_supports(any_w, "3.12", "aarch64"))
        native = "apex-0.1-cp312-cp312-linux_x86_64.whl"
        self.assertTrue(build_index.wheel_supports(native, "3.12", "x86_64"))
        self.assertFalse(build_index.wheel_supports(native, "3.11", "x86_64"))
        self.assertFalse(build_index.wheel_supports(native, "3.12", "aarch64"))

    def test_hash_fragment(self) -> None:
        self.assertEqual(
            "#sha256:abc", build_index.hash_fragment("sha256:abc")
        )
        self.assertEqual("", build_index.hash_fragment(None))
        self.assertEqual("", build_index.hash_fragment("md5:abc"))


class HashFragmentIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def test_page_links_carry_sha256_when_digest_known(self) -> None:
        a = asset("apex-0.1-cp312-cp312-linux_x86_64.whl", ["cu130/torch2.13"])
        a.update({"tag": "apex-master-py3.12-torch2.13.0-cu130", "digest": "sha256:deadbeef"})
        build_index.build_index([a], self.out, "cu130/torch2.13", set())
        page = (self.out / "cu130/torch2.13/simple/apex/index.html").read_text()
        self.assertIn(a["url"] + "#sha256:deadbeef", page)
        self.assertNotIn(a["url"] + "#sha256:deadbeef#", page)


class UniversalDedupeTests(unittest.TestCase):
    def _b(self, tag, worlds, digest, name="megatron_bridge-0.6.2-py3-none-any.whl"):
        a = asset(name, worlds, url=f"https://example.invalid/{tag}/{name}")
        a.update({"tag": tag, "digest": digest})
        return a

    def test_home_world_release_wins_over_mirrors(self) -> None:
        home = self._b("mb-v0.6.2-py3.12-torch2.13.0-cu130", ["cu130/torch2.13"], "sha256:home")
        mirror = self._b("mb-v0.6.2-py3.12-torch2.13.0-cu132", ["cu132/torch2.13"], "sha256:mirror")
        out = build_index.dedupe_universal("cu130/torch2.13", [mirror, home], {"megatron-bridge"})
        self.assertEqual([home], out)

    def test_mirrored_world_picks_greatest_tag_deterministically(self) -> None:
        old = self._b("mb-v0.6.2-py3.12-torch2.12.1-cu132", ["cu132/torch2.12"], "sha256:old")
        new = self._b("mb-v0.6.2-py3.12-torch2.13.0-cu132", ["cu132/torch2.13"], "sha256:new")
        out = build_index.dedupe_universal("cu130/torch2.13", [old, new], {"megatron-bridge"})
        self.assertEqual([new], out)

    def test_distinct_versions_are_not_collapsed(self) -> None:
        v052 = self._b("mb-r0.5.0-py3.12-torch2.13.0", ["cu130/torch2.13"], "sha256:a",
                       name="megatron_bridge-0.5.2-py3-none-any.whl")
        v062 = self._b("mb-v0.6.2-py3.12-torch2.13.0-cu132", ["cu132/torch2.13"], "sha256:b")
        out = build_index.dedupe_universal("cu130/torch2.13", [v052, v062], {"megatron-bridge"})
        self.assertEqual([v052["url"], v062["url"]], sorted(a["url"] for a in out))

    def test_build_index_lists_one_href_per_mirrored_version(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        old = self._b("mb-v0.6.2-py3.12-torch2.12.1-cu132", ["cu132/torch2.12"], "sha256:old")
        new = self._b("mb-v0.6.2-py3.12-torch2.13.0-cu132", ["cu132/torch2.13"], "sha256:new")
        build_index.build_index([old, new], out, "cu132/torch2.13", {"megatron-bridge"})
        # In 2.13 the home release (new) wins; in 2.12 the home release (old) wins.
        page213 = (out / "cu132/torch2.13/simple/megatron-bridge/index.html").read_text()
        self.assertEqual(1, page213.count("<a "))
        self.assertIn(new["url"], page213)
        self.assertNotIn(old["url"], page213)
        page212 = (out / "cu132/torch2.12/simple/megatron-bridge/index.html").read_text()
        self.assertEqual(1, page212.count("<a "))
        self.assertIn(old["url"], page212)
        self.assertNotIn(new["url"], page212)


class CoverageGateTests(unittest.TestCase):
    VERSIONS = {
        "build_matrix": [
            {"arch": "x86_64", "cuda": "13.2.2", "python": "3.11", "torch": "2.13.0"},
            {"arch": "x86_64", "cuda": "13.2.2", "python": "3.12", "torch": "2.13.0"},
        ],
        "components": {"apex": {"wheel_packages": ["apex"]}},
    }

    @staticmethod
    def _a(name, world):
        a = asset(name, [world], url=f"https://example.invalid/{name}")
        a.update({"tag": "t", "digest": None})
        return a

    def test_complete_matrix_has_no_gaps(self) -> None:
        by_world = {
            "cu132/torch2.13": [
                self._a("apex-0.1-cp311-cp311-linux_x86_64.whl", "cu132/torch2.13"),
                self._a("apex-0.1-cp312-cp312-linux_x86_64.whl", "cu132/torch2.13"),
            ]
        }
        self.assertEqual([], build_index.coverage_gaps(self.VERSIONS, by_world))

    def test_missing_python_arch_cell_is_a_gap(self) -> None:
        # cp312 wheel is absent: exactly the cu132/torch2.12 flash-attention shape.
        by_world = {
            "cu132/torch2.13": [
                self._a("apex-0.1-cp311-cp311-linux_x86_64.whl", "cu132/torch2.13"),
            ]
        }
        gaps = build_index.coverage_gaps(self.VERSIONS, by_world)
        self.assertEqual(1, len(gaps))
        self.assertEqual("3.12", gaps[0]["python"])
        self.assertEqual("x86_64", gaps[0]["arch"])

    def test_universal_wheel_covers_every_cell(self) -> None:
        versions = {
            "build_matrix": [
                {"arch": "x86_64", "cuda": "13.2.2", "python": "3.12", "torch": "2.13.0"},
                {"arch": "aarch64", "cuda": "13.2.2", "python": "3.12", "torch": "2.13.0"},
            ],
            "components": {"megatron-bridge": {"wheel_packages": ["megatron-bridge"]}},
        }
        by_world = {
            "cu132/torch2.13": [
                self._a("megatron_bridge-0.6.2-py3-none-any.whl", "cu132/torch2.13")
            ]
        }
        self.assertEqual([], build_index.coverage_gaps(versions, by_world))


class CompatAliasTests(unittest.TestCase):
    def test_compat_alias_release_is_skipped_silently(self, ) -> None:
        releases = [
            {"tagName": "apex-master-py3.12-torch2.13.0",
             "name": "[compat alias, not indexed] apex-master-py3.12-torch2.13.0 - canonical: apex-master-py3.12-torch2.13.0-cu130"},
        ]
        view = {"assets": [
            {"name": "apex-0.1-cp312-cp312-linux_x86_64.whl",
             "url": "https://example.invalid/a.whl", "digest": "sha256:x"},
        ]}
        orig = build_index._gh_json
        import io
        import contextlib

        def fake(*args):  # type: ignore[no-untyped-def]
            return view if args[:2] == ("release", "view") else releases
        try:
            build_index._gh_json = fake  # type: ignore[assignment]
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                assets = build_index.list_wheel_assets("o/r")
            self.assertEqual([], assets)
            self.assertNotIn("CUDA/torch segment", err.getvalue())
        finally:
            build_index._gh_json = orig  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
