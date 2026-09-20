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
BRIDGE = asset("megatron_bridge-0.5.2-py3-none-any.whl", ["cu130/torch2.13"])


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

    def build(self, assets, default="cu130/torch2.13"):
        return build_index.build_index(assets, self.out, default)

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


class DefaultWorldTests(unittest.TestCase):
    def test_default_world_comes_from_the_first_matrix_row(self) -> None:
        versions = {
            "build_matrix": [
                {"cuda": "13.0.2", "python": "3.11", "torch": "2.13.0"},
                {"cuda": "13.0.2", "python": "3.12", "torch": "2.13.0"},
            ]
        }
        self.assertEqual("cu130/torch2.13", build_index.default_world(versions))


if __name__ == "__main__":
    unittest.main()
