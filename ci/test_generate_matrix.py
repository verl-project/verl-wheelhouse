#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import generate_matrix


class ExistingReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.versions = {
            "build_matrix": [
                {
                    "cuda": "13.0.2",
                    "python": "3.12",
                    "torch": "2.11.0",
                }
            ],
            "components": {
                "demo": {
                    "ref": "v1.2.3",
                    "wheel_packages": ["demo", "demo-helper"],
                }
            },
        }
        self.release = {
            "name": "demo v1.2.3 - cu13.0.2 py3.12 torch2.11.0",
            "assets": [
                {"name": "demo-1.2.3-cp312-cp312-linux_x86_64.whl"},
                {"name": "demo_helper-1.2.3-py3-none-any.whl"},
            ],
        }

    def test_exact_release_covers_component(self) -> None:
        covered, _ = generate_matrix.release_covers_component(
            self.versions, "demo", self.release
        )
        self.assertTrue(covered)

    def test_dependency_title_must_match_exactly(self) -> None:
        self.release["name"] = "demo v1.2.3 - cu12.8.1 py3.12 torch2.11.0"
        covered, reason = generate_matrix.release_covers_component(
            self.versions, "demo", self.release
        )
        self.assertFalse(covered)
        self.assertIn("title mismatch", reason)

    def test_every_expected_wheel_package_is_required(self) -> None:
        self.release["assets"].pop()
        covered, reason = generate_matrix.release_covers_component(
            self.versions, "demo", self.release
        )
        self.assertFalse(covered)
        self.assertIn("demo-helper", reason)

    def test_cuda_arch_list_is_part_of_the_release_title(self) -> None:
        self.versions["components"]["demo"]["torch_cuda_arch_list"] = "8.0;9.0;10.0"
        covered, reason = generate_matrix.release_covers_component(
            self.versions, "demo", self.release
        )
        self.assertFalse(covered)
        self.assertIn("sm8.0;9.0;10.0", reason)

    def test_cuda_arch_override_is_per_cpu_arch(self) -> None:
        self.versions["build_matrix"].append(
            {
                "arch": "aarch64",
                "cuda": "13.0.2",
                "python": "3.12",
                "torch": "2.11.0",
            }
        )
        self.versions["components"]["demo"]["torch_cuda_arch_list"] = "8.0;9.0;10.0"
        self.versions["components"]["demo"]["arch_overrides"] = {
            "aarch64": {"torch_cuda_arch_list": "9.0;10.0"}
        }
        title = generate_matrix.release_title(
            "v1.2.3",
            "demo",
            generate_matrix.component_combos(self.versions, "demo"),
            self.versions,
        )
        self.assertIn("cu13.0.2 py3.12 torch2.11.0 sm8.0;9.0;10.0", title)
        self.assertIn("aarch64 cu13.0.2 py3.12 torch2.11.0 sm9.0;10.0", title)

    def test_null_cuda_arch_list_keeps_the_existing_title(self) -> None:
        self.versions["components"]["demo"]["torch_cuda_arch_list"] = None
        covered, _ = generate_matrix.release_covers_component(
            self.versions, "demo", self.release
        )
        self.assertTrue(covered)

    @patch("generate_matrix.inspect_release")
    def test_matching_component_is_removed_from_builds(self, inspect_release) -> None:
        inspect_release.return_value = self.release
        needed = generate_matrix.components_needing_build(
            self.versions, ["demo"], "owner/repo"
        )
        self.assertEqual([], needed)
        inspect_release.assert_called_once_with("owner/repo", "demo-v1.2.3")

    @patch("generate_matrix.inspect_release")
    def test_detection_failure_keeps_build(self, inspect_release) -> None:
        inspect_release.return_value = None
        needed = generate_matrix.components_needing_build(
            self.versions, ["demo"], "owner/repo"
        )
        self.assertEqual(["demo"], needed)


if __name__ == "__main__":
    unittest.main()
