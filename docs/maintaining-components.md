# Maintaining verl-wheelhouse components

This guide covers the two most common maintenance tasks for this repo:
bumping an existing component's pinned version, and adding a brand-new
component to the wheelhouse. Both are driven almost entirely by
`versions.yaml` (repo root) plus the small set of files listed below - no
other logic needs to change for routine work.

## Files involved

| File | Role |
|---|---|
| `versions.yaml` | The version map: arch/CUDA/Python/Torch build matrix + per-component config |
| `ci/generate_matrix.py` | Expands `versions.yaml` into the GitHub Actions matrix (no edits needed for routine work) |
| `ci/release_meta.py` | Computes each (component, Python) release's tag/title/notes (no edits needed for routine work) |
| `ci/build_scripts/common.sh` | Shared bash helpers (no edits needed unless adding new shared logic) |
| `ci/build_scripts/<builder>.sh` | The actual wheel-build command for one component |
| `.github/workflows/build-<component>.yml` | Per-component trigger workflow |
| `.github/workflows/_build.yml` | Reusable build workflow (no edits needed) |
| `.github/workflows/_ensure_release.yml` | Legacy helper for 3.12 bare-tag releases only; release creation now runs inline in `_build.yml` (no edits needed) |
| `.github/workflows/build-all.yml`, `release.yml` | Already build every component via `--component all` (no edits needed) |

## Upgrading an existing component's version

1. Open `versions.yaml` and find the component under `components:`.
2. Update its `ref:` to the new git tag/branch/commit you want to pin.
3. Check the upstream project's release notes for anything else that changed:
   - New/removed supported GPU architectures → update `torch_cuda_arch_list`
     (and the copy under `arch_overrides`, if the component has one).
   - New build-time environment variables or flags → update
     `ci/build_scripts/<builder>.sh` and/or `extra_env` in `versions.yaml`.
   - A cuDNN version bump or new system package requirement → update
     `requires_cudnn` and/or `ci/build_scripts/common.sh`'s `install_cudnn`.
4. Regenerate and sanity-check the matrix locally:

   ```bash
   pip install pyyaml
   python3 ci/generate_matrix.py --component <name> | python3 -m json.tool
   ```

5. Commit the `versions.yaml` change. Pushing to `main` automatically
   triggers that component's `build-<component>.yml` (its `paths:` filter
   matches `versions.yaml`). The push skips a matrix row only when *all* of
   the following agree with the target release: the configured
   CUDA/Python/Torch title; the release's `wheelhouse-build-manifest.json`
   asset (a snapshot of every build input - dependency versions,
   `torch_cuda_arch_list`, env vars, the builder command, `max_jobs`,
   `runs_on`, and sha256 fingerprints of `_build.yml`, the builder script,
   `common.sh` and any declared `patches`; a copy is also embedded in the
   release notes as a fallback); the expected `wheel_packages` on both CPU
   arches; and - downloaded and inspected directly - that the published
   wheels' `.nv_fatbin` sections actually cover every SM arch
   `torch_cuda_arch_list` promises. Anything unverifiable (download failure,
   no manifest, old schema) fails closed and rebuilds. Otherwise, on
   success, that push creates (or reuses) that component's per-Python
   releases - tag `<component>-<new-ref>` for the legacy 3.12 interpreter
   and `<component>-<new-ref>-pyX.Y` for every other interpreter, each
   titled `<component> <new-ref> - cu.. pyX.Y torch..` listing only that
   interpreter's combos (see `ci/release_meta.py`) - uploads each new wheel
   to its matching release, and republishes the package index, so it's
   `pip install`-able right away. Bumping `ref` therefore starts brand-new
   releases; the previous ref's releases are left untouched as history. You
   can also trigger the workflow
   manually from the Actions tab (`workflow_dispatch`) to test before
   merging (manual runs build but skip publishing).
6. Once you're happy, push a tag matching `v*` (`git tag vX.Y.Z && git push
   --tags`) to run `release.yml`: a full sweep that rebuilds every
   component and uploads to each (component, Python) release, same as
   step 5 but across the whole matrix at once. The pushed tag is only a
   trigger - it does not itself become a release.

`apex` is the one exception - both verl Dockerfiles this repo mirrors build
it unpinned from `main`, so there's no fixed version to bump; every build of
`apex` simply picks up whatever `main` currently contains.

## Adding a brand-new component

Worked checklist, using a hypothetical `xformers` component as the example:

1. **Add the submodule:**

   ```bash
   git submodule add https://github.com/facebookresearch/xformers.git xformers
   ```

2. **Add a `components:` entry to `versions.yaml`**, following the schema
   documented in that file's own comments:

   ```yaml
   xformers:
     path: xformers
     ref: v0.0.29
     builder: xformers
     wheel_packages: [xformers]
     torch_cuda_arch_list: "8.0;9.0;10.0;12.0"
     requires_cudnn: false
     max_jobs: 2
     runs_on: ubuntu-24.04
     arches: [x86_64]
     extra_env: {}
   ```

   Start with `arches: [x86_64]` and add arm64 as a follow-up once the
   x86_64 build is green - see the next section. No `python_versions` field
   means the component builds for every interpreter in `build_matrix`
   (3.11 and 3.12 today); add `python_versions: ["3.12"]` only if its wheel
   is `py3-none-any` or upstream `requires-python` excludes 3.11.

3. **Write `ci/build_scripts/xformers.sh`.** Use an existing script as a
   template (`ci/build_scripts/apex.sh` is a good default shape); every
   script follows the same pattern:

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   source "${SCRIPT_DIR}/common.sh"

   export_extra_env
   # ... install prerequisite pip packages, export the project's own
   # documented build-time env vars (mirror its own CI/Dockerfile exactly) ...
   python setup.py bdist_wheel --dist-dir=dist   # or: pip wheel -w dist --no-deps .

   echo "Built wheels:"
   ls -al dist
   ```

   Then `chmod +x ci/build_scripts/xformers.sh`. The script always runs with
   its CWD already set to the component's checkout (see `_build.yml`'s
   `working-directory`), and must leave the final wheel(s) in `dist/`
   relative to that CWD - the reusable workflow uploads `<path>/dist/*.whl`.

4. **Copy a per-component workflow.** Duplicate
   `.github/workflows/build-flashinfer.yml` to
   `.github/workflows/build-xformers.yml` and adjust:
   - `name:` → `Build xformers`
   - the `paths:` entries → `versions.yaml`, `ci/build_scripts/common.sh`,
     `ci/build_scripts/xformers.sh`, `.github/workflows/_build.yml`,
     `.github/workflows/build-xformers.yml`
   - the `--component flashinfer` argument in the `compute-matrix` job →
     `--component xformers`

   There is no per-workflow release job: on push builds `_build.yml` creates
   the release inline, keyed by the matrix row's `release_tag`. Everything
   else - `workflow_dispatch`, the reusable `_build.yml` call's structure,
   and the trailing `publish-index` job that runs after a successful push
   build - is component-agnostic and can be copied as-is. `ci/release_meta.py`
   automatically computes `xformers`'s per-Python release tags/titles (bare
   `<component>-<ref>` for 3.12, `<component>-<ref>-pyX.Y` otherwise) once
   its `components:` entry exists in `versions.yaml`.

   If your new component's `runs_on` is **self-hosted**, copy
   `.github/workflows/build-apex.yml` instead: it additionally forwards a
   `secrets: { BYTED_PROXY: ... }` block to `_build.yml`, which routes
   otherwise-slow GitHub uploads through the optional `BYTED_PROXY` repo
   secret. **Drop that block for a normal `ubuntu-*` component** -
   `_build.yml`'s `runner.environment == 'self-hosted'` proxy gate is a no-op
   on GitHub-hosted runners anyway.

5. **Update `README.md`'s component table and repo-layout listing** to
   mention the new component/workflow.

6. **Validate before pushing:**

   ```bash
   pip install pyyaml
   python3 ci/generate_matrix.py --list-components   # should list the new name
   python3 ci/generate_matrix.py --component xformers | python3 -m json.tool
   bash -n ci/build_scripts/xformers.sh
   ```

7. `build-all.yml` and `release.yml` both use `--component all`, so the new
   component is automatically included in the weekly sanity sweep and in
   every future release - no changes needed there.

## Building a component for another CPU arch (arm64)

`build_matrix` carries an `arch` field (`x86_64` / `aarch64`, spelled the way
`uname -m` reports it), and each component chooses which of those arches it
is built for:

```yaml
components:
  flash-attention:
    runs_on: ubuntu-24.04
    torch_cuda_arch_list: "8.0;9.0;10.0;12.0"
    # no `arches` field -> every arch in build_matrix
    arch_overrides:
      aarch64:
        runs_on: ubuntu-24.04-arm
        torch_cuda_arch_list: "9.0;10.0"

  megatron-bridge:
    arches: [x86_64] # opt out of arm64 entirely
```

`flash-attention`, `apex`, `TransformerEngine`, `flashinfer`, `deep-ep`,
`flash-mla` and `fast-hadamard-transform` all build both
arches. `Megatron-Bridge` is the only opt-out, and not for lack of a runner:
it emits a single `py3-none-any` wheel that already installs on arm64, so a
second job would just race an identical asset name onto the same release.

To turn arm64 on for a component:

1. Remove its `arches: [x86_64]` line (or add `aarch64` to the list).
2. Add an `arch_overrides.aarch64` block with, at minimum, an arm64
   `runs_on`. This must be a **GitHub-hosted** runner - `ubuntu-24.04-arm`,
   GitHub's free 4 vCPU arm64 machine - so that arm64 support never depends
   on someone standing up matching hardware first; the one self-hosted
   machine this repo uses is x86_64-only. `ci/generate_matrix.py` refuses to
   emit a non-x86_64 row on a self-hosted runner, and `_build.yml` asserts
   `uname -m` matches the row's arch, so a runner/arch mismatch fails in
   seconds instead of after a multi-hour build.
3. Narrow `torch_cuda_arch_list` for that arch. Every CUDA-capable arm64
   host is a server/superchip part - GH200 (`9.0`) and GB200 (`10.0`) -
   so carrying the x86_64 list's `8.0`/`12.0` only burns runner hours.
4. If the component produces any `py3-none-any` wheel, make the build script
   skip it on arm (check `$TARGET_ARCH`, which `_build.yml` exports).
   Otherwise both arches upload the same asset name onto the same release.
5. Regenerate the matrix and confirm both rows appear with the right runners:

   ```bash
   python3 ci/generate_matrix.py --component <name> | python3 -m json.tool
   python3 ci/release_meta.py --component <name> --python all
   ```

   (`--python all` prints every per-interpreter release; omit it to
   self-select the locally running interpreter.)

6. Smoke-test the new arch on its own before letting a push build both:
   trigger `build-<component>.yml` from the Actions tab and set its **arch**
   input to that arch. `workflow_dispatch` runs build without publishing, and
   the arch filter keeps the other arches off the runners entirely. The same
   filter is available locally:

   ```bash
   python3 ci/generate_matrix.py --component <name> --arch aarch64
   ```

Adding an arch to a component changes each of its per-interpreter release
titles, which is what makes the next push rebuild those releases (the skip
check requires an exact title match *and* a wheel per package per arch,
matched on each wheel's platform tag). Titles leave `x86_64` implicit, so
x86_64-only components keep their existing titles and are not disturbed when
a new arch enters `build_matrix`.

The rest of the toolchain is already arch-agnostic: `Jimver/cuda-toolkit`
switches its apt repo to NVIDIA's `sbsa` path on arm64, `common.sh`'s
`install_cudnn` maps `aarch64` → `sbsa`, `install_nccl` resolves its version
from whatever repo is configured, and `pip install torch --index-url
.../cu130` picks the `manylinux_2_28_aarch64` wheel by itself.

## Building a component for another Python version

`build_matrix` rows carry a quoted `python` field (`"3.11"` / `"3.12"` -
quoted so YAML doesn't parse it as a float), and each component chooses which
interpreters it builds for with `python_versions`. It is the exact Python
counterpart of `arches`:

```yaml
components:
  flash-attention:
    # no `python_versions` field -> every python version in build_matrix

  megatron-bridge:
    python_versions: ["3.12"]  # opt out of 3.11
```

The six CUDA-extension components (flash-attention, apex, TransformerEngine,
deep-ep, flash-mla, fast-hadamard-transform) build for every version in the
matrix because their wheels carry interpreter-specific `cp311`/`cp312` tags.
`megatron-bridge` and `flashinfer` pin `["3.12"]`: their wheels are portable
`py3-none-any`, so one wheel installs on every interpreter and a second build
would only re-upload an identical asset, and Megatron-Bridge's upstream
`requires-python` (">=3.12,<3.13") excludes 3.11 entirely.

To add a Python version:

1. Confirm prerequisites first: PyTorch publishes `cp<abi>` wheels for the
   target CUDA index on **every arch the component builds for**
   (`curl -sI https://download.pytorch.org/whl/cu130/...`), and the
   component itself (and its build script) supports the interpreter.
2. Append one `build_matrix` entry per arch, copying the existing
   CUDA/Torch pins, with `python: "<version>"`.
3. Add the version to the static `options:` list of the `python` dispatch
   input in **every** `.github/workflows/build-*.yml`. GitHub Actions choice
   lists cannot be generated dynamically, so they mirror `build_matrix` by
   hand; `generate_matrix.py` rejects any value the matrix doesn't know.
4. Pin `python_versions: [...]` on any component whose wheels are
   `py3-none-any` or whose upstream `requires-python` excludes the new
   version, so it doesn't re-upload an identical asset or fail outright.
5. Regenerate the matrix and smoke-test one interpreter without touching the
   others, locally or via the workflow's **python** dispatch input:

   ```bash
   python3 ci/generate_matrix.py --component <name> --python 3.11
   python3 ci/generate_matrix.py --component all   # full matrix, e.g. 27 rows
   ```

   An empty result for an opted-out component (e.g.
   `--component megatron-bridge --python 3.11`) is expected; the workflow
   skips its build job via `has_builds`.

Each interpreter gets its **own release**: the interpreter the wheelhouse
shipped before the split (3.12) keeps the bare `<component>-<ref>` tag, and
every other interpreter gets `<component>-<ref>-pyX.Y` (see
`LEGACY_RELEASE_PYTHON` in `ci/generate_matrix.py`); the title lists only
that interpreter's combos. Adding a Python version therefore never edits
the existing releases - their tag, title and manifest stay byte-identical
and their rows keep skipping - while the new interpreter's rows build into
new `-pyX.Y` releases. Components whose `python_versions` excludes the new
version are not disturbed either, which is exactly why the pure-Python
components pin it (they must not upload the same `py3-none-any` asset to
two releases).

## Arch-list conventions

Different build systems want `torch_cuda_arch_list` in different string
shapes; the field in `versions.yaml` is treated as canonical and converted
where needed:

| Consumer | Format | Handled by |
|---|---|---|
| apex, deep-ep | dotted + semicolons, e.g. `8.0;9.0;12.0` | used as-is |
| flash-attention, TransformerEngine | undotted, e.g. `80;90;120` | `ci/build_scripts/common.sh`'s `arch_list_strip_dots` |
| flashinfer | space-separated with PTX-family suffixes, e.g. `8.0 9.0a 12.0f` | given verbatim in `versions.yaml` (suffixes can't be derived mechanically) |
| Megatron-Bridge (pure-Python), flash-mla (hardcodes `sm90a`/`sm100f`), fast-hadamard-transform (derives nine gencodes from the toolkit version) | n/a | set `torch_cuda_arch_list: null` for components that build no CUDA code, or that hardcode their own gencode flags |

The arch list is a promise that gets verified, not just a build flag:

- **Pre-upload gate.** After the build, `_build.yml` runs `ci/cuda_archs.py` over
  `dist/*.whl` and fails the job if the wheels' `.nv_fatbin` sections don't
  contain every declared arch (`cuobjdump` when available, a stdlib ELF
  section scan otherwise - no runner dependency). This catches the
  silent-drop class of failure, e.g. deep-ep's sm_80 cubin, which exists
  only because `ci/patches/enable_deep_ep_sm80.py` rewrites the checkout at
  build time: if that patch ever stops applying, the build fails before
  upload rather than shipping a wheel that crashes on A100. Components
  without an arch list (`null`) skip the gate; so does any component with
  `verify_wheel_archs: false` (currently flashinfer, which rehosts prebuilt
  wheels - including a ~1.2 GB data wheel - instead of compiling).
- **Post-upload re-verification.** Push-time skip detection downloads the
  release's wheels and re-checks the same fatbin coverage, so a wheel whose
  declared arches were never actually fat-binned is rebuilt even though
  `versions.yaml` still promises them.
- **Build-input fingerprinting.** Editing anything that changes wheel
  contents - the builder script, `common.sh`, `.github/workflows/_build.yml`,
  or a file listed in a component's `patches:` - changes a sha256 in the
  release manifest and forces a rebuild, without needing a version bump.
  Declare every patch file the builder applies under `patches:` in
  `versions.yaml` (paths are repo-root-relative and must exist), so the
  fingerprint is complete.

Inspect or gate wheels manually:

```bash
python3 ci/cuda_archs.py dist/*.whl                      # report found arches
python3 ci/cuda_archs.py dist/*.whl --require "8.0;9.0;10.0"  # exit 1 if any is missing
```

## Previewing the build plan on a pull request

Every PR that touches `versions.yaml`, `ci/`, or the workflows runs the
**Build plan dry run** check (`.github/workflows/pr-build-dry-run.yml`): it
compares the PR's matrix against the live releases and posts a markdown table
to the PR (and the job summary) showing, per component/arch row, whether the
next push would **build** or **skip** and why. Nothing is built or published
- the output is for a human to judge, e.g. a ref bump should show builds for
that component only, while a docs-only change should show all skips. The dry
run does not download wheels (`--no-verify-wheels`, wheels can be multi-GB);
fatbin coverage stays gated at build time and re-verified at push time. Run
it locally:

```bash
python3 ci/generate_matrix.py --skip-existing-releases \
  --repo verl-project/verl-wheelhouse --no-verify-wheels --report build-plan.md
```

## See also

- [`README.md`](../README.md) for the full pipeline architecture and how to
  trigger builds / install wheels.
- `versions.yaml`'s inline comments for the authoritative field-by-field
  schema of `build_matrix` and `components`.
