# DeepGEMM wheels

The `deep-gemm` component builds upstream DeepSeek-AI/DeepGEMM at
`5081f8513af57e61696de4f4f04b0e707ffb503e`, including its pinned CUTLASS and
DeepJIT submodules. The wheel's distribution version is `2.8.2+5081f85`.
The git revision is retained in both the filename and package metadata.

## Build and runtime compatibility

DeepGEMM initially builds only VeOmni's current GPU dependency stack,
selected with component-level `torch_versions` and `cuda_versions` filters:

| PyTorch | CUDA toolkit | Installation index world |
|---|---|---|
| 2.11.0 | 13.0.2 | `cu130/torch2.11` |

It builds for Python 3.11/3.12 and native x86_64/aarch64, with CXX11 ABI
enabled: 4 wheels across 2 releases. Releases are separate for each Python
version; CPU architecture is encoded in the wheel filename. These match
VeOmni's supported Python versions and GPU dependency wheel platforms.
Additional torch/CUDA versions require an explicit filter update after
downstream validation; changes to the global matrix alone do not opt this
component into a new torch/CUDA stack.

The builder uses Ubuntu 24.04, GCC 13 and `libdw-dev`. It produces native
`linux_x86_64` / `linux_aarch64` wheels, **not audited manylinux wheels**.
Use a compatible glibc/libstdc++ runtime, such as the same Ubuntu 24.04
base. Wheel tags do not enforce torch, CUDA or libstdc++ compatibility.
Validate other runtime images before deploying these extensions there.

The wheel packages the host extension and the headers needed by DeepJIT.
It saves the source checkout and C++ extension build at installation time;
GPU kernels still compile on first use of a new JIT specialization. Keep
a compatible CUDA toolkit (including NVCC) and C++20 toolchain available
at runtime. The default JIT cache is `~/.dj`; `DG_JIT_CACHE_DIR` can select
another writable directory. Build-machine JIT settings are not embedded
in the wheel.

Hopper and Blackwell dispatch are in the same source package; model shapes
do not require separate wheels. CPU architecture, Python ABI and the
torch/CUDA stack do. This does not extend upstream's supported devices or
operator layouts. `torch_cuda_arch_list` is `null` because this component
does not bake GPU cubins into a fatbin at wheel-build time.

## Installation after publication

Provision torch 2.11.0+cu130 and its matching runtime toolchain, then use
the explicit index for this stack:

```bash
pip install --no-deps 'deep-gemm==2.8.2+5081f85' \
  --index-url https://verl-project.github.io/verl-wheelhouse/cu130/torch2.11/simple/
```

This command becomes available after the component's builds and index
publication succeed. The default `/simple/` index follows the configured
default world, currently `cu132/torch2.13`; use an explicit world URL to
avoid changing ABI stacks implicitly. `--no-deps` assumes runtime
dependencies are already installed; it does not make an incompatible
environment compatible.

## Validation and release

`build-deep-gemm.yml` supports manual architecture and Python filters.
Manual runs build artifacts without publishing; successful pushes to
`main` publish through the shared component release workflow. `build-all`
and the tag-triggered release sweep discover this component automatically.

The builder checks the resulting wheel's native extension, revision and
DeepGEMM/CUTLASS/CuTe header payload. CPU-only hosted runners do not execute
GPU kernels. Before consuming a release for training, install the actual
wheel outside its source checkout and validate with a fresh JIT cache on
Hopper and Blackwell, including grouped forward, dgrad and wgrad. A wheel
build alone is not numerical or performance validation.
