#!/usr/bin/env bash
# CUDA arch-list formatting shared by the flash-attention and
# transformer_engine builders. Sourced, not executed. Part of only those
# builders' build-input fingerprint.

# ---------------------------------------------------------------------------
# arch_list_strip_dots: convert the canonical "8.0;9.0;10.0;12.0" arch list
# into the undotted "80;90;100;120" form that flash-attention's
# FLASH_ATTN_CUDA_ARCHS and TransformerEngine's NVTE_CUDA_ARCHS expect.
# apex consumes TORCH_CUDA_ARCH_LIST in the canonical dotted form directly
# (no conversion needed), and flashinfer's list is given pre-formatted in
# versions.yaml (with PTX-family suffix letters like "9.0a"/"12.0f") since
# those can't be derived mechanically.
# ---------------------------------------------------------------------------
arch_list_strip_dots() {
  echo "$1" | tr -d '.'
}
