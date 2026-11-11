#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -f vendor/craftax.cu ] || ./vendor.sh
: "${JAXLIB_INCLUDE:=$(python3 -c 'import jax.ffi; print(jax.ffi.include_dir())')}"
: "${CRAFTAXCU_ARCH:=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}"

nvcc -O3 -arch="sm_${CRAFTAXCU_ARCH}" --expt-relaxed-constexpr -fmad=false \
    -Xcompiler -fPIC -shared \
    -I"$JAXLIB_INCLUDE" \
    binding.cu ffi.cc -o libcraftaxcu.so
echo "built $(pwd)/libcraftaxcu.so"
