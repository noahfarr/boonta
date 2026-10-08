#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${JAXLIB_INCLUDE:=$(python3 -c 'import jax.ffi; print(jax.ffi.include_dir())')}"
gcc -std=c11 -O3 -march=native -fPIC -fopenmp -c kaggriculture.c -o kaggriculture.o
g++ -std=c++20 -O3 -DNDEBUG -fPIC -fopenmp -shared -I. -I"$JAXLIB_INCLUDE" \
    kaggriculture.o ffi.cc -o libkaggriculture.so -lm
echo "built $(pwd)/libkaggriculture.so"
