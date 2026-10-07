#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

stale=0
if [ ! -f vendor/build/src/ale/libale.a ]; then
    stale=1
elif [ -n "$(find vendor/ale/src -name '*.cxx' -newer vendor/build/src/ale/libale.a -print -quit)" ]; then
    stale=1
fi
if [ "$stale" = 1 ]; then
    cmake -S vendor/ale -B vendor/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CPP_LIB=ON -DBUILD_PYTHON_LIB=OFF -DBUILD_VECTOR_LIB=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON > /dev/null
    cmake --build vendor/build -j"$(nproc)" --target ale ale-lib > /dev/null
    rm -f vendor/build/src/ale/libale.a
    ar rcs vendor/build/src/ale/libale.a $(find vendor/build -name '*.o')
fi

: "${JAXLIB_INCLUDE:=$(python3 -c 'import jax.ffi; print(jax.ffi.include_dir())')}"
g++ -std=c++17 -O3 -DNDEBUG -fPIC -fopenmp -shared \
    -Ivendor/ale/src -Ivendor/ale/src/ale -Ivendor/build/src -Ivendor/build/src/ale \
    -I"$JAXLIB_INCLUDE" \
    vectorize.cpp ffi.cc vendor/build/src/ale/libale.a -o libale_vec.so -lz
echo "built $(pwd)/libale_vec.so"
