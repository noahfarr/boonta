#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build
CFLAGS="-std=c99 -O2 -fno-semantic-interposition -DNDEBUG -fPIC"
if [ ! -f build/peanut.o ] || [ peanut.c -nt build/peanut.o ] || [ vendor/peanut_gb.h -nt build/peanut.o ] || [ profile.c -nt build/peanut.o ]; then
    rm -f build/peanut.gcda
    if gcc $CFLAGS -fprofile-generate -c peanut.c -o build/peanut.o \
        && gcc -std=c99 -O2 -DNDEBUG -fprofile-generate profile.c build/peanut.o -o build/profile \
        && ./build/profile ../roms/pokemon_red.gb 3000 > /dev/null; then
        gcc $CFLAGS -fprofile-use -fprofile-correction -c peanut.c -o build/peanut.o
    else
        echo "profile run failed, building without profile guidance" >&2
        gcc $CFLAGS -c peanut.c -o build/peanut.o
    fi
fi

: "${JAXLIB_INCLUDE:=$(python3 -c 'import jax.ffi; print(jax.ffi.include_dir())')}"
g++ -std=c++17 -O2 -DNDEBUG -fPIC -fopenmp -shared \
    -I. -I"$JAXLIB_INCLUDE" \
    pool.cpp ffi.cc build/peanut.o -o libpokemon.so
echo "built $(pwd)/libpokemon.so"
