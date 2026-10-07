#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

pin="v0.12.0"
rm -rf vendor
git clone --depth 1 --branch "$pin" \
    https://github.com/Farama-Foundation/Arcade-Learning-Environment.git vendor/ale
git -C vendor/ale rev-parse HEAD > vendor/.commit
git -C vendor/ale apply ../../rebase_frame_pointer.patch
echo "pinned $pin"
