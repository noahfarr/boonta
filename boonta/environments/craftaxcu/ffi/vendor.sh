#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

remote="${CRAFTAXCU_REMOTE:-https://github.com/Infatoshi/craftax.cu}"
pin="${CRAFTAXCU_COMMIT:-13202363e1f7a4bb0e7b95255b6425c5eab16f07}"

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
git clone --quiet --no-checkout "$remote" "$scratch"
git -C "$scratch" checkout --quiet "$pin"

rm -rf vendor
mkdir -p vendor
cp "$scratch"/craftax_full.cu vendor/craftax.cu
cp "$scratch"/LICENSE "$scratch"/NOTICE vendor/

patch -s -p1 -d vendor < craftax.patch

echo "$pin" > vendor/.commit
echo "pinned $pin"
