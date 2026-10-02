#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
iree_rev=2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d
source_dir="$root/.cache/iree-src"
build_dir="$root/.cache/iree-build"
mkdir -p "$root/.cache"
if [ ! -d "$source_dir/.git" ]; then
  git init "$source_dir"
  git -C "$source_dir" remote add origin https://github.com/iree-org/iree.git
fi
if [ "$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || true)" != "$iree_rev" ]; then
  git -C "$source_dir" fetch --depth 1 origin "$iree_rev"
  git -C "$source_dir" checkout --detach "$iree_rev"
fi
git -C "$source_dir" submodule update --init --depth 1
"$root/.venv/bin/cmake" --fresh -G Ninja -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="${IREE_C_COMPILER:-gcc}" -DCMAKE_CXX_COMPILER="${IREE_CXX_COMPILER:-g++}" \
  -DCMAKE_MAKE_PROGRAM="$root/.venv/bin/ninja" \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_CMAKE_PLUGIN_PATHS="$root/compiler" \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=OFF \
  -DIREE_TARGET_BACKEND_CUDA=ON -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON -DIREE_HAL_DRIVER_CUDA=ON
"$root/.venv/bin/cmake" --build "$build_dir" --target iree-compile iree-opt --parallel "${BUILD_JOBS:-8}"
