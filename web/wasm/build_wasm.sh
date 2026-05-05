#!/usr/bin/env bash
# Compiles frft_engine_wasm.cpp + frft_bindings.cpp + KissFFT → WASM.
# Output: dist/frft.js  +  dist/frft.wasm
#
# Usage:
#   ./build_wasm.sh           # release build (-O2)
#   ./build_wasm.sh debug     # debug build   (-O0, assertions on)
#
# Requires emcc on PATH — activate emsdk first:
#   source /path/to/emsdk/emsdk_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Preflight checks ─────────────────────────────────────────────────────────

if ! command -v emcc &>/dev/null; then
    echo ""
    echo "  emcc not found."
    echo "  Install and activate the Emscripten SDK first:"
    echo ""
    echo "    git clone https://github.com/emscripten-core/emsdk.git ~/emsdk"
    echo "    cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest"
    echo "    source ~/emsdk/emsdk_env.sh"
    echo ""
    exit 1
fi

if [ ! -f "kissfft/kiss_fft.h" ]; then
    echo "KissFFT not found — run 'make' first to fetch it."
    exit 1
fi

# ── Build mode ───────────────────────────────────────────────────────────────

MODE="${1:-release}"

if [ "$MODE" = "debug" ]; then
    OPT_FLAGS="-O0 -g"
    ASSERT_FLAGS="-s ASSERTIONS=2"
    OUT_DIR="dist/debug"
    echo "Building debug WASM..."
else
    OPT_FLAGS="-O2"
    ASSERT_FLAGS="-s ASSERTIONS=0"
    OUT_DIR="dist"
    echo "Building release WASM..."
fi

mkdir -p "$OUT_DIR"

# ── Compile ──────────────────────────────────────────────────────────────────

# Compile KissFFT as C (no -std=c++17, no linker flags)
emcc \
    $OPT_FLAGS \
    -Dkiss_fft_scalar=double \
    -I. \
    -c kissfft/kiss_fft.c \
    -o "$OUT_DIR/kiss_fft.o"

# Compile C++ sources and link everything
emcc \
    -std=c++17 \
    $OPT_FLAGS \
    $ASSERT_FLAGS \
    --bind \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME='FRFTModule' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s ENVIRONMENT='web,worker' \
    -s EXPORTED_RUNTIME_METHODS='["HEAPF64"]' \
    -s NO_EXIT_RUNTIME=1 \
    -Dkiss_fft_scalar=double \
    -I. \
    frft_engine_wasm.cpp \
    frft_bindings.cpp \
    "$OUT_DIR/kiss_fft.o" \
    -o "$OUT_DIR/frft.js"

echo ""
echo "Done:"
echo "  $OUT_DIR/frft.js"
echo "  $OUT_DIR/frft.wasm"
echo ""
echo "Load in JS with:"
echo "  import FRFTModule from './frft.js';"
echo "  const Module = await FRFTModule();"
echo "  const proc = new Module.FRFTProcessor();"
