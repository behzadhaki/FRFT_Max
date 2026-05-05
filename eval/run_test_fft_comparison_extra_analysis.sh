#!/bin/bash
# Shell script to compile and run frft_test_fft_comparison_extra_analysis (macOS only)
#
# Usage: ./run_test_fft_comparison_extra_analysis.sh [options]
#
# Options are passed directly to the test program:
#   --exact-bin-only     Run only the exact-bin sine test
#   --impulse-only       Run only the impulse response test
#   --quick              Small window sizes for a fast smoke test
#   --sample-rate SR     Sample rate in Hz (default: 44100)
#   --n-analysis N       Max frames per sine test (default: 20)
#   --output-dir DIR     Base output directory
#   --help               Show detailed help
#
# Two test types are produced:
#   exact_bin/  — sine waves whose frequency falls exactly on an FFT bin
#                 (no spectral leakage); up to 1000 log-uniform bin freqs per window
#   impulse/    — unit impulse (delta at sample 0); FRFT result compared both
#                 against FFT and against the analytical ground truth (1/sqrt(N))
#
# Note: FFT is normalised by sqrt(N) before comparison with FRFT.

set -e  # Exit on error

echo "🔨 Compiling frft_test_fft_comparison_extra_analysis..."

# Find FFTW3 paths (Homebrew)
FFTW3_INCLUDE=""
FFTW3_LIB=""

if [ -d "/opt/homebrew/opt/fftw" ]; then
    FFTW3_INCLUDE="/opt/homebrew/opt/fftw/include"
    FFTW3_LIB="/opt/homebrew/opt/fftw/lib/libfftw3.a"
elif [ -d "/opt/homebrew/include" ] && [ -f "/opt/homebrew/lib/libfftw3.a" ]; then
    FFTW3_INCLUDE="/opt/homebrew/include"
    FFTW3_LIB="/opt/homebrew/lib/libfftw3.a"
elif [ -d "/usr/local/opt/fftw" ]; then
    FFTW3_INCLUDE="/usr/local/opt/fftw/include"
    FFTW3_LIB="/usr/local/opt/fftw/lib/libfftw3.a"
elif [ -d "/usr/local/include" ] && [ -f "/usr/local/lib/libfftw3.a" ]; then
    FFTW3_INCLUDE="/usr/local/include"
    FFTW3_LIB="/usr/local/lib/libfftw3.a"
else
    echo "❌ Error: FFTW3 not found!"
    echo "Install with: brew install fftw"
    exit 1
fi

echo "✅ Found FFTW3:"
echo "   Include: $FFTW3_INCLUDE"
echo "   Library: $FFTW3_LIB"

# Set directories (assumes script is run from eval/ directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
EVAL_DIR="$SCRIPT_DIR"

# Compile
clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I"$SRC_DIR" \
    -I"$FFTW3_INCLUDE" \
    "$SRC_DIR/frft_engine.cpp" \
    "$EVAL_DIR/frft_test_fft_comparison_extra_analysis.cpp" \
    "$FFTW3_LIB" \
    -lm \
    -o "$EVAL_DIR/frft_test_fft_comparison_extra_analysis"

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Running extra analysis tests..."
    echo "================================================"
    "$EVAL_DIR/frft_test_fft_comparison_extra_analysis" "$@"
else
    echo "❌ Compilation failed!"
    exit 1
fi