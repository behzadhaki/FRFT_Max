#!/bin/bash
# Shell script to compile and run frft_test_fft_comparison (macOS only)
#
# Usage: ./run_test_fft_comparison.sh [options]
#
# Options are passed directly to the test program. Common options:
#   --quick              Run quick test with fewer parameters
#   --help               Show detailed help
#
# Examples:
#   ./run_test_fft_comparison.sh                    # Run with defaults
#   ./run_test_fft_comparison.sh --quick            # Run quick test
#
# Note: FFT is normalized by sqrt(N) before comparison with FRFT

set -e  # Exit on error

echo "🔨 Compiling frft_test_fft_comparison..."

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
    "$EVAL_DIR/frft_test_fft_comparison.cpp" \
    "$FFTW3_LIB" \
    -lm \
    -o "$EVAL_DIR/frft_test_fft_comparison"

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Running test..."
    echo "================================================"
    # Pass all command-line arguments to the test program
    "$EVAL_DIR/frft_test_fft_comparison" "$@"
else
    echo "❌ Compilation failed!"
    exit 1
fi