#!/bin/bash
# Shell script to compile and run frft_test_rt_timing_performance (macOS only)
#
# Usage: ./frft_test_rt_timing_performance.sh [options]
#
# Options:
#   --num-signals N      Number of random signals per test (default: 100)
#   --sample-rate SR     Sample rate in Hz (default: 44100)
#   --output FILE        Output filename (default: test_results/rt_timing_performance.txt)
#   --help               Show detailed help
#
# Examples:
#   ./frft_test_rt_timing_performance.sh                    # Run with defaults
#   ./frft_test_rt_timing_performance.sh --num-signals 50   # Run with 50 signals

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     FRFT Real-Time Performance Timing Test                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

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
echo ""

# Set directories (assumes script is run from eval/ directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
EVAL_DIR="$SCRIPT_DIR"

# Check if frft_engine.cpp exists
if [ ! -f "$SRC_DIR/frft_engine.cpp" ]; then
    echo "Error: frft_engine.cpp not found at $SRC_DIR/frft_engine.cpp"
    echo "Please make sure you're running this script from the eval/ directory"
    exit 1
fi

echo "🔨 Compiling frft_test_rt_timing_performance..."
echo ""

# Compile
clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I"$SRC_DIR" \
    -I"$FFTW3_INCLUDE" \
    "$SRC_DIR/frft_engine.cpp" \
    "$EVAL_DIR/frft_test_rt_timing_performance.cpp" \
    "$FFTW3_LIB" \
    -lm \
    -o "$EVAL_DIR/frft_test_rt_timing_performance"

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Running timing performance tests..."
    echo ""
    echo "This will test:"
    echo "  - Window sizes: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072"
    echo "  - Alpha values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0"
    echo "  - 100 random signals per window/alpha combination"
    echo "  - Total: 15,400 inference tests (14 window sizes × 11 alpha values × 100 signals)"
    echo ""
    echo "This may take a few minutes..."
    echo ""
    echo "================================================"

    # Pass all command-line arguments to the test program
    "$EVAL_DIR/frft_test_rt_timing_performance" "$@"

    # Check if execution was successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "Test completed successfully!"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        echo "Results are available in:"
        echo "  - test_results/rt_timing_performance.txt (summary statistics)"
        echo "  - test_results/rt_timing_performance_detailed.txt (all measurements)"
        echo ""
        echo "You can analyze the results using Python, Excel, or any data analysis tool."
        echo ""
    else
        echo ""
        echo "❌ Test execution failed!"
        exit 1
    fi
else
    echo ""
    echo "❌ Compilation failed!"
    exit 1
fi