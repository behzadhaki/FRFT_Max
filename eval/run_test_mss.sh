#!/bin/bash
# Shell script to compile and run generate_homomorphism_mss_wavfiles (macOS only)

set -e  # Exit on error

# Parse command line arguments
USE_WINDOWING=0
if [ "$1" == "--windowed" ] || [ "$1" == "-w" ]; then
    USE_WINDOWING=1
    echo "🪟 Windowing mode enabled"
fi

echo "🔨 Compiling generate_homomorphism_mss_wavfiles..."

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

# Check if source files exist
if [ ! -f "$SRC_DIR/frft_engine.cpp" ]; then
    echo "❌ Error: frft_engine.cpp not found at $SRC_DIR/frft_engine.cpp"
    exit 1
fi

if [ ! -f "$EVAL_DIR/generate_homomorphism_mss_wavfiles.cpp" ]; then
    echo "❌ Error: generate_homomorphism_mss_wavfiles.cpp not found at $EVAL_DIR/generate_homomorphism_mss_wavfiles.cpp"
    exit 1
fi

# Compile
clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I"$SRC_DIR" \
    -I"$FFTW3_INCLUDE" \
    "$SRC_DIR/frft_engine.cpp" \
    "$EVAL_DIR/generate_homomorphism_mss_wavfiles.cpp" \
    "$FFTW3_LIB" \
    -lm \
    -o "$EVAL_DIR/generate_homomorphism_mss_wavfiles"

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Running WAV file generator..."
    echo "================================================"
    if [ $USE_WINDOWING -eq 1 ]; then
        echo "Mode: WINDOWED (512, 1024, 2048, 4096 samples with 4x overlap)"
        echo "This will generate WAV files with overlap-add processing"
    else
        echo "Mode: DIRECT (full 1-second transform)"
        echo "This will generate ~26,000 WAV files (13 frequencies × 1000 samples × 2 + 13 sources)"
    fi
    echo "This may take several minutes..."
    echo "================================================"
    echo ""

    if [ $USE_WINDOWING -eq 1 ]; then
        "$EVAL_DIR/generate_homomorphism_mss_wavfiles" --windowed
    else
        "$EVAL_DIR/generate_homomorphism_mss_wavfiles"
    fi

    if [ $? -eq 0 ]; then
        echo ""
        echo "================================================"
        echo "✅ WAV file generation complete!"
        echo ""
        echo "Next steps:"
        echo "1. Run the Python analysis:"
        echo "   python3 analyze_homomorphism_mss.py"
        echo ""
        echo "Output location: ./homomorphism_mss_sources/"
        echo "================================================"
    else
        echo "❌ WAV file generation failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    exit 1
fi