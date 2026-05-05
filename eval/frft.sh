#!/bin/bash
# Simple shell script to compile and run simple_frft (macOS/Linux)

# Find FFTW3 paths (Homebrew on macOS or standard Linux paths)
FFTW3_INCLUDE=""
FFTW3_LIB=""

if [ -d "/opt/homebrew/opt/fftw" ]; then
    # macOS Apple Silicon
    FFTW3_INCLUDE="/opt/homebrew/opt/fftw/include"
    FFTW3_LIB="/opt/homebrew/opt/fftw/lib/libfftw3.a"
elif [ -d "/opt/homebrew/include" ] && [ -f "/opt/homebrew/lib/libfftw3.a" ]; then
    FFTW3_INCLUDE="/opt/homebrew/include"
    FFTW3_LIB="/opt/homebrew/lib/libfftw3.a"
elif [ -d "/usr/local/opt/fftw" ]; then
    # macOS Intel
    FFTW3_INCLUDE="/usr/local/opt/fftw/include"
    FFTW3_LIB="/usr/local/opt/fftw/lib/libfftw3.a"
elif [ -d "/usr/local/include" ] && [ -f "/usr/local/lib/libfftw3.a" ]; then
    FFTW3_INCLUDE="/usr/local/include"
    FFTW3_LIB="/usr/local/lib/libfftw3.a"
elif [ -d "/usr/include" ] && [ -f "/usr/lib/libfftw3.a" ]; then
    # Linux
    FFTW3_INCLUDE="/usr/include"
    FFTW3_LIB="/usr/lib/libfftw3.a"
elif [ -d "/usr/include" ] && [ -f "/usr/lib/x86_64-linux-gnu/libfftw3.a" ]; then
    # Ubuntu/Debian
    FFTW3_INCLUDE="/usr/include"
    FFTW3_LIB="/usr/lib/x86_64-linux-gnu/libfftw3.a"
else
    echo "❌ Error: FFTW3 not found!"
    echo "Install with:"
    echo "  macOS: brew install fftw"
    echo "  Ubuntu/Debian: sudo apt-get install libfftw3-dev"
    exit 1
fi

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"

# Check if frft_engine.cpp exists
if [ ! -f "$SRC_DIR/frft_engine.cpp" ]; then
    echo "❌ Error: frft_engine.cpp not found at $SRC_DIR/frft_engine.cpp"
    echo "Please ensure the src directory is in the parent directory"
    exit 1
fi

# Check if simple_frft.cpp exists
if [ ! -f "$SCRIPT_DIR/simple_frft.cpp" ]; then
    echo "❌ Error: simple_frft.cpp not found at $SCRIPT_DIR/simple_frft.cpp"
    exit 1
fi

# Check if executable exists or needs recompilation
NEED_COMPILE=0
if [ ! -f "$SCRIPT_DIR/simple_frft" ]; then
    NEED_COMPILE=1
elif [ "$SCRIPT_DIR/simple_frft.cpp" -nt "$SCRIPT_DIR/simple_frft" ]; then
    NEED_COMPILE=1
elif [ "$SRC_DIR/frft_engine.cpp" -nt "$SCRIPT_DIR/simple_frft" ]; then
    NEED_COMPILE=1
fi

# Compile if needed
if [ $NEED_COMPILE -eq 1 ]; then
    echo "🔨 Compiling simple_frft..."

    clang++ -std=c++17 -O3 -march=native -ffast-math \
        -I"$SRC_DIR" \
        -I"$FFTW3_INCLUDE" \
        "$SRC_DIR/frft_engine.cpp" \
        "$SCRIPT_DIR/simple_frft.cpp" \
        "$FFTW3_LIB" \
        -lm \
        -o "$SCRIPT_DIR/simple_frft"

    if [ $? -ne 0 ]; then
        echo "❌ Compilation failed!"
        exit 1
    fi
    echo "✅ Compilation successful!"
fi

# Run the program with all arguments passed to this script
"$SCRIPT_DIR/simple_frft" "$@"