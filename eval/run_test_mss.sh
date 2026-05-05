#!/bin/bash
# Shell script to compile and run generate_homomorphism_mss_wavfiles (macOS only)

set -e  # Exit on error

# Default values
DURATION=""
WINSIZES=""
FREQS=""
ALPHA_MIN=""
ALPHA_MAX=""
ALPHA_STEP=""

# Parse command line arguments
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dur <seconds>              Duration in seconds (default: 1.0)"
    echo "  --winsizes <sizes>           Comma-separated window sizes or 'single' for direct mode"
    echo "                               Examples: '512,1024,2048,4096' or 'single'"
    echo "                               Default: 512,1024,2048,4096 (windowed)"
    echo "  --freqs <frequencies>        Comma-separated frequencies in Hz"
    echo "                               Default: 100,200,300,440,500,1000,1500,2000,3000,4000,6000,8000,10000"
    echo "  --alpha-min <value>          Minimum alpha value for grid (default: -2.0)"
    echo "  --alpha-max <value>          Maximum alpha value for grid (default: 2.0)"
    echo "  --alpha-step <value>         Step size for alpha grid (default: 0.1)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  Windowed mode (0.5 seconds, custom window sizes and frequencies):"
    echo "    $0 --dur 0.5 --winsizes 512,1024,2048,4096 --freqs 100,440,1000,2000,4000,8000"
    echo ""
    echo "  Direct mode (non-windowed, 0.5 seconds):"
    echo "    $0 --dur 0.5 --winsizes single --freqs 100,440,1000,2000,4000,8000"
    echo ""
    echo "  Custom alpha grid (coarser grid for faster generation):"
    echo "    $0 --dur 0.5 --winsizes single --freqs 440,1000 --alpha-min -1 --alpha-max 1 --alpha-step 0.5"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --dur)
            DURATION="$2"
            shift 2
            ;;
        --winsizes)
            WINSIZES="$2"
            shift 2
            ;;
        --freqs)
            FREQS="$2"
            shift 2
            ;;
        --alpha-min)
            ALPHA_MIN="$2"
            shift 2
            ;;
        --alpha-max)
            ALPHA_MAX="$2"
            shift 2
            ;;
        --alpha-step)
            ALPHA_STEP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

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

    # Build command arguments
    CMD_ARGS=""

    if [ -n "$DURATION" ]; then
        CMD_ARGS="$CMD_ARGS --dur $DURATION"
        echo "Duration: $DURATION seconds"
    else
        echo "Duration: 1.0 seconds (default)"
    fi

    if [ -n "$WINSIZES" ]; then
        CMD_ARGS="$CMD_ARGS --winsizes $WINSIZES"
        if [ "$WINSIZES" == "single" ]; then
            echo "Mode: DIRECT (full transform)"
        else
            echo "Mode: WINDOWED (overlap-add)"
            echo "Window sizes: $WINSIZES"
        fi
    else
        echo "Mode: WINDOWED (overlap-add) - default"
        echo "Window sizes: 512,1024,2048,4096 (default)"
    fi

    if [ -n "$FREQS" ]; then
        CMD_ARGS="$CMD_ARGS --freqs $FREQS"
        echo "Frequencies: $FREQS Hz"
    else
        echo "Frequencies: 100,200,300,440,500,1000,1500,2000,3000,4000,6000,8000,10000 Hz (default)"
    fi

    if [ -n "$ALPHA_MIN" ]; then
        CMD_ARGS="$CMD_ARGS --alpha-min $ALPHA_MIN"
        echo "Alpha min: $ALPHA_MIN"
    else
        echo "Alpha min: -2.0 (default)"
    fi

    if [ -n "$ALPHA_MAX" ]; then
        CMD_ARGS="$CMD_ARGS --alpha-max $ALPHA_MAX"
        echo "Alpha max: $ALPHA_MAX"
    else
        echo "Alpha max: 2.0 (default)"
    fi

    if [ -n "$ALPHA_STEP" ]; then
        CMD_ARGS="$CMD_ARGS --alpha-step $ALPHA_STEP"
        echo "Alpha step: $ALPHA_STEP"
    else
        echo "Alpha step: 0.1 (default)"
    fi

    echo ""
    echo "This may take several minutes..."
    echo "================================================"
    echo ""

    # Run the generator with arguments
    if [ -n "$CMD_ARGS" ]; then
        "$EVAL_DIR/generate_homomorphism_mss_wavfiles" $CMD_ARGS
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
        echo "Output location: ./test_results/homomorphism_mss_grid_windowed/"
        echo "            or: ./test_results/homomorphism_mss_grid_direct/"
        echo "================================================"
    else
        echo "❌ WAV file generation failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    exit 1
fi