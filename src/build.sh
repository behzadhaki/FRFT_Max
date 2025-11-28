#!/bin/bash

# Build script for FRFT C++ Python bindings

set -e  # Exit on error

echo "========================================"
echo "FRFT C++ Python Bindings - Build Script"
echo "========================================"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.6+."
    exit 1
fi

# Check for pip
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "Error: pip not found. Please install pip."
    exit 1
fi

# Determine pip command
PIP_CMD="pip3"
if command -v pip &> /dev/null; then
    PIP_CMD="pip"
fi

# Check for FFTW3
echo ""
echo "Checking for FFTW3..."
if pkg-config --exists fftw3 2>/dev/null; then
    echo "✓ FFTW3 found"
elif [ -f "/usr/lib/libfftw3.so" ] || [ -f "/usr/local/lib/libfftw3.so" ] || [ -f "/opt/homebrew/lib/libfftw3.dylib" ]; then
    echo "✓ FFTW3 library found"
else
    echo "Warning: FFTW3 may not be installed."
    echo "On Ubuntu/Debian: sudo apt-get install libfftw3-dev"
    echo "On macOS: brew install fftw"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
$PIP_CMD install -r requirements.txt

# Build options
echo ""
echo "Build method:"
echo "1) setup.py (recommended for most users)"
echo "2) CMake (recommended for C++ developers)"
read -p "Choose build method (1 or 2): " BUILD_METHOD

if [ "$BUILD_METHOD" = "1" ]; then
    echo ""
    echo "Building with setup.py..."
    python3 setup.py build_ext --inplace
    
    echo ""
    echo "✓ Build complete!"
    echo ""
    echo "To install system-wide, run:"
    echo "  pip install ."
    echo ""
    echo "Or for development mode:"
    echo "  pip install -e ."
    
elif [ "$BUILD_METHOD" = "2" ]; then
    # Check for CMake
    if ! command -v cmake &> /dev/null; then
        echo "Error: cmake not found. Please install CMake 3.12+."
        exit 1
    fi
    
    echo ""
    echo "Building with CMake..."
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    cd ..
    
    echo ""
    echo "✓ Build complete!"
    echo ""
    echo "The module is in: build/frft_cpp.so"
    echo ""
    echo "To use it, either:"
    echo "  1. Copy build/frft_cpp.so to your project directory"
    echo "  2. Add to PYTHONPATH: export PYTHONPATH=\$PYTHONPATH:$(pwd)/build"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Run quick test
echo ""
read -p "Run quick test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running quick test..."
    
    # Add build directory to PYTHONPATH if using CMake
    if [ "$BUILD_METHOD" = "2" ]; then
        export PYTHONPATH=$PYTHONPATH:$(pwd)/build
    fi
    
    python3 benchmark_frft.py quick
fi

echo ""
echo "========================================"
echo "Build process complete!"
echo "========================================"