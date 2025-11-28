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
    echo "Installation options:"
    echo "1) No installation (use locally only)"
    echo "2) Install for current user (accessible in all environments)"
    echo "3) Install system-wide (requires sudo)"
    echo "4) Development mode (changes reflected immediately)"
    read -p "Choose installation option (1-4): " INSTALL_OPTION

    case $INSTALL_OPTION in
        1)
            echo "Skipping installation. Module available in current directory."
            ;;
        2)
            echo "Installing for current user..."
            $PIP_CMD install --user .
            echo "✓ Installed! Package 'frft_cpp' is now accessible from any environment."
            ;;
        3)
            echo "Installing system-wide..."
            sudo $PIP_CMD install .
            echo "✓ Installed system-wide!"
            ;;
        4)
            echo "Installing in development mode..."
            $PIP_CMD install --user -e .
            echo "✓ Installed in development mode!"
            echo "Note: Changes to the code will require rebuilding with 'python3 setup.py build_ext --inplace'"
            ;;
        *)
            echo "Invalid choice. Skipping installation."
            ;;
    esac
    
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