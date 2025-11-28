from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import setuptools
import pybind11

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        return pybind11.get_include()

def get_fftw_paths():
    """
    Find FFTW3 installation paths across different systems.
    Returns (include_dirs, library_dirs, libraries)
    """
    include_dirs = []
    library_dirs = []
    libraries = ['fftw3']

    # Try Homebrew on macOS (both Intel and Apple Silicon)
    homebrew_prefixes = [
        '/opt/homebrew',  # Apple Silicon
        '/usr/local',      # Intel Mac
    ]

    for prefix in homebrew_prefixes:
        inc_path = os.path.join(prefix, 'include')
        lib_path = os.path.join(prefix, 'lib')
        if os.path.exists(inc_path) and os.path.exists(lib_path):
            fftw_header = os.path.join(inc_path, 'fftw3.h')
            if os.path.exists(fftw_header):
                include_dirs.append(inc_path)
                library_dirs.append(lib_path)
                print(f"Found FFTW3 via Homebrew at: {prefix}")
                return include_dirs, library_dirs, libraries

    # Try to use brew --prefix (if brew is available)
    try:
        result = subprocess.run(['brew', '--prefix', 'fftw'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            prefix = result.stdout.strip()
            inc_path = os.path.join(prefix, 'include')
            lib_path = os.path.join(prefix, 'lib')
            if os.path.exists(inc_path):
                include_dirs.append(inc_path)
                library_dirs.append(lib_path)
                print(f"Found FFTW3 via 'brew --prefix fftw': {prefix}")
                return include_dirs, library_dirs, libraries
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try pkg-config
    try:
        result = subprocess.run(['pkg-config', '--cflags-only-I', 'fftw3'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            cflags = result.stdout.strip()
            if cflags:
                include_dirs = [flag[2:] for flag in cflags.split() if flag.startswith('-I')]

        result = subprocess.run(['pkg-config', '--libs-only-L', 'fftw3'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            ldflags = result.stdout.strip()
            if ldflags:
                library_dirs = [flag[2:] for flag in ldflags.split() if flag.startswith('-L')]

        if include_dirs or library_dirs:
            print(f"Found FFTW3 via pkg-config")
            return include_dirs, library_dirs, libraries
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Standard Linux paths
    standard_paths = [
        '/usr/include',
        '/usr/local/include',
        '/usr/include/x86_64-linux-gnu',
    ]

    for path in standard_paths:
        fftw_header = os.path.join(path, 'fftw3.h')
        if os.path.exists(fftw_header):
            print(f"Found FFTW3 at standard location: {path}")
            # Don't need to add standard paths explicitly on Linux
            return [], [], libraries

    # Check environment variables
    if 'FFTW_ROOT' in os.environ:
        prefix = os.environ['FFTW_ROOT']
        inc_path = os.path.join(prefix, 'include')
        lib_path = os.path.join(prefix, 'lib')
        if os.path.exists(inc_path):
            include_dirs.append(inc_path)
            library_dirs.append(lib_path)
            print(f"Found FFTW3 via FFTW_ROOT environment variable: {prefix}")
            return include_dirs, library_dirs, libraries

    print("WARNING: Could not automatically locate FFTW3.")
    print("If build fails, set FFTW_ROOT environment variable:")
    print("  export FFTW_ROOT=/path/to/fftw")
    print("Or install via:")
    print("  macOS: brew install fftw")
    print("  Ubuntu/Debian: sudo apt-get install libfftw3-dev")

    return include_dirs, library_dirs, libraries

# Get FFTW paths
fftw_include_dirs, fftw_library_dirs, fftw_libraries = get_fftw_paths()

# Combine all include directories
all_include_dirs = [
                       get_pybind_include(),
                       pybind11.get_include(),
                       '.',  # Current directory for frft_engine.h
                   ] + fftw_include_dirs

# Combine all library directories
all_library_dirs = fftw_library_dirs

ext_modules = [
    Extension(
        'frft_cpp',
        sources=[
            'frft_bindings.cpp',
            'frft_engine.cpp',
        ],
        include_dirs=all_include_dirs,
        library_dirs=all_library_dirs,
        libraries=fftw_libraries,
        language='c++',
        extra_compile_args=['-std=c++14', '-O3'],
    ),
]

setup(
    name='frft_cpp',
    version='0.1.0',
    author='Your Name',
    description='Fast Fractional Fourier Transform implementation',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0', 'numpy'],
    setup_requires=['pybind11>=2.6.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.6',
)