#!/bin/bash

set -e

# Variables
TORCH_VERSION="2.6.0"  # Change this to your required version

# Set root installation directory based on the platform
if [[ "$(uname -s)" == "Darwin" ]]; then
  ROOT_INSTALL_DIR="/Library/Application Support/libtorch"  # macOS
elif [[ "$(uname -s)" == "Linux" ]]; then
  ROOT_INSTALL_DIR="/opt/libtorch"  # Linux
else
  echo "Unsupported platform. Exiting."
  exit 1
fi

DEBUG_DIR="$ROOT_INSTALL_DIR/libtorch-${TORCH_VERSION}-Debug"
RELEASE_DIR="$ROOT_INSTALL_DIR/libtorch-${TORCH_VERSION}-Release"

# Determine platform-specific file name
if [[ "$(uname -s)" == "Darwin" ]]; then
  PLATFORM="MacOS"
  if [[ $(uname -m) == "arm64" ]]; then
    DEBUG_FILE="libtorch-macos-arm64-${TORCH_VERSION}.zip"
    RELEASE_FILE="libtorch-macos-arm64-${TORCH_VERSION}.zip"
  else
    DEBUG_FILE="libtorch-macos-x86_64-${TORCH_VERSION}.zip"
    RELEASE_FILE="libtorch-macos-x86_64-${TORCH_VERSION}.zip"
  fi
elif [[ "$(uname -s)" == "Linux" ]]; then
  PLATFORM="Linux"
  DEBUG_FILE="libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip"
  RELEASE_FILE="libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip"
else
  echo "Unsupported platform. Exiting."
  exit 1
fi

# Function to download and install libtorch
install_libtorch() {
  local build_type=$1
  local target_dir=$2
  local file_name=$3

  echo "Installing libtorch ($build_type) to $target_dir"

  # Remove existing directory
  if [[ -d "$target_dir" ]]; then
    echo "Removing existing directory: $target_dir"
    rm -rf "$target_dir"
  fi

  mkdir -p "$target_dir"

  # Download the file
  local download_url="https://download.pytorch.org/libtorch/cpu/$file_name"
  local temp_file="/tmp/$file_name"

  echo "Downloading $file_name from $download_url"
  curl -L -o "$temp_file" "$download_url"

  # Extract the archive
  echo "Extracting $file_name"
  unzip -q "$temp_file" -d "$target_dir"
  rm "$temp_file"

  echo "Libtorch ($build_type) installed successfully to $target_dir"
}

# Install Debug and Release versions
install_libtorch "Debug" "$DEBUG_DIR" "$DEBUG_FILE"
install_libtorch "Release" "$RELEASE_DIR" "$RELEASE_FILE"

# Move .so files to /usr/lib on Linux
if [[ "$(uname -s)" == "Linux" ]]; then
  echo "Moving .so files to /usr/lib..."
  find "$DEBUG_DIR" "$RELEASE_DIR" -type f -name "*.so*" -exec sudo cp {} /usr/lib/ \;
  echo ".so files have been moved to /usr/lib"
fi

# in MacOs, move .dylib files to /usr/local/lib
#if [[ "$(uname -s)" == "Darwin" ]]; then
#  echo "Moving .dylib files to /usr/local/lib..."
#  find "$DEBUG_DIR" "$RELEASE_DIR" -type f -name "*.dylib" -exec sudo cp {} /usr/local/lib/ \;
#  echo ".dylib files have been moved to /usr/local/lib"
#fi

# Output result
echo "Libtorch installations completed:"
echo "  Debug: $DEBUG_DIR"
echo "  Release: $RELEASE_DIR"
