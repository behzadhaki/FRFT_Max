#!/bin/bash
set -e

DEST_DIR="lib/max-sdk"

if [ -d "$DEST_DIR" ]; then
    echo "✅ Max SDK already exists at $DEST_DIR"
else
    echo "⬇️ Cloning Max SDK with submodules..."
    git clone --recurse-submodules https://github.com/Cycling74/max-sdk.git "$DEST_DIR"
    echo "✅ Max SDK cloned successfully"
fi
