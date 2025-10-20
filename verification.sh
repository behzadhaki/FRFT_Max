#!/bin/bash

BUNDLE_PATH="$HOME/Documents/Max 9/Packages/GrooveTransformer/externals/GrooveTransformer.mxo"
BINARY="$BUNDLE_PATH/Contents/MacOS/GrooveTransformer"

echo "🔍 Verifying: $BINARY"
echo

# 1. Check if binary and dylibs exist
echo "📂 Bundle contents:"
ls -lh "$BUNDLE_PATH/Contents/MacOS"
echo

# 2. Check architecture
echo "🧬 Architecture:"
lipo -archs "$BINARY"
for dylib in "$BUNDLE_PATH"/Contents/MacOS/lib*.dylib; do
  echo "- $(basename "$dylib"): $(lipo -archs "$dylib")"
done
echo

# 3. Check linked libraries
echo "🔗 Linked libraries:"
otool -L "$BINARY"
echo

# 4. Check for libomp linkage
echo "🔍 libomp.dylib patched correctly?"
otool -L "$BUNDLE_PATH/Contents/MacOS/libtorch_cpu.dylib" | grep libomp
echo

# 5. Check code signature
echo "🔐 Code signing:"
codesign --verify --deep --strict --verbose=2 "$BUNDLE_PATH"
echo

# 6. Gatekeeper assessment
echo "🛡  Gatekeeper status:"
spctl --assess --type execute --verbose "$BUNDLE_PATH"
echo
