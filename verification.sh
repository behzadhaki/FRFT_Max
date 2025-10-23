#!/bin/bash
set -e

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
PACKAGE_DIR="$HOME/Documents/Max 9/Packages/FRFT/externals"

# If a specific bundle is provided, verify only that one
# Otherwise, verify all .mxo bundles in the directory
if [ -n "$1" ]; then
  BUNDLES=("$PACKAGE_DIR/$1")
else
  # Find all .mxo bundles (compatible with older bash versions)
  BUNDLES=()
  while IFS= read -r -d '' bundle; do
    BUNDLES+=("$bundle")
  done < <(find "$PACKAGE_DIR" -maxdepth 1 -name "*.mxo" -type d -print0)
fi

# Check if any bundles were found
if [ ${#BUNDLES[@]} -eq 0 ]; then
  echo "❌ No .mxo bundles found in: $PACKAGE_DIR"
  exit 1
fi

echo "🔍 Found ${#BUNDLES[@]} bundle(s) to verify"
echo

# ------------------------------------------------------------------------------
# Process each bundle
# ------------------------------------------------------------------------------
for BUNDLE_PATH in "${BUNDLES[@]}"; do
  BUNDLE_NAME=$(basename "$BUNDLE_PATH")
  BINARY="$BUNDLE_PATH/Contents/MacOS/${BUNDLE_NAME%.mxo}"

  echo "════════════════════════════════════════════════════════════════════════"
  echo "🧩 Verifying: $BUNDLE_NAME"
  echo "════════════════════════════════════════════════════════════════════════"
  echo

  # ------------------------------------------------------------------------------
  # 1. Check if bundle exists
  # ------------------------------------------------------------------------------
  if [ ! -d "$BUNDLE_PATH" ]; then
    echo "❌ Bundle not found: $BUNDLE_PATH"
    continue
  fi

  # ------------------------------------------------------------------------------
  # 2. Check and remove quarantine attribute
  # ------------------------------------------------------------------------------
  echo "🔒 Quarantine attribute check:"
  if xattr "$BUNDLE_PATH" 2>/dev/null | grep -q "com.apple.quarantine"; then
    echo "⚠️  Found com.apple.quarantine attribute - removing..."
    xattr -d -r com.apple.quarantine "$BUNDLE_PATH" 2>/dev/null || true
    echo "✅ Quarantine attribute removed"
  else
    echo "✅ No quarantine attribute found"
  fi
  echo

  # ------------------------------------------------------------------------------
  # 3. Show bundle contents
  # ------------------------------------------------------------------------------
  echo "📂 Bundle contents:"
  ls -lh "$BUNDLE_PATH/Contents/MacOS"
  echo

  # ------------------------------------------------------------------------------
  # 4. Sign all dylibs and binary (deep code sign)
  # ------------------------------------------------------------------------------
  echo "🔏 Signing binary and dylibs..."
  for file in "$BUNDLE_PATH"/Contents/MacOS/*; do
    if [[ -f "$file" ]]; then
      echo "  • Signing $(basename "$file")"
      codesign --force --sign - --timestamp=none --deep "$file" 2>/dev/null || true
    fi
  done
  echo "✅ Signing complete."
  echo

  # ------------------------------------------------------------------------------
  # 5. Check architecture
  # ------------------------------------------------------------------------------
  echo "🧬 Architectures:"
  if [ -f "$BINARY" ]; then
    echo "$(basename "$BINARY"): $(lipo -archs "$BINARY")"
  fi
  for dylib in "$BUNDLE_PATH"/Contents/MacOS/lib*.dylib; do
    [ -f "$dylib" ] && echo "$(basename "$dylib"): $(lipo -archs "$dylib")"
  done
  echo

  # ------------------------------------------------------------------------------
  # 6. Linked libraries
  # ------------------------------------------------------------------------------
  echo "🔗 Linked libraries for main binary:"
  otool -L "$BINARY"
  echo

  # ------------------------------------------------------------------------------
  # 7. Check for libomp linkage
  # ------------------------------------------------------------------------------
  echo "🔍 libomp.dylib patched correctly?"
  otool -L "$BUNDLE_PATH/Contents/MacOS/libtorch_cpu.dylib" | grep libomp || echo "⚠️ libomp not found"
  echo

  # ------------------------------------------------------------------------------
  # 8. Verify code signature
  # ------------------------------------------------------------------------------
  echo "🔐 Verifying code signature..."
  codesign --verify --deep --strict --verbose=2 "$BUNDLE_PATH"
  echo "✅ Code signature verification OK."
  echo

  # ------------------------------------------------------------------------------
  # 9. Gatekeeper assessment
  # ------------------------------------------------------------------------------
  # Gatekeeper check (optional for distribution)
  if [ "${CHECK_GATEKEEPER:-0}" -eq 1 ]; then
      spctl --assess --type execute --verbose "$BUNDLE_PATH"
  else
      echo "🛡️  Gatekeeper check skipped (local ad-hoc signing)."
  fi

  echo
done

echo "════════════════════════════════════════════════════════════════════════"
echo "✅ Verification complete for ${#BUNDLES[@]} bundle(s)"
echo "════════════════════════════════════════════════════════════════════════"