#!/bin/bash
# Test FRFT composition: FRFT(α) ∘ FRFT(α) = FRFT(2α)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Testing FRFT Composition Property"
echo "════════════════════════════════════════════════════════════"
echo "Property: FRFT(α₂) ∘ FRFT(α₁) = FRFT(α₁ + α₂)"
echo ""

TEST_INPUT="2,1,-1,-2,-1,1"

# Test 1: FRFT(1) then extract output
echo "Step 1: Apply FRFT(α=1) to input [$TEST_INPUT]"
echo "────────────────────────────────────────────────────────────"
OUTPUT1=$($SCRIPT_DIR/frft.sh --alpha 1 --input $TEST_INPUT | grep "Real part:" -A 1 | tail -n 1)
echo "Output: $OUTPUT1"
echo ""

# Extract the real part values (this is a bit hacky but works)
REAL_VALUES=$(echo "$OUTPUT1" | sed 's/\[//g' | sed 's/\]//g' | tr -d ' ')

echo "Step 2: Apply FRFT(α=1) again to the output"
echo "────────────────────────────────────────────────────────────"
echo "Input for second FRFT: $REAL_VALUES"
OUTPUT2=$($SCRIPT_DIR/frft.sh --alpha 1 --input $REAL_VALUES 2>/dev/null | grep "Real part:" -A 1 | tail -n 1)
echo "Output after FRFT(1)∘FRFT(1): $OUTPUT2"
echo ""

echo "Step 3: Compare with direct FRFT(α=2)"
echo "────────────────────────────────────────────────────────────"
OUTPUT_DIRECT=$($SCRIPT_DIR/frft.sh --alpha 2 --input $TEST_INPUT | grep "Real part:" -A 1 | tail -n 1)
echo "Output from direct FRFT(2): $OUTPUT_DIRECT"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "Expected results:"
echo "- FRFT(1) ∘ FRFT(1) should equal FRFT(2)"
echo "- FRFT(2) is time-reversal (α=2 mod 4 = 2)"
echo "- Input [2,1,-1,-2,-1,1] reversed is [1,-1,-2,-1,1,2]"
echo "- With scaling: approximately [√6, -√6, -2√6, -√6, √6, 2√6]"
echo ""
echo "Note: Small numerical differences are expected due to"
echo "floating-point precision and the fact we're extracting only"
echo "the real part (discarding imaginary part which should be ~0)"