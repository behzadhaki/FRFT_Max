#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_frft_window.sh
#
# Creates a 256-sample rectangular window (128 zeros, 128 ones),
# runs simple_frft with alpha=2, then writes three WAV files:
#   original_signal.wav   – the raw rectangular window
#   original_reversed.wav – the rectangular window, time-reversed
#   frft_output.wav       – the FRFT output (magnitude), alpha=2
#
# Dependencies: simple_frft binary (same dir or on PATH), python3
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRFT_BIN="${SCRIPT_DIR}/simple_frft"   # adjust path if needed
SAMPLE_RATE=8000                        # Hz – change as desired
N=256

# ── 1. Locate binary ─────────────────────────────────────────
if [[ ! -x "$FRFT_BIN" ]]; then
    FRFT_BIN="$(command -v simple_frft 2>/dev/null || true)"
    if [[ -z "$FRFT_BIN" ]]; then
        echo "ERROR: simple_frft binary not found. Put it next to this script or on PATH." >&2
        exit 1
    fi
fi
echo "Using binary: $FRFT_BIN"

# ── 2. Build input string: 128 zeros then 128 ones ───────────
INPUT=""
for ((i=0; i<128; i++)); do
    INPUT="${INPUT}0,"
done
for ((i=0; i<128; i++)); do
    if ((i < 127)); then
        INPUT="${INPUT}1,"
    else
        INPUT="${INPUT}1"
    fi
done

echo "Input: 128 zeros followed by 128 ones (N=${N})"

# ── 3. Run FRFT with alpha=2 ──────────────────────────────────
FRFT_OUTPUT=$("$FRFT_BIN" --alpha 2.0 --input "$INPUT")
echo "FRFT complete."

# ── 4. Parse output and write WAV files with Python ──────────
python3 - <<PYEOF
import re, struct, sys, math

frft_output = """${FRFT_OUTPUT}"""

def parse_array(label, text):
    pattern = rf'{re.escape(label)}.*?\[([^\]]+)\]'
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        sys.exit(f"ERROR: Could not parse section '{label}' from FRFT output.")
    return [float(x.strip()) for x in m.group(1).split(',')]

real_out = parse_array("Real part:", frft_output)
imag_out = parse_array("Imaginary part:", frft_output)

N = ${N}
SR = ${SAMPLE_RATE}

assert len(real_out) == N, f"Expected {N} real values, got {len(real_out)}"
assert len(imag_out) == N, f"Expected {N} imag values, got {len(imag_out)}"

def write_wav(filename, samples, sample_rate):
    """Write a list of floats as a normalised 16-bit mono PCM WAV."""
    max_amp = max(abs(s) for s in samples) or 1.0
    pcm = [int(s / max_amp * 32767) for s in samples]

    data_size  = len(pcm) * 2   # 2 bytes per sample
    chunk_size = 36 + data_size

    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', chunk_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))            # PCM
        f.write(struct.pack('<H', 1))            # mono
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 2))  # byte rate
        f.write(struct.pack('<H', 2))            # block align
        f.write(struct.pack('<H', 16))           # bits per sample
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        for s in pcm:
            f.write(struct.pack('<h', s))

    print(f"  Wrote {filename}  ({len(pcm)} samples @ {sample_rate} Hz)")

# ── original rectangular window ──────────────────────────────
original = [0.0] * 128 + [1.0] * 128
write_wav("original_signal.wav", original, SR)

# ── original time-reversed ───────────────────────────────────
write_wav("original_reversed.wav", list(reversed(original)), SR)

# ── FRFT output (magnitude) ───────────────────────────────────
magnitude = [math.sqrt(r**2 + i**2) for r, i in zip(real_out, imag_out)]
write_wav("frft_output.wav", magnitude, SR)

print()
print("Done.")
print("  original_signal.wav   – rectangular window (128 zeros, 128 ones)")
print("  original_reversed.wav – rectangular window, time-reversed")
print("  frft_output.wav       – |FRFT(alpha=2)| of the window")
PYEOF

echo ""
echo "WAV files written to the current directory."