# FRFT Web / WASM

Builds the FRFT engine for browser use (WebAssembly + Web Audio API).

## Directory layout

```
web/
├── wasm/
│   ├── frft_engine_wasm.h      KissFFT-based engine (drop-in for WASM builds)
│   ├── frft_engine_wasm.cpp    Implementation — algorithm identical to src/
│   ├── test_engine.cpp         Cross-comparison test: KissFFT vs FFTW
│   └── Makefile                Fetches KissFFT, builds & runs the test
└── demos/                      Static HTML demo pages (Stage 5, coming soon)
```

---

## Prerequisites

| Tool | Install |
|------|---------|
| C++17 compiler (`g++` / `clang++`) | ships with Xcode Command Line Tools |
| FFTW3 | `brew install fftw` |
| `curl` | ships with macOS |
| Emscripten (`emcc`) | needed for Stage 3 onwards — not required for the test |

Check you have what you need for the test:

```bash
g++ --version
brew list fftw
```

---

## Stage 1 — Run the cross-comparison test

The test compiles both engines side-by-side and verifies they produce identical
output. KissFFT source files are fetched automatically on first run.

```bash
cd web/wasm
make test
```

First run downloads three KissFFT headers/sources into `web/wasm/kissfft/` and
then compiles. Subsequent runs skip the download step.

Expected output (all lines should show `PASS`):

```
FRFT Engine Test: KissFFT (wasm) vs FFTW (original)
============================================================

[1] Cross-comparison (FFTW vs KissFFT)
  PASS  N=64  alpha=0.250000   max_err = 8.88e-16
  ...

[2] Round-trip consistency (FFTW error vs KissFFT error must match)
  PASS  N=128  alpha=0.250000  (FFTW=0.9619 KissFFT=0.9619)  max_err = 1.11e-16
  ...

All tests passed.
```

> **Note on Tests 2–4:** The round-trip and additivity errors shown (e.g. ~1.0)
> are a known property of this discrete FRFT algorithm — they appear identically
> in both the FFTW and KissFFT engines. Test 1 (cross-comparison at ~1e-15) is
> the definitive check that the KissFFT port is correct.

### Other Makefile targets

```bash
make          # build only, don't run
make test     # build and run
make clean    # remove build artefacts and the kissfft/ folder
```

---

## Stage 2 — Emscripten bindings

`web/wasm/frft_bindings.cpp` wraps `FRFTEngine` in an `FRFTProcessor` class
and exposes it to JavaScript via `emscripten::embind`.

Two usage patterns are available from JS:

**Pattern A — zero-copy (AudioWorklet)**
Write audio data directly to/from WASM heap buffers using typed array views:

```js
proc.prepare(512);

// Map JS views onto internal WASM buffers (no copy)
const inR  = new Float64Array(Module.HEAPF64.buffer, proc.inputRealPtr(),  512);
const inI  = new Float64Array(Module.HEAPF64.buffer, proc.inputImagPtr(),  512);
const outR = new Float64Array(Module.HEAPF64.buffer, proc.outputRealPtr(), 512);
const outI = new Float64Array(Module.HEAPF64.buffer, proc.outputImagPtr(), 512);

inR.set(audioSamples);      // write input
proc.process(512, alpha);   // run FRFT
// outR / outI now hold results
```

**Pattern B — convenience (offline / scripting)**
Pass Float64Arrays in, get a plain JS object back:

```js
const result = proc.processArrays(realIn, imagIn, alpha);
// result.real → Float64Array
// result.imag → Float64Array
```

## Stage 3 — WASM build

### Install Emscripten (one time)

```bash
git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
cd ~/emsdk
./emsdk install latest
./emsdk activate latest
source ~/emsdk/emsdk_env.sh   # adds emcc to PATH for this shell session
```

Add the `source` line to your `~/.zshrc` if you want `emcc` available by default.

### Build

```bash
cd web/wasm
make test          # fetch KissFFT and run native test first (if not done yet)
make wasm          # release build → dist/frft.js + dist/frft.wasm
make wasm_debug    # debug build   → dist/debug/frft.js + dist/debug/frft.wasm
```

Or call the script directly:

```bash
./build_wasm.sh           # release
./build_wasm.sh debug     # debug (O0, assertions on)
```

### Output

| File | Description |
|------|-------------|
| `dist/frft.js` | JS glue — loads and wraps the WASM module |
| `dist/frft.wasm` | Compiled engine |

### Use in JS

```js
import FRFTModule from './frft.js';

const Module = await FRFTModule();
const proc = new Module.FRFTProcessor();

proc.prepare(512);

// Pattern A — zero-copy (AudioWorklet)
const inR  = new Float64Array(Module.HEAPF64.buffer, proc.inputRealPtr(),  512);
const outR = new Float64Array(Module.HEAPF64.buffer, proc.outputRealPtr(), 512);
inR.set(samples);
proc.process(512, 0.5);
// outR holds the result

// Pattern B — convenience
const result = proc.processArrays(realIn, imagIn, 0.5);
// result.real, result.imag → Float64Array
```

## Stage 4 — AudioWorklet

Two files in `web/audio/`:

| File | Role |
|------|------|
| `frft-processor.js` | Runs in the audio thread — loads WASM, accumulates samples, calls `FRFTProcessor` |
| `frft-node.js` | Main-thread helper — registers the worklet, waits for WASM ready, exposes a clean API |

### How it works

Web Audio delivers **128 samples per render quantum**. The processor accumulates
these into a larger block (`bufSize`, default 512), applies the FRFT, then drains
the result 128 samples at a time. This adds one block of latency (~11 ms at 44.1 kHz
with `bufSize=512`).

The WASM heap pointers from Stage 2 (Pattern A) are used throughout — audio data
is written directly into WASM memory with no intermediate copies.

### Wiring it up

```js
import { createFRFTNode } from './audio/frft-node.js';

const ctx  = new AudioContext();
const node = await createFRFTNode(ctx, {
    processorUrl: '/audio/frft-processor.js',
    frftJsUrl:    '/wasm/dist/frft.js',
    alpha:        0.5,
    bufSize:      512,
});

sourceNode.connect(node);
node.connect(ctx.destination);

// Change parameters at any time
node.setAlpha(0.75);
node.setBufSize(1024);
```

### Alpha guide

| Alpha | Effect |
|-------|--------|
| `0` | Identity (passthrough) |
| `0.5` | Halfway between time and frequency domain |
| `1` | Standard FFT (frequency domain) |
| `2` | Time-reversal |

## Stage 5 — Demo page

`web/demos/index.html` — single-page interactive demo.

### Features

- **Three audio sources**: sine wave (440 Hz), frequency sweep (100–4000 Hz), or file upload
- **α slider** (−2 to +2) with live parameter changes while audio is running
- **Quick-set buttons**: −1 (Inv. FFT), 0 (passthrough), 0.5 (mid), 1 (FFT), 2 (reversal)
- **Block size selector**: 128 / 256 / 512 / 1024 / 2048 samples
- **Dual oscilloscope**: input (dry) and output (FRFT) waveforms side by side

### Run locally

```bash
cd web
python3 -m http.server 8000
# open http://localhost:8000/demos/
```

Browsers block AudioWorklets on `file://` — a local server is required.

---

## Stage 6 — Embed widgets

Two self-contained iframe-friendly pages in `web/demos/`:

| File | Purpose |
|------|---------|
| `embed_interactive_frft.html` | Full controls — source, FRFT params, playback, download |
| `embed_non_interactive_frft.html` | Display-only — params via URL, auto-processes on load |
| `test-embed.html` | Local harness for testing both embeds at arbitrary sizes |

Both pages are transparent-background, dark-themed, and resize to whatever `width`/`height` the parent `<iframe>` sets.

---

### Interactive embed

All FRFT parameters are exposed as in-page controls. The user can change anything and the result reprocesses automatically.

**Embed:**
```html
<iframe src="embed_interactive_frft.html?w=600&h=260"
        width="600" height="260" frameborder="0"></iframe>
```

**URL params:**

| Param | Default | Description |
|-------|---------|-------------|
| `w` | — | Frame width in px (also sets `<html>` width) |
| `h` | — | Frame height in px |
| `dur` | `3` | Initial duration in seconds for generated signals |

Everything else (source type, frequency, α, block size, overlap, mode) is controlled interactively inside the embed.

**Features:**
- Source: sine wave, frequency sweep, or audio file upload
- FRFT mode: full-spectrum or half-spectrum
- α, block size (16 → 131072 or "all"), overlap factor
- Source / FRFT spectrogram tabs with drag-to-zoom
- Play / stop with live spectrum strip
- Download ZIP (source WAV + FRFT WAV + both spectrograms as PNG)

---

### Non-interactive embed

All parameters come from the URL. The embed auto-processes on load and shows the result — no user controls.

**Embed:**
```html
<iframe src="embed_non_interactive_frft.html?type=sweep&dur=10&alpha=0.1&blocksize=16384&overlap=4"
        width="500" height="200" frameborder="0"></iframe>
```

**URL params:**

| Param | Default | Description |
|-------|---------|-------------|
| `w` | — | Frame width in px |
| `h` | — | Frame height in px |
| `type` | `sweep` | Signal source: `sweep`, `sine`, or `file` |
| `dur` | `10` (sweep) / `3` (sine) | Duration in seconds |
| `freq` | `440` | Frequency in Hz (sine only) |
| `alpha` | `0.10` | Fractional order (0 – 4) |
| `blocksize` | `16384` | FRFT block size, or `all` for one-shot |
| `overlap` | `4` | Overlap factor (1, 2, 4 …) |
| `halfspec` | `0` | `1` to enable half-spectrum mode |
| `url` | — | URL of an audio file to fetch and process (see below) |

A bottom settings bar shows all active parameters.

---

#### Source modes

**Generated signals**

```
?type=sweep&dur=10&alpha=0.1&blocksize=16384&overlap=4
?type=sine&freq=880&dur=5&alpha=0.5
```

**File from your assets folder**

Pass a relative or absolute path via `url=`. No CORS issues since it's same-origin:

```
?url=../assets/piano.wav&alpha=0.1&blocksize=16384&overlap=4
?url=/assets/piano.wav&alpha=0.25
```

**File from an external URL**

The remote server must send `Access-Control-Allow-Origin: *`. Freesound's CDN does this, so you can use preview URLs directly:

```
?url=https://cdn.freesound.org/previews/414/414090_4921277-hq.mp3&alpha=0.1
```

To find a Freesound preview URL: open the sound page → right-click the waveform player → **Copy audio address**.

**Manual file picker**

When `type=file` is set without a `url=`, a **📂 Choose file** button appears in the controls bar. The user picks a file and it auto-processes:

```
?type=file&alpha=0.1&blocksize=16384&overlap=4
```

---

### Test harness

Open `web/demos/test-embed.html` in a local server. It renders both embeds side by side with controls for width, height, and all URL params, reloading the iframes on demand.

```bash
cd web
python3 -m http.server 8000
# open http://localhost:8000/demos/test-embed.html
```
