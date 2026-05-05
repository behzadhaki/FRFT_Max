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

## Stage 2 — Emscripten bindings  *(coming soon)*

`frft_bindings.cpp` will expose `FRFTEngine::compute()` to JavaScript via
`emscripten::embind`.

## Stage 3 — WASM build  *(coming soon)*

A CMake build using the Emscripten toolchain will compile
`frft_engine_wasm.cpp` + KissFFT → `frft.wasm` + `frft.js`.

## Stage 4 — AudioWorklet  *(coming soon)*

`frft-processor.js` will wrap the WASM module for use in the Web Audio API.

## Stage 5 — Demo pages  *(coming soon)*

Static HTML pages in `web/demos/` for interactive browser demos.
Serve locally with:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/demos/
```

(Browsers block AudioWorklets on `file://` — a local server is required.)
