// frft-worker.js — runs the offline FRFT+WOLA loop in a dedicated thread,
// keeping the main thread (and UI) fully responsive during computation.
//
// Message protocol:
//   Main → Worker:
//     { type: 'init',    jsUrl, wasmBaseUrl }          load WASM module
//     { type: 'process', samples, alpha, bufSize,
//                        overlapFactor, jobId,
//                        halfSpectrum }                 start a job
//     { type: 'cancel' }                               cancel running job
//
//   Worker → Main:
//     { type: 'ready' }
//     { type: 'progress', jobId, value }               0..1
//     { type: 'result',   jobId, output }              Float32Array (transferable)
//     { type: 'cancelled', jobId }
//     { type: 'error',    message }
//
// halfSpectrum mode pipeline (block size B):
//   1. Zero-phase analysis: window B time-domain samples with Hann_B, then
//      circularly shift by B/2 within the frame (matches pfft~'s fftin~ convention,
//      equivalent to multiplying every FFT bin k by (-1)^k = e^{jπk}).
//      Zero-pad to fftN = nextPow2(B) for radix-2 FFT.
//   2. Run radix-2 FFT → take first frftN = fftN/2 bins as complex half-spectrum.
//   3. Feed bins directly to FRFT engine (engine applies internal fftshift).
//   4. FRFT (size frftN) via WASM.
//   5. fftshift engine output (centered→natural), real IFFT → fftN real samples.
//   6. Zero-phase synthesis: read IFFT output at (i + B/2) % B, multiply by Hann_B,
//      WOLA into output (matches pfft~'s fftout~ reconstruction).
//
// This approach works for any B (including non-power-of-2 "all" mode) by
// zero-padding to the next power of 2 for the FFT/IFFT step only.

'use strict';

let Module = null;

// currentJobId tracks which job is authorised to run.
// Setting it to -1 cancels whatever is running.
let currentJobId = -1;

self.onmessage = (e) => {
    const { type } = e.data;

    if (type === 'init') {
        (async () => {
            try {
                importScripts(e.data.jsUrl);
                Module = await FRFTModule({
                    locateFile: (f) => e.data.wasmBaseUrl + f + '?v=' + e.data.cacheBust,
                });
                self.postMessage({ type: 'ready' });
            } catch (err) {
                self.postMessage({ type: 'error', message: String(err) });
            }
        })();
        return;
    }

    if (type === 'cancel') {
        currentJobId = -1;
        return;
    }

    if (type === 'process') {
        const { samples, alpha, bufSize, overlapFactor, jobId, halfSpectrum, noWindow } = e.data;
        currentJobId = jobId;

        (async () => {
            try {
                const result = await computeFRFT(
                    samples, alpha, bufSize, overlapFactor, jobId,
                    halfSpectrum === true, noWindow === true
                );
                if (result === null) {
                    self.postMessage({ type: 'cancelled', jobId });
                } else {
                    self.postMessage({ type: 'result', jobId, output: result }, [result.buffer]);
                }
            } catch (err) {
                self.postMessage({ type: 'error', message: String(err) });
            }
        })();
    }
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}

// In-place radix-2 Cooley-Tukey FFT (forward).  N must be a power of 2.
function fftInPlace(re, im, N) {
    let j = 0;
    for (let i = 1; i < N; i++) {
        let bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            let t = re[i]; re[i] = re[j]; re[j] = t;
                t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (let len = 2; len <= N; len <<= 1) {
        const half = len >> 1;
        const ang  = -2 * Math.PI / len;
        const wcos = Math.cos(ang), wsin = Math.sin(ang);
        for (let i = 0; i < N; i += len) {
            let wr = 1, wi = 0;
            for (let k = 0; k < half; k++) {
                const tr = wr*re[i+k+half] - wi*im[i+k+half];
                const ti = wr*im[i+k+half] + wi*re[i+k+half];
                re[i+k+half] = re[i+k] - tr;
                im[i+k+half] = im[i+k] - ti;
                re[i+k] += tr;
                im[i+k] += ti;
                const nwr = wr*wcos - wi*wsin;
                wi = wr*wsin + wi*wcos; wr = nwr;
            }
        }
    }
}

// Real IFFT: halfN complex bins → 2*halfN real samples.
// Extends with conjugate symmetry, applies IFFT, writes real part into re[].
// re[] and im[] must each have length 2*halfN and are used as scratch.
function realIfft(halfSpecRe, halfSpecIm, halfN, re, im) {
    const N = 2 * halfN;

    // DC bin — forced real
    re[0] = halfSpecRe[0];
    im[0] = 0.0;

    // Positive frequencies
    for (let k = 1; k < halfN; k++) {
        re[k] =  halfSpecRe[k];
        im[k] =  halfSpecIm[k];
    }

    // Nyquist bin — not available from half-spectrum, set to 0
    re[halfN] = 0.0;
    im[halfN] = 0.0;

    // Negative frequencies — conjugate symmetry
    for (let k = 1; k < halfN; k++) {
        re[N - k] =  halfSpecRe[k];
        im[N - k] = -halfSpecIm[k];
    }

    // IFFT via conjugate trick: IFFT(X) = (1/N) conj(FFT(conj(X)))
    for (let i = 0; i < N; i++) im[i] = -im[i];
    fftInPlace(re, im, N);
    const scale = 1.0 / N;
    for (let i = 0; i < N; i++) re[i] *= scale;
    // re[0..N-1] now holds the N real output samples; im[] is discarded
}

// ── Core computation ──────────────────────────────────────────────────────────

async function computeFRFT(samples, alpha, bufSize, overlapFactor, jobId, halfSpectrum, noWindow) {
    // frameN  — time-domain window / OLA frame size (= bufSize)
    // fftN    — FFT size for half-spectrum: nextPow2(frameN) so radix-2 works
    //           for any bufSize including non-power-of-2 "all" mode
    // frftN   — FRFT block size
    //   full spectrum: frftN = frameN   (real input, imag = 0)
    //   half spectrum: frftN = fftN/2  (complex freq-domain bins)
    const frameN = bufSize;
    const fftN   = halfSpectrum ? nextPow2(frameN) : frameN;
    const frftN  = halfSpectrum ? fftN >> 1 : frameN;
    const H      = Math.max(1, Math.floor(frameN / overlapFactor));
    const L      = samples.length;

    const proc = new Module.FRFTProcessor();
    proc.prepare(frftN);

    const inRPtr  = proc.inputRealPtr();
    const inIPtr  = proc.inputImagPtr();
    const outRPtr = proc.outputRealPtr();
    const outIPtr = proc.outputImagPtr();

    let heapBuf  = Module.HEAPF64.buffer;
    let heapInR  = new Float64Array(heapBuf, inRPtr,  frftN);
    let heapInI  = new Float64Array(heapBuf, inIPtr,  frftN);
    let heapOutR = new Float64Array(heapBuf, outRPtr, frftN);
    let heapOutI = new Float64Array(heapBuf, outIPtr, frftN);

    function remapHeap() {
        const buf = Module.HEAPF64.buffer;
        if (buf !== heapBuf) {
            heapBuf  = buf;
            heapInR  = new Float64Array(buf, inRPtr,  frftN);
            heapInI  = new Float64Array(buf, inIPtr,  frftN);
            heapOutR = new Float64Array(buf, outRPtr, frftN);
            heapOutI = new Float64Array(buf, outIPtr, frftN);
        }
    }

    // Analysis + synthesis window (size frameN for both modes).
    // noWindow → rectangular (no tapering), matching the paper's no-window mode.
    const win = new Float32Array(frameN);
    if (noWindow) {
        win.fill(1.0);
    } else {
        for (let i = 0; i < frameN; i++)
            win[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / (frameN - 1));
    }

    // WOLA normalisation (synthesis window size frameN)
    let wNorm = 0;
    for (let p = 0; p < H; p++) {
        for (let k = 0; k < overlapFactor; k++) {
            const wi = p + k * H;
            if (wi < frameN) wNorm += win[wi] * win[wi];
        }
    }
    wNorm /= H;

    const output     = new Float32Array(L);
    const firstFrame = -(overlapFactor - 1) * H;

    let totalFrames = 0;
    for (let fs = firstFrame; fs < L; fs += H) totalFrames++;
    let framesDone = 0;

    // Scratch buffers for half-spectrum (fftN size — covers zero-padded frame)
    const fftRe  = halfSpectrum ? new Float64Array(fftN) : null;
    const fftIm  = halfSpectrum ? new Float64Array(fftN) : null;
    const ifftRe = halfSpectrum ? new Float64Array(fftN) : null;  // realIfft output (fftN)
    const ifftIm = halfSpectrum ? new Float64Array(fftN) : null;  // realIfft scratch

    // fftshift offset for the FRFT engine = frftN/2
    const halfFrft  = frftN >> 1;
    // Zero-phase circular-shift amount for half-spectrum analysis/synthesis
    const halfFrame = frameN >> 1;

    const YIELD_MS = 50;
    let lastYield  = performance.now();

    for (let frameStart = firstFrame; frameStart < L; frameStart += H) {
        remapHeap();

        if (halfSpectrum) {
            // ── Half-spectrum path ────────────────────────────────────────────
            // Matches pfft~'s zero-phase analysis/synthesis convention:
            //   analysis  — window sample[i] and place at (i + halfFrame) % frameN
            //               before FFT (= circular shift of windowed frame by N/2)
            //   synthesis — read ifftRe[(i + halfFrame) % frameN] and multiply by
            //               win[i] before accumulating (= circular shift back)
            // This ensures our FFT bins have the same (-1)^k phase factor that
            // pfft~'s fftin~/fftout~ introduce via their zero-phase alignment.

            // 1. Zero-phase analysis: window + circularly shift by halfFrame,
            //    zero-pad to fftN for radix-2 FFT.
            for (let i = 0; i < fftN; i++) { fftRe[i] = 0.0; fftIm[i] = 0.0; }
            for (let i = 0; i < frameN; i++) {
                const idx = frameStart + i;
                fftRe[(i + halfFrame) % frameN] =
                    (idx >= 0 && idx < L ? samples[idx] : 0.0) * win[i];
            }

            // 2. FFT of fftN → first frftN bins are the positive half-spectrum
            fftInPlace(fftRe, fftIm, fftN);

            // 3. Feed bins directly — engine's internal fftshift centers DC
            //    (no pre-fftshift here, unlike the time-domain full-spectrum path)
            for (let i = 0; i < frftN; i++) {
                heapInR[i] = fftRe[i];
                heapInI[i] = fftIm[i];
            }

            // 4. FRFT
            proc.process(frftN, alpha);
            remapHeap();

            // 5. fftshift output (centered → natural order), then real IFFT
            for (let i = 0; i < frftN; i++) {
                fftRe[i] = heapOutR[(i + halfFrft) % frftN];
                fftIm[i] = heapOutI[(i + halfFrft) % frftN];
            }
            realIfft(fftRe, fftIm, frftN, ifftRe, ifftIm);

            // 6. Zero-phase synthesis: undo circular shift (read from
            //    (i + halfFrame) % frameN), apply synthesis Hann + OLA.
            for (let i = 0; i < frameN; i++) {
                const idx = frameStart + i;
                if (idx >= 0 && idx < L)
                    output[idx] += ifftRe[(i + halfFrame) % frameN] * win[i];
            }

        } else {
            // ── Full-spectrum path (original, unchanged) ──────────────────────

            for (let i = 0; i < frftN; i++) {
                const idx = frameStart + i;
                heapInR[(i + halfFrft) % frftN] =
                    (idx >= 0 && idx < L ? samples[idx] : 0.0) * win[i];
            }
            heapInI.fill(0.0);

            const amod = ((alpha % 4) + 4) % 4;
            const isId = amod < 1e-9 || amod > 4 - 1e-9;

            if (isId) {
                for (let i = 0; i < frftN; i++) {
                    const idx = frameStart + i;
                    if (idx >= 0 && idx < L)
                        output[idx] += heapInR[(i + halfFrft) % frftN] * win[i];
                }
            } else {
                proc.process(frftN, alpha);
                remapHeap();
                for (let i = 0; i < frftN; i++) {
                    const w = win[i];
                    if (w < 1e-10) continue;
                    const idx = frameStart + i;
                    if (idx >= 0 && idx < L)
                        output[idx] += heapOutR[i] * w;
                }
            }
        }

        framesDone++;

        const now = performance.now();
        if (now - lastYield >= YIELD_MS) {
            lastYield = now;
            self.postMessage({ type: 'progress', jobId, value: framesDone / totalFrames });
            await new Promise(r => setTimeout(r, 0));
            if (currentJobId !== jobId) { proc.delete(); return null; }
        }
    }

    self.postMessage({ type: 'progress', jobId, value: 1 });

    // Scalar normalisation
    for (let i = 0; i < L; i++) output[i] /= wNorm;

    // Peak-normalise to prevent clipping
    let peak = 0;
    for (let i = 0; i < L; i++) { const a = Math.abs(output[i]); if (a > peak) peak = a; }
    if (peak > 1e-8) for (let i = 0; i < L; i++) output[i] /= peak;

    // Trim OLA-invalid edges: (R-1)*H samples each side
    const trimAmt = (overlapFactor - 1) * H;
    const trimEnd = L - trimAmt;

    proc.delete();
    return trimAmt > 0 && trimEnd > trimAmt ? output.slice(trimAmt, trimEnd) : output;
}
