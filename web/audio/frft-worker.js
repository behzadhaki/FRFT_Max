// frft-worker.js — runs the offline FRFT+WOLA loop in a dedicated thread,
// keeping the main thread (and UI) fully responsive during computation.
//
// Message protocol:
//   Main → Worker:
//     { type: 'init',    jsUrl, wasmBaseUrl }          load WASM module
//     { type: 'process', samples, alpha, bufSize,
//                        overlapFactor, jobId }         start a job
//     { type: 'cancel' }                               cancel running job
//
//   Worker → Main:
//     { type: 'ready' }
//     { type: 'progress', jobId, value }               0..1
//     { type: 'result',   jobId, output }              Float32Array (transferable)
//     { type: 'cancelled', jobId }
//     { type: 'error',    message }

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
                    locateFile: (f) => e.data.wasmBaseUrl + f,
                });
                self.postMessage({ type: 'ready' });
            } catch (err) {
                self.postMessage({ type: 'error', message: String(err) });
            }
        })();
        return;
    }

    if (type === 'cancel') {
        currentJobId = -1;  // any running job will see its id no longer matches
        return;
    }

    if (type === 'process') {
        const { samples, alpha, bufSize, overlapFactor, jobId } = e.data;
        currentJobId = jobId;

        (async () => {
            try {
                const result = await computeFRFT(samples, alpha, bufSize, overlapFactor, jobId);
                if (result === null) {
                    self.postMessage({ type: 'cancelled', jobId });
                } else {
                    // Transfer the ArrayBuffer so main thread gets it zero-copy.
                    self.postMessage({ type: 'result', jobId, output: result }, [result.buffer]);
                }
            } catch (err) {
                self.postMessage({ type: 'error', message: String(err) });
            }
        })();
    }
};

// ── Core computation ────────────────────────────────────────────────────────

async function computeFRFT(samples, alpha, bufSize, overlapFactor, jobId) {
    const N = bufSize;
    const H = Math.max(1, Math.floor(N / overlapFactor));
    const L = samples.length;

    const proc = new Module.FRFTProcessor();
    proc.prepare(N);

    const inRPtr  = proc.inputRealPtr();
    const inIPtr  = proc.inputImagPtr();
    const outRPtr = proc.outputRealPtr();

    let heapBuf = Module.HEAPF64.buffer;
    let heapInR  = new Float64Array(heapBuf, inRPtr,  N);
    let heapInI  = new Float64Array(heapBuf, inIPtr,  N);
    let heapOutR = new Float64Array(heapBuf, outRPtr, N);

    function remapHeap() {
        const buf = Module.HEAPF64.buffer;
        if (buf !== heapBuf) {
            heapBuf  = buf;
            heapInR  = new Float64Array(buf, inRPtr,  N);
            heapInI  = new Float64Array(buf, inIPtr,  N);
            heapOutR = new Float64Array(buf, outRPtr, N);
        }
    }

    // Hann analysis + synthesis window
    const win = new Float32Array(N);
    for (let i = 0; i < N; i++)
        win[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / (N - 1));

    // Scalar WOLA normalisation
    let wNorm = 0;
    for (let p = 0; p < H; p++) {
        for (let k = 0; k < overlapFactor; k++) {
            const wi = p + k * H;
            if (wi < N) wNorm += win[wi] * win[wi];
        }
    }
    wNorm /= H;

    const output    = new Float32Array(L);
    const firstFrame = -(overlapFactor - 1) * H;

    let totalFrames = 0;
    for (let fs = firstFrame; fs < L; fs += H) totalFrames++;
    let framesDone = 0;

    // Time-based yielding: yield whenever ≥50 ms of wall-clock time has elapsed
    // since the last yield.  This adapts automatically:
    //   • Small N, many frames  → batch many frames per yield (low overhead)
    //   • Large N, few frames   → yield after every frame (each takes >50 ms)
    //   • "All" mode, 1 frame   → no yield possible mid-call; pulse shows activity
    const YIELD_MS = 50;
    let lastYield  = performance.now();

    for (let frameStart = firstFrame; frameStart < L; frameStart += H) {
        remapHeap();

        for (let i = 0; i < N; i++) {
            const idx = frameStart + i;
            heapInR[i] = (idx >= 0 && idx < L ? samples[idx] : 0.0) * win[i];
        }
        heapInI.fill(0.0);

        proc.process(N, alpha);
        remapHeap();

        for (let i = 0; i < N; i++) {
            const w = win[i];
            if (w < 1e-10) continue;
            const idx = frameStart + i;
            if (idx >= 0 && idx < L)
                output[idx] += heapOutR[i] * w;
        }

        framesDone++;

        const now = performance.now();
        if (now - lastYield >= YIELD_MS) {
            lastYield = now;
            self.postMessage({ type: 'progress', jobId, value: framesDone / totalFrames });
            await new Promise(r => setTimeout(r, 0));
            // After yielding, check if this job has been superseded
            if (currentJobId !== jobId) { proc.delete(); return null; }
        }
    }

    // Send final 100% progress tick
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
