// AudioWorklet processor — Hann windowing + overlap-add FRFT.
//
// Init flow:
//   Main thread posts { type:'init', jsText, wasmBinary } after addModule().
//   Processor evaluates jsText (Emscripten glue) in global scope via indirect
//   eval, instantiates FRFTModule with the pre-fetched binary, then replies
//   { type:'ready' }.
//
// Processing:
//   Input samples are written into a ring buffer. Every hopSize samples a new
//   FRFT frame is triggered:
//     1. Copy last bufSize samples from ring → apply Hamming window → WASM heap
//     2. Run FRFT (imaginary input = 0)
//     3. WOLA: multiply real output by synthesis window, add into accumulator
//     4. Drain hopSize normalised samples into output FIFO
//     5. Shift accumulator left by hopSize
//
// Parameters (sent via MessagePort at any time):
//   { type:'setAlpha',         value: number }   fractional order [-2, 2]
//   { type:'setBufSize',       value: number }   even integer
//   { type:'setOverlapFactor', value: number }   one of 1,2,4,8,16,32

const VALID_OVERLAP = new Set([1, 2, 4, 8, 16, 32]);

class FRFTAudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const p = options.processorOptions || {};
        this._alpha         = p.alpha         ?? 0.5;
        this._bufSize       = p.bufSize       ?? 512;
        this._overlapFactor = p.overlapFactor ?? 4;
        this._ready         = false;

        this._allocBuffers();

        this.port.onmessage = (e) => {
            const { type } = e.data;

            if (type === 'init') {
                try {
                    // Indirect eval → var FRFTModule lands on globalThis
                    (0, eval)(e.data.jsText);

                    FRFTModule({ wasmBinary: e.data.wasmBinary })
                    .then(M => {
                        this._M    = M;
                        this._proc = new M.FRFTProcessor();
                        this._proc.prepare(this._bufSize);
                        this._mapHeap();
                        this._ready = true;
                        this.port.postMessage({ type: 'ready' });
                    })
                    .catch(err => this.port.postMessage({ type: 'error', message: String(err) }));
                } catch (err) {
                    this.port.postMessage({ type: 'error', message: String(err) });
                }
            }

            if (type === 'setAlpha') {
                this._alpha = e.data.value;
            }

            if (type === 'setBufSize') {
                const n = e.data.value;
                if (n < 2 || n % 2 !== 0) return;
                this._bufSize = n;
                this._allocBuffers();
                if (this._ready) { this._proc.prepare(n); this._mapHeap(); }
            }

            if (type === 'setOverlapFactor') {
                if (!VALID_OVERLAP.has(e.data.value)) return;
                this._overlapFactor = e.data.value;
                this._allocBuffers();
                // No need to re-prepare WASM — bufSize unchanged
            }
        };
    }

    // ── Buffer allocation ──────────────────────────────────────────────────

    _allocBuffers() {
        const N = this._bufSize;
        const R = this._overlapFactor;
        const H = N / R;           // hopSize (always integer: N and R are powers-of-2-friendly)
        this._hopSize = H;

        // Hann window — tapers to exactly 0 at both edges, eliminating
        // frame-boundary discontinuities regardless of overlap factor.
        this._win = new Float32Array(N);
        for (let i = 0; i < N; i++)
            this._win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1));

        // WOLA normalisation: mean of the sum of R overlapping (win_analysis × win_synthesis)
        // products at positions 0..H-1.  Both windows are the same Hann, so this is
        // mean of sum(win²).  Satisfies COLA exactly at ≥50% overlap (R≥2).
        let normSum = 0;
        for (let p = 0; p < H; p++) {
            for (let k = 0; k < R; k++) {
                const wi = p + k * H;
                if (wi < N) normSum += this._win[wi] * this._win[wi];
            }
        }
        this._norm = normSum / H;

        // Circular input ring buffer (holds last N samples)
        this._inRing     = new Float32Array(N);
        this._inWritePos = 0;
        this._hopCounter = 0;

        // Working frame (for FRFT input, reused each hop)
        this._frame = new Float32Array(N);

        // OLA accumulator
        this._ola = new Float32Array(N);

        // Output FIFO (circular, size N is enough: max H*(R-1) < N samples queued)
        this._outQ     = new Float32Array(N);
        this._outRead  = 0;
        this._outWrite = 0;
        this._outCount = 0;
    }

    // Remap WASM heap views after bufSize change or first init
    _mapHeap() {
        const N = this._bufSize;
        const b = this._M.HEAPF64.buffer;
        this._heapInR  = new Float64Array(b, this._proc.inputRealPtr(),  N);
        this._heapInI  = new Float64Array(b, this._proc.inputImagPtr(),  N);
        this._heapOutR = new Float64Array(b, this._proc.outputRealPtr(), N);
    }

    // ── FRFT frame + OLA ──────────────────────────────────────────────────

    _processFrame() {
        const N = this._bufSize;
        const H = this._hopSize;

        // Read last N samples from ring in chronological order
        const start = this._inWritePos; // oldest sample
        for (let i = 0; i < N; i++) {
            this._frame[i] = this._inRing[(start + i) % N];
        }

        // Apply Hamming window → WASM heap; imaginary = 0
        for (let i = 0; i < N; i++) {
            this._heapInR[i] = this._frame[i] * this._win[i];
        }
        this._heapInI.fill(0);

        // FRFT
        this._proc.process(N, this._alpha);

        // OLA: add synthesis-windowed real part of output into accumulator (WOLA).
        // The synthesis window ensures every frame tapers to zero at its edges,
        // eliminating inter-frame discontinuities regardless of alpha.
        for (let i = 0; i < N; i++) {
            this._ola[i] += this._heapOutR[i] * this._win[i];
        }

        // Drain hopSize normalised samples into output FIFO
        const norm = this._norm;
        for (let i = 0; i < H; i++) {
            this._outQ[this._outWrite] = this._ola[i] / norm;
            this._outWrite = (this._outWrite + 1) % N;
            this._outCount++;
        }

        // Shift OLA accumulator left by hopSize, zero the tail
        this._ola.copyWithin(0, H);
        this._ola.fill(0, N - H);
    }

    // ── Audio render callback ──────────────────────────────────────────────

    process(inputs, outputs) {
        const inCh  = inputs[0]?.[0];
        const outCh = outputs[0]?.[0];
        if (!outCh) return true;

        const N = this._bufSize;

        for (let i = 0; i < outCh.length; i++) {
            // Write new input sample into ring
            this._inRing[this._inWritePos] = inCh ? inCh[i] : 0;
            this._inWritePos = (this._inWritePos + 1) % N;

            // Trigger a frame every hopSize samples
            if (this._ready && ++this._hopCounter >= this._hopSize) {
                this._hopCounter = 0;
                this._processFrame();
            }

            // Output from FIFO (silence until first frame is ready)
            if (this._outCount > 0) {
                outCh[i] = this._outQ[this._outRead];
                this._outRead = (this._outRead + 1) % N;
                this._outCount--;
            } else {
                outCh[i] = 0;
            }
        }

        return true;
    }
}

registerProcessor('frft-processor', FRFTAudioProcessor);
