// AudioWorklet processor for the FRFT engine.
//
// Initialisation flow:
//   1. Main thread registers this module via audioContext.audioWorklet.addModule()
//   2. Main thread creates an AudioWorkletNode and posts { type: 'init', frftJsUrl }
//   3. This processor loads frft.js via importScripts, instantiates FRFTProcessor,
//      then posts { type: 'ready' } back to the main thread
//
// Audio processing:
//   - Input:  mono signal (real audio, imaginary = 0)
//   - Output: real part of the FRFT result
//   - Alpha:  sent at any time via { type: 'setAlpha', value: <number> }
//
// Buffer accumulation:
//   Web Audio delivers 128 samples per render quantum. For FRFT to capture
//   meaningful frequency content, samples are accumulated into a larger block
//   (bufSize, default 512). This adds one block of latency.

class FRFTAudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        this._ready   = false;
        this._alpha   = (options.processorOptions && options.processorOptions.alpha) || 0.5;
        this._bufSize = (options.processorOptions && options.processorOptions.bufSize) || 512;

        // Ring buffers: accumulate input, drain output
        this._inBuf  = new Float32Array(this._bufSize);
        this._outBuf = new Float32Array(this._bufSize);
        this._inPos  = 0;   // write head in _inBuf
        this._outPos = this._bufSize; // read head — starts at end so first block outputs silence

        this.port.onmessage = (e) => {
            const { type } = e.data;

            if (type === 'init') {
                // Load Emscripten glue synchronously, then instantiate async
                importScripts(e.data.frftJsUrl);
                FRFTModule().then(Module => {
                    this._Module = Module;
                    this._proc   = new Module.FRFTProcessor();
                    this._proc.prepare(this._bufSize);

                    // Map reusable typed views onto WASM heap (Pattern A)
                    this._heapInR  = new Float64Array(Module.HEAPF64.buffer,
                                         this._proc.inputRealPtr(),  this._bufSize);
                    this._heapInI  = new Float64Array(Module.HEAPF64.buffer,
                                         this._proc.inputImagPtr(),  this._bufSize);
                    this._heapOutR = new Float64Array(Module.HEAPF64.buffer,
                                         this._proc.outputRealPtr(), this._bufSize);

                    this._ready = true;
                    this.port.postMessage({ type: 'ready' });
                });
            }

            if (type === 'setAlpha') {
                this._alpha = e.data.value;
            }

            if (type === 'setBufSize') {
                // Resize buffers — takes effect on next full block
                const n = e.data.value;
                if (n % 2 !== 0 || n < 2) return; // must be even
                this._bufSize = n;
                this._inBuf   = new Float32Array(n);
                this._outBuf  = new Float32Array(n);
                this._inPos   = 0;
                this._outPos  = n; // drain silence until first block is ready

                if (this._ready) {
                    this._proc.prepare(n);
                    this._heapInR  = new Float64Array(this._Module.HEAPF64.buffer,
                                         this._proc.inputRealPtr(),  n);
                    this._heapInI  = new Float64Array(this._Module.HEAPF64.buffer,
                                         this._proc.inputImagPtr(),  n);
                    this._heapOutR = new Float64Array(this._Module.HEAPF64.buffer,
                                         this._proc.outputRealPtr(), n);
                }
            }
        };
    }

    // Called every 128 samples by the audio thread
    process(inputs, outputs) {
        const input  = inputs[0]?.[0];   // mono input channel
        const output = outputs[0]?.[0];  // mono output channel

        if (!output) return true;

        const frameSize = output.length; // always 128

        for (let i = 0; i < frameSize; i++) {
            // ---- Accumulate input ----------------------------------------
            this._inBuf[this._inPos++] = input ? input[i] : 0.0;

            if (this._inPos === this._bufSize) {
                this._inPos = 0;

                if (this._ready) {
                    // Write into WASM heap (real audio, imaginary = 0)
                    for (let k = 0; k < this._bufSize; k++) {
                        this._heapInR[k] = this._inBuf[k];
                    }
                    this._heapInI.fill(0.0);

                    // Run FRFT
                    this._proc.process(this._bufSize, this._alpha);

                    // Copy real part of result to output ring buffer
                    for (let k = 0; k < this._bufSize; k++) {
                        this._outBuf[k] = this._heapOutR[k];
                    }
                    this._outPos = 0;
                }
            }

            // ---- Drain output --------------------------------------------
            output[i] = this._outPos < this._bufSize
                ? this._outBuf[this._outPos++]
                : 0.0; // silence until first block is ready
        }

        return true; // keep processor alive
    }
}

registerProcessor('frft-processor', FRFTAudioProcessor);
