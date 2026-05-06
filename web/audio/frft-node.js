// Main-thread helper for the FRFT AudioWorklet.
//
// Usage:
//   import { createFRFTNode } from './frft-node.js';
//
//   const ctx  = new AudioContext();
//   const node = await createFRFTNode(ctx, {
//       processorUrl: '/audio/frft-processor.js',
//       frftJsUrl:    '/wasm/dist/frft.js',
//       alpha:        0.5,   // initial fractional order
//       bufSize:      512,   // accumulation block size (must be even)
//   });
//
//   sourceNode.connect(node);
//   node.connect(ctx.destination);
//
//   node.setAlpha(0.75);   // change alpha at any time
//   node.setBufSize(1024); // change block size at any time

/**
 * Creates and initialises an FRFT AudioWorkletNode.
 *
 * @param {AudioContext} audioContext
 * @param {object} opts
 * @param {string} opts.processorUrl  URL of frft-processor.js
 * @param {string} opts.frftJsUrl     URL of the Emscripten frft.js glue file
 * @param {number} [opts.alpha=0.5]         Initial fractional order
 * @param {number} [opts.bufSize=512]       Processing block size (must be even)
 * @param {number} [opts.overlapFactor=4]   OLA overlap factor (1|2|4|8|16|32)
 * @returns {Promise<AudioWorkletNode>}
 */
export async function createFRFTNode(audioContext, opts = {}) {
    const {
        processorUrl  = '/audio/frft-processor.js',
        frftJsUrl     = '/wasm/dist/frft.js',
        alpha         = 0.5,
        bufSize       = 512,
        overlapFactor = 4,
    } = opts;

    // Register the processor script with this AudioContext
    await audioContext.audioWorklet.addModule(processorUrl);

    const node = new AudioWorkletNode(audioContext, 'frft-processor', {
        numberOfInputs:  1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: { alpha, bufSize, overlapFactor },
    });

    // Fetch both files on the main thread where URLs resolve correctly.
    // Sending the JS source as text avoids importScripts() path issues inside
    // the AudioWorklet; sending the binary skips Emscripten's internal fetch.
    const wasmUrl    = frftJsUrl.replace(/frft\.js$/, 'frft.wasm');
    const [jsText, wasmBinary] = await Promise.all([
        fetch(frftJsUrl).then(r => r.text()),
        fetch(wasmUrl).then(r => r.arrayBuffer()),
    ]);

    // Wait for the WASM module to finish loading inside the worklet
    const ready = new Promise((resolve, reject) => {
        node.port.onmessage = (e) => {
            if (e.data.type === 'ready') resolve();
            if (e.data.type === 'error') reject(new Error(e.data.message));
        };
        node.onprocessorerror = (e) => reject(new Error(e.message));
    });

    // Transfer the binary (zero-copy); JS text is a plain string copy
    node.port.postMessage({ type: 'init', jsText, wasmBinary }, [wasmBinary]);

    await ready;

    // ---- Convenience methods ------------------------------------------------

    /** Set the fractional order (clamped to [-2, 2]). */
    node.setAlpha = (value) => {
        const clamped = Math.max(-2, Math.min(2, value));
        node.port.postMessage({ type: 'setAlpha', value: clamped });
    };

    /** Change the processing block size (must be a positive even integer). */
    node.setBufSize = (value) => {
        node.port.postMessage({ type: 'setBufSize', value });
    };

    /** Change the OLA overlap factor (1 | 2 | 4 | 8 | 16 | 32). */
    node.setOverlapFactor = (value) => {
        node.port.postMessage({ type: 'setOverlapFactor', value });
    };

    return node;
}
