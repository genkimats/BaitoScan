import * as ort from "onnxruntime-web";
import wasmModuleUrl from "../assets/onnxruntime/ort-wasm-simd-threaded.jsep.mjs?url";
import wasmBinaryUrl from "../assets/onnxruntime/ort-wasm-simd-threaded.jsep.wasm?url";

ort.env.wasm.wasmPaths = {
  wasm: wasmBinaryUrl,
  mjs: wasmModuleUrl,
};

ort.env.wasm.simd = true;
ort.env.wasm.numThreads = 1;

console.log("🧠 onnxruntime wasm path set:", ort.env.wasm.wasmPaths);

let session: ort.InferenceSession | null = null;
// === Character Set ===
// Must match Python's CHARS = "0123456789/:~ "
// +1 blank token at the end handled separately
const CHARSET = "0123456789/:~ ";
const BLANK_INDEX = CHARSET.length; // last class index

function ctcGreedyDecode(logits: any): { text: string; conf: number } {
  // logits is Tensor from onnxruntime (Float32Array)
  // Shape: [1, seqLen, numClasses]
  const data = logits.data;
  const [batch, seqLen, numClasses] = logits.dims;

  let prev = -1;
  let text = "";
  let totalConf = 0;
  let count = 0;

  for (let t = 0; t < seqLen; t++) {
    // Find argmax for this timestep
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const val = data[t * numClasses + c];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = c;
      }
    }

    // Skip blanks and duplicates
    if (maxIdx !== BLANK_INDEX && maxIdx !== prev) {
      text += CHARSET[maxIdx];
      totalConf += Math.exp(maxVal); // convert log prob to prob estimate
      count++;
    }
    prev = maxIdx;
  }

  const conf = count ? totalConf / count : 0;
  return { text, conf };
}


self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  try {
    switch (type) {
      case "load": {
        session = await ort.InferenceSession.create("/models/baitoscan-crnn_1.onnx", {
          executionProviders: ["wasm"],
        });
        (self as any).postMessage({ type: "ready" });
        break;
      }

      case "infer": {
        if (!session) throw new Error("Model not loaded");
        const { input, shape } = payload as { input: Float32Array; shape: number[] };
        const tensor = new ort.Tensor("float32", input, shape);
        const output = await session.run({ input: tensor });
        const logits = output[Object.keys(output)[0]];
        const decoded = ctcGreedyDecode(logits);
        console.log("🧾 OCR decoded text:", decoded.text, "(confidence:", decoded.conf.toFixed(2) + ")");
        (self as any).postMessage({
          type: "result",
          text: decoded.text,
          confidence: decoded.conf,
        });
        break;
      }
    }
  } catch (err: any) {
    (self as any).postMessage({ type: "error", message: err.message });
  }
};
