import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  try {
    switch (type) {
      case "load": {
        // load model (e.g. /models/baitoscan-crnn.onnx)
        const modelUrl = payload?.url || "/models/baitoscan-crnn.onnx";
        session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        (self as any).postMessage({ type: "ready" });
        break;
      }

      case "infer": {
        if (!session) throw new Error("Model not loaded");
        const { input, shape } = payload as { input: Float32Array; shape: number[] };
        const tensor = new ort.Tensor("float32", input, shape);
        const output = await session.run({ input: tensor });
        (self as any).postMessage({
          type: "result",
          result: output[Object.keys(output)[0]],
        });
        break;
      }

      default:
        console.warn("Unknown worker message:", type);
    }
  } catch (err: any) {
    (self as any).postMessage({ type: "error", message: err.message });
  }
};

export {}; // ensure it's a module
