import { useEffect, useRef, useState } from "react";

export function useOCRWorker() {
  const workerRef = useRef<Worker | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/ocr.worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;

    worker.onmessage = (e) => {
      const { type, message } = e.data;
      if (type === "ready") setIsReady(true);
      if (type === "error") setError(message);
    };

    worker.postMessage({ type: "load", payload: { url: "/models/baitoscan-crnn.onnx" } });

    return () => {
      worker.terminate();
    };
  }, []);

  const runInference = (input: Float32Array, shape: number[]) => {
    return new Promise<string>((resolve, reject) => {
      const worker = workerRef.current;
      if (!worker || !isReady) return reject("Worker not ready");

      const handleResult = (e: MessageEvent) => {
        if (e.data.type === "result") {
          resolve(e.data.text);
          worker.removeEventListener("message", handleResult);
        }
        if (e.data.type === "error") {
          worker.removeEventListener("message", handleResult);
          reject(e.data.message ?? "Inference failed");
        }
      };

      worker.addEventListener("message", handleResult);
      worker.postMessage({ type: "infer", payload: { input, shape } });
    });
  };

  return { isReady, error, runInference };
}
