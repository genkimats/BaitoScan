import { useCallback, useEffect, useRef, useState } from "react";
import { prepareTesseractImage } from "../lib/image";

interface RecognizeResponse {
  text: string;
  confidence: number;
}

interface WorkerResultEvent {
  type: "result";
  id: string;
  text: string;
  confidence: number;
}

interface WorkerProgressEvent {
  type: "progress";
  id: string | null;
  progress: number;
  status: string;
}

interface WorkerErrorEvent {
  type: "error";
  id?: string;
  message: string;
}

type WorkerEvents = WorkerResultEvent | WorkerProgressEvent | WorkerErrorEvent | { type: "ready" };

export function useTesseractWorker(lang = "eng") {
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const [isReady, setIsReady] = useState(false);
  const [progress, setProgress] = useState<number>(0);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/tess.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;

    const handleMessage = (event: MessageEvent<WorkerEvents>) => {
      const data = event.data;
      switch (data.type) {
        case "ready":
          setIsReady(true);
          break;
        case "progress":
          if (typeof data.progress === "number") {
            setProgress(data.progress);
          }
          setStatus(data.status ?? "");
          break;
        case "error":
          setError(data.message);
          break;
      }
    };

    worker.addEventListener("message", handleMessage);
    worker.postMessage({ type: "load", payload: { lang } });

    return () => {
      worker.removeEventListener("message", handleMessage);
      worker.terminate();
      workerRef.current = null;
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
    };
  }, [lang]);

  const updatePreview = useCallback((blob: Blob | File) => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    const url = URL.createObjectURL(blob);
    previewUrlRef.current = url;
    setPreviewUrl(url);
  }, []);

  const recognize = useCallback(
    async (file: File) => {
      const worker = workerRef.current;
      if (!worker) throw new Error("Worker not initialized");
      if (!isReady) throw new Error("Tesseract worker not ready");

      const processed = await prepareTesseractImage(file);
      updatePreview(processed);
      const payload =
        processed instanceof File
          ? processed
          : new File([processed], file.name || "ocr-input.png", { type: "image/png" });
      const id = `job-${++requestIdRef.current}`;
      setError(null);
      setProgress(0);
      setStatus("");

      return new Promise<RecognizeResponse>((resolve, reject) => {
        const handleMessage = (event: MessageEvent<WorkerEvents>) => {
          const data = event.data;
          if (data.type === "result" && data.id === id) {
            worker.removeEventListener("message", handleMessage);
            resolve({ text: data.text, confidence: data.confidence });
          }
          if (data.type === "error" && data.id === id) {
            worker.removeEventListener("message", handleMessage);
            reject(new Error(data.message));
          }
        };

        worker.addEventListener("message", handleMessage);
        worker.postMessage({ type: "recognize", id, payload });
      });
    },
    [isReady, updatePreview],
  );

  const resetError = useCallback(() => {
    setError(null);
    setProgress(0);
    setStatus("");
  }, []);

  return {
    isReady,
    progress,
    status,
    error,
    recognize,
    resetError,
     previewUrl,
  };
}

export type TesseractWorkerHook = ReturnType<typeof useTesseractWorker>;
