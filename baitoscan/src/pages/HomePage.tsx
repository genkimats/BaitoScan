import { useState } from "react";
import PageWrapper from "../components/Layout/PageWrapper";
import ImageCapture from "../components/Capture/ImageCapture";
import { useOCRWorker } from "../hooks/useOCRWorker";
import { imageToTensor } from "../lib/image";

export default function HomePage() {
  const { isReady, error, runInference } = useOCRWorker();
  const [result, setResult] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);

  const handleImageSelect = async (file: File) => {
    if (!isReady) {
      alert("OCR model is still loading. Try again in a moment.");
      return;
    }
    try {
      setIsProcessing(true);
      const tensor = await imageToTensor(file);
      const shape = [1, 1, 64, 256];
      const text = await runInference(tensor, shape);
      setResult(text);
    } catch (err) {
      // Log to surface unexpected inference failures during development
      console.error(err);
      setResult("");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <PageWrapper>
      <div className="grid gap-6 max-w-md mx-auto text-center">
        <h2 className="text-xl font-semibold">Scan Your Shift</h2>
        <ImageCapture onImageSelect={handleImageSelect} />
        <div className="text-sm mt-3">
          {isReady ? (
            <span className="text-green-600">✅ OCR worker ready</span>
          ) : (
            <span className="text-gray-500">Loading OCR model...</span>
          )}
          {error && <p className="text-red-600">Error: {error}</p>}
        </div>
        {isProcessing && <p className="text-sm text-blue-600">Reading receipt…</p>}
        {result && !isProcessing && (
          <div className="bg-gray-100 rounded-md p-3 mt-3 text-left">
            <p className="font-mono text-sm break-words">{result}</p>
          </div>
        )}
      </div>
    </PageWrapper>
  );
}
