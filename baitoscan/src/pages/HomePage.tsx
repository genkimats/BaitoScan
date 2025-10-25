import PageWrapper from "../components/Layout/PageWrapper";
import ImageCapture from "../components/Capture/ImageCapture";
import { useOCRWorker } from "../hooks/useOCRWorker";

export default function HomePage() {
  const { isReady, error } = useOCRWorker();

  const handleImageSelect = (file: File) => {
    console.log("📸 Selected image:", file.name);
    // In the future: send preprocessed image tensor to worker
  };

  return (
    <PageWrapper>
      <div className="grid gap-6 max-w-md mx-auto text-center">
        <h2 className="text-xl font-semibold">Scan Your Shift</h2>
        <p className="text-gray-600 text-sm">
          Upload or take a photo of your handwritten work hours.
        </p>
        <ImageCapture onImageSelect={handleImageSelect} />
        <div className="text-sm mt-3">
          {isReady ? (
            <span className="text-green-600">✅ OCR worker loaded</span>
          ) : (
            <span className="text-gray-500">Loading OCR model...</span>
          )}
          {error && <p className="text-red-600">Error: {error}</p>}
        </div>
      </div>
    </PageWrapper>
  );
}
