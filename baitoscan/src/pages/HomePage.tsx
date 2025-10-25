import PageWrapper from "../components/Layout/PageWrapper";
import ImageCapture from "../components/Capture/ImageCapture";

export default function HomePage() {
  const handleImageSelect = (file: File) => {
    console.log("Selected image:", file.name);
    // Future: send to OCR worker
  };

  return (
    <PageWrapper>
      <div className="grid gap-6 max-w-md mx-auto">
        <h2 className="text-xl font-semibold text-center">Scan Your Shift</h2>
        <p className="text-gray-600 text-center text-sm">
          Upload or take a photo of your handwritten work hours.
        </p>
        <ImageCapture onImageSelect={handleImageSelect} />
      </div>
    </PageWrapper>
  );
}
