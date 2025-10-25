import { useState } from "react";

interface ImageCaptureProps {
  onImageSelect: (file: File) => void;
}

export default function ImageCapture({ onImageSelect }: ImageCaptureProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please select an image file.");
      return;
    }
    setError(null);
    setPreview(URL.createObjectURL(f));
    onImageSelect(f);
  };

  return (
    <div className="grid gap-3 text-center">
      <label className="block">
        <input
          type="file"
          accept="image/*"
          capture="environment"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
          className="hidden"
          id="cameraInput"
        />
        <div className="border-2 border-dashed border-gray-400 rounded-lg p-6 cursor-pointer hover:border-blue-400 transition">
          <p className="text-sm text-gray-600">
            📷 Tap or click to take a photo / upload
          </p>
        </div>
      </label>

      {error && <p className="text-red-600 text-sm">{error}</p>}

      {preview && (
        <div className="mt-3">
          <img
            src={preview}
            alt="preview"
            className="max-h-64 mx-auto rounded-md shadow"
          />
          <p className="text-xs text-gray-500 mt-1">Preview</p>
        </div>
      )}
    </div>
  );
}
