const TARGET_WIDTH = 330;
const TARGET_HEIGHT = 50;
const CHANNEL_SIZE = TARGET_WIDTH * TARGET_HEIGHT;

const SCALE_MIN = 1.5;
const SCALE_MAX = 3;
const TARGET_MAX_WIDTH = 900;

export async function imageToTensor(file: File): Promise<Float32Array> {
  const img = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  canvas.width = TARGET_WIDTH;
  canvas.height = TARGET_HEIGHT;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const { data } = imageData;

  const tensor = new Float32Array(CHANNEL_SIZE);
  for (let i = 0; i < CHANNEL_SIZE; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = (r + g + b) / 3 / 255; // normalize to [0,1]
    tensor[i] = gray;
  }
  return tensor;
}

export async function preprocessImage(file: File): Promise<Float32Array> {
  const img = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(TARGET_WIDTH, TARGET_HEIGHT);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);

  const imageData = ctx.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
  const { data } = imageData;
  const gray = new Float32Array(CHANNEL_SIZE);
  for (let y = 0; y < TARGET_HEIGHT; y++) {
    for (let x = 0; x < TARGET_WIDTH; x++) {
      const idx = (y * TARGET_WIDTH + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const value = (r + g + b) / 3 / 255.0;
      gray[y * TARGET_WIDTH + x] = value * 2 - 1;
    }
  }
  return gray;
}

export async function prepareTesseractImage(file: File): Promise<Blob> {
  if (typeof document === "undefined") {
    return file;
  }
  const bitmap = await createImageBitmap(file);
  const scaleBase = Math.min(TARGET_MAX_WIDTH / Math.max(bitmap.width, 1), SCALE_MAX);
  const scale = Math.max(SCALE_MIN, scaleBase);
  const width = Math.max(Math.round(bitmap.width * scale), 1);
  const height = Math.max(Math.round(bitmap.height * scale), 1);
  const margin = Math.round(height * 0.05);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const pixels = data.length / 4;
  let sum = 0;
  let sumSq = 0;

  for (let i = 0; i < data.length; i += 4) {
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    sum += gray;
    sumSq += gray * gray;
    data[i] = data[i + 1] = data[i + 2] = gray;
  }

  const mean = sum / pixels;
  const variance = Math.max(sumSq / pixels - mean * mean, 0);
  const std = Math.sqrt(variance);
  let threshold = Math.max(0, Math.min(255, mean - std * 1.2));
  console.log(`Binarization threshold: ${threshold.toFixed(2)}`);

  for (let i = 0; i < data.length; i += 4) {
    const value = data[i] < threshold ? 0 : 255;
    data[i] = data[i + 1] = data[i + 2] = value;
    data[i + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);

  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = width;
  outputCanvas.height = height + margin * 2;
  const outputCtx = outputCanvas.getContext("2d")!;
  outputCtx.fillStyle = "#ffffff";
  outputCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
  outputCtx.drawImage(canvas, 0, margin);

  bitmap.close?.();

  return new Promise((resolve) => {
    outputCanvas.toBlob((blob) => resolve(blob ?? new Blob()), "image/png", 1);
  });
}
