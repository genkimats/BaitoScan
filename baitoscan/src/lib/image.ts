export async function imageToTensor(file: File): Promise<Float32Array> {
  const img = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  canvas.width = 256;
  canvas.height = 64;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  const tensor = new Float32Array(1 * 1 * 64 * 256);
  for (let i = 0; i < 64 * 256; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = (r + g + b) / 3 / 255; // normalize to [0,1]
    tensor[i] = gray;
  }
  return tensor;
}

export async function preprocessImage(file: File): Promise<Float32Array> {
  // 1. Read as image
  const img = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(256, 64);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, 256, 64);

  // 2. Get grayscale data
  const imageData = ctx.getImageData(0, 0, 256, 64);
  const { data } = imageData;
  const gray = new Float32Array(1 * 1 * 64 * 256);
  for (let i = 0; i < 64; i++) {
    for (let j = 0; j < 256; j++) {
      const idx = (i * 256 + j) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const value = (r + g + b) / 3 / 255.0; // [0,1]
      // normalize to [-1,1]
      gray[i * 256 + j] = value * 2 - 1;
    }
  }
  return gray;
}
