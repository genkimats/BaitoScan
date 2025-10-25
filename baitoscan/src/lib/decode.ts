const CHARS = "0123456789/:~ "; // must match training charset

export function greedyCTCDecode(logits: Float32Array, width: number, numClasses: number): string {
  // logits are flattened [batch=1, timesteps=width, classes=numClasses]
  const result: number[] = [];
  let prev = -1;
  for (let t = 0; t < width; t++) {
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const val = logits[t * numClasses + c];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = c;
      }
    }
    // skip blanks and duplicates
    if (maxIdx !== prev && maxIdx < CHARS.length) {
      result.push(maxIdx);
    }
    prev = maxIdx;
  }
  return result.map(i => CHARS[i]).join("").trim();
}
