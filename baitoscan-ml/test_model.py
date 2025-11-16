"""Quick script to compare CRNN predictions against ground-truth labels."""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from train_crnn import CRNN, CHARS


CHECKPOINT = Path("checkpoints/baitoscan_crnn_1.pth")
IMAGES_DIR = Path("data/train_real")
TARGET_WIDTH = 300
TARGET_HEIGHT = 50
BLANK_INDEX = len(CHARS)


def load_image(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def read_label(path: Path) -> str:
    label_path = path.with_suffix(".txt")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    return label_path.read_text(encoding="utf-8").strip()


def ctc_greedy_decode(indices: Iterable[int]) -> str:
    text = []
    prev = -1
    for idx in indices:
        if idx == BLANK_INDEX:
            prev = idx
            continue
        if idx == prev:
            continue
        text.append(CHARS[idx])
        prev = idx
    return "".join(text)


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model = CRNN(num_classes=len(CHARS))
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found under {IMAGES_DIR}")

    with torch.no_grad():
        for img_path in image_paths:
            tensor = load_image(img_path)
            logits = model(tensor).squeeze(0)  # [T, num_classes + 1]
            pred_indices = logits.argmax(dim=-1).cpu().numpy()
            predicted = ctc_greedy_decode(pred_indices)
            target = read_label(img_path)

            match = predicted == target
            print(f"{img_path.name}: predicted='{predicted}' | expected='{target}' | match={match}")


if __name__ == "__main__":
    main()
