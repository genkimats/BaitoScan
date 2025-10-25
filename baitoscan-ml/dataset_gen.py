import os
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from tqdm import tqdm

OUT_DIR = "data/train"
FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",  # playful
    "/System/Library/Fonts/Supplemental/Chalkboard.ttc",     # handwriting-like
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf"
]

FONTS = [f for f in FONT_PATHS if Path(f).exists()]
def random_time_line():
    m = random.randint(1, 12)
    d = random.randint(1, 31)
    sh = random.randint(6, 22)
    sm = random.choice(["00", "15", "30", "45"])
    eh = (sh + random.randint(4, 9)) % 24
    em = random.choice(["00", "15", "30", "45"])
    return f"{m}/{d} {sh}:{sm} ~ {eh}:{em}"

def render_text(text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    img = Image.new("L", (256, 64), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((5, 10), text, font=font, fill=0)
    return np.array(img)

def augment(img: np.ndarray) -> np.ndarray:
    rows, cols = img.shape
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    dx, dy = random.randint(-5, 5), random.randint(-3, 3)
    pts2 = np.float32([[0+dx,0+dy],[cols-dx,0+dy],[0+dx,rows-dy],[cols-dx,rows-dy]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (cols, rows), borderValue=255)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.randint(0, 15, img.shape, dtype='uint8')
    img = cv2.add(img, noise)
    return img

def generate_dataset(n=10000, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(n), desc="Generating dataset"):
        font = ImageFont.truetype(random.choice(FONTS), random.randint(26, 32))
        txt = random_time_line()
        img = render_text(txt, font)
        img = augment(img)
        cv2.imwrite(f"{out_dir}/{i:05d}.png", img)
        with open(f"{out_dir}/{i:05d}.txt", "w") as f:
            f.write(txt)
    print(f"✅ Generated {n} samples in {out_dir}")

if __name__ == "__main__":
    generate_dataset()
