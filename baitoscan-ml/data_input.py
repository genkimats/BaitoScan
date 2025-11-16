import argparse
import sys
from pathlib import Path
from typing import List, Optional

#!/usr/bin/env python3


IMAGE_EXTS = {".png"}


def default_images_dir() -> Path:
    # This file is in .../BaitoScan/baitoscan-ml/data_input.py
    # Images are expected under .../BaitoScan/data/train_real
    return Path(__file__).resolve().parent.parent / "data" / "train_real"


def find_images(dir_path: Path) -> List[Path]:
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    files = [p for p in sorted(dir_path.iterdir()) if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return files


def show_image(img_path: Path) -> None:
    """Show the image using OpenCV's imshow in a non-blocking way.

    If cv2 is not installed, fall back to printing the path and an install hint.
    """
    try:
        import cv2
    except Exception:
        print("OpenCV (cv2) is not installed. Install with: pip install opencv-python", file=sys.stderr)
        print(f"Image path: {img_path}")
        return

    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("cv2.imread returned None (failed to read image)")

        # If image has alpha channel, convert to BGR for display
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        window_name = img_path.name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, img)
        # Allow OpenCV GUI to process events and render window; non-blocking
        cv2.waitKey(1)
    except Exception as e:
        print(f"Failed to open image {img_path}: {e}", file=sys.stderr)


def read_multiline_input(prompt: str) -> Optional[str]:
    """
    Reads multi-line content. End input with an empty line on its own.
    Commands:
      :q  -> quit program
      :s  -> skip this image (returns empty string)
      :r  -> reopen image (returns None so caller can re-show and re-prompt)
    Returns:
      - None to request re-show
      - "" to skip
      - text content otherwise
    """
    print(prompt)
    print("Finish with an empty line. Commands: :q quit, :s skip, :r reopen image")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            # Treat EOF as quit
            return ":q"
        if line == "":
            break
        if line.strip() in {":q", ":s", ":r"}:
            return line.strip()
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Annotate images by typing their content into the terminal.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=default_images_dir(),
        help="Directory containing images (default: data/train_real relative to repo root).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files.",
    )
    args = parser.parse_args()

    img_dir: Path = args.dir
    overwrite: bool = args.overwrite

    images = find_images(img_dir)
    if not images:
        print(f"No images found in {img_dir}")
        return

    total = len(images)
    print(f"Found {total} images in {img_dir}")

    for idx, img_path in enumerate(images, start=1):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            print(f"[{idx}/{total}] {img_path.name} -> txt exists, skipping (use --overwrite to force).")
            continue

        print(f"[{idx}/{total}] Showing {img_path.name}")
        show_image(img_path)

        while True:
            res = read_multiline_input("Enter content for this image:")
            if res == ":q":
                print("Quitting.")
                return
            if res == ":s":
                print("Skipped.")
                break
            if res == ":r":
                print("Reopening image...")
                show_image(img_path)
                continue
            content = res if isinstance(res, str) else ""
            if content.strip() == "":
                print("Empty content. Skipped.")
                break

            try:
                txt_path.write_text(content, encoding="utf-8")
                print(f"Wrote {txt_path.name}")
            except Exception as e:
                print(f"Failed to write {txt_path}: {e}", file=sys.stderr)
            break

    # Close any OpenCV windows if available
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    main()