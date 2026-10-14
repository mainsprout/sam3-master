"""Overlay MOT17 ground truth and tracker boxes on selected frames."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_boxes(path, frames, ground_truth=False):
    boxes = {frame: [] for frame in frames}
    with path.open(newline="") as file:
        for row in csv.reader(file):
            if len(row) < 7:
                continue
            frame = int(float(row[0]))
            if frame not in boxes:
                continue
            if ground_truth and (int(float(row[6])) != 1 or
                                 (len(row) > 7 and int(float(row[7])) != 1)):
                continue
            track_id = int(float(row[1]))
            x, y, width, height = map(float, row[2:6])
            boxes[frame].append((track_id, x, y, width, height))
    return boxes


def draw_boxes(draw, boxes, color, label):
    for track_id, x, y, width, height in boxes:
        draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
        draw.text((x, max(0, y - 14)), f"{label} {track_id}", fill=color,
                  stroke_width=2, stroke_fill="black")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("test/results/overlays"))
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 10, 25, 50])
    args = parser.parse_args()

    frames = set(args.frames)
    gt = read_boxes(args.gt, frames, ground_truth=True)
    pred = read_boxes(args.result, frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for frame in args.frames:
        image_path = args.sequence_dir / "img1" / f"{frame:06d}.jpg"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        draw_boxes(draw, gt[frame], "lime", "GT")
        draw_boxes(draw, pred[frame], "red", "SAM3")
        output_path = args.output_dir / f"frame-{frame:06d}.jpg"
        image.save(output_path, quality=95)
        print(f"frame {frame}: GT={len(gt[frame])}, SAM3={len(pred[frame])} -> {output_path}")


if __name__ == "__main__":
    main()
