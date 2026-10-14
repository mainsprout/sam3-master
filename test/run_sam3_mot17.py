"""Run SAM3 person tracking on one MOT17 sequence and save MOTChallenge output."""

from pathlib import Path
import argparse

import numpy as np
from sam3.model_builder import build_sam3_predictor


def to_numpy(value):
    """Convert torch/numpy/list values to a numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def run(sequence_dir: Path, output_file: Path, prompt: str, score_threshold: float):
    # MOT17 image sequence is passed directly to SAM3 as a JPEG directory.
    image_dir = sequence_dir / "img1"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    # Build the predictor. SAM3 may download/load its checkpoint on first run.
    predictor = build_sam3_predictor(version="sam3")
    session = predictor.handle_request(
        {"type": "start_session", "resource_path": str(image_dir)}
    )
    session_id = session["session_id"]

    # A text prompt on the first frame asks SAM3 to find and track all persons.
    predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        }
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        # Propagate the prompted objects through every frame.
        for output in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            frame_index = int(output["frame_index"])
            frame_number = frame_index + 1  # MOTChallenge frame numbers start at 1.

            boxes = to_numpy(output.get("out_boxes_xywh", []))
            scores = to_numpy(output.get("out_probs", []))
            object_ids = to_numpy(output.get("out_obj_ids", []))

            if boxes.size == 0:
                continue
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            object_ids = object_ids.reshape(-1)

            # Write: frame, id, x, y, w, h, score, class, visibility.
            for box, score, object_id in zip(boxes, scores, object_ids):
                score = float(score)
                if score < score_threshold:
                    continue
                x, y, width, height = map(float, box)
                rows.append(
                    f"{frame_number},{int(object_id)},{x:.3f},{y:.3f},"
                    f"{width:.3f},{height:.3f},{score:.6f},-1,-1,-1"
                )
    finally:
        # Release the predictor session even when inference raises an error.
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    output_file.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"Saved {len(rows)} tracking rows to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    args = parser.parse_args()
    run(args.sequence_dir, args.output, args.prompt, args.score_threshold)
