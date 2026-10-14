"""Run SAM3 person tracking on one MOT17 sequence and save MOTChallenge output."""

from pathlib import Path
import argparse
import sys

import numpy as np
from PIL import Image

# Allow direct execution via `python test/run_sam3_mot17.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def to_numpy(value):
    """Convert torch/numpy/list values to a numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def run(
    sequence_dir: Path,
    output_file: Path,
    prompt: str,
    score_threshold: float,
    offload_to_cpu: bool = False,
):
    # MOT17 image sequence is passed directly to SAM3 as a JPEG directory.
    image_dir = sequence_dir / "img1"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    image_paths = sorted(image_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not image_paths:
        raise ValueError(f"No JPG frames found: {image_dir}")
    frame_numbers = [int(path.stem) for path in image_paths]
    if len(set(frame_numbers)) != len(frame_numbers):
        raise ValueError("Duplicate frame numbers")
    with Image.open(image_paths[0]) as image:
        image_width, image_height = image.size
    print(f"Input: {len(image_paths)} frames, {image_width}x{image_height}, {image_dir}")
    from sam3.model_builder import build_sam3_predictor

    # Build the predictor. SAM3 may download/load its checkpoint on first run.
    # SAM3 base predictor does not accept an internal probability threshold.
    # We keep the threshold for filtering the exported MOT result below.
    predictor = build_sam3_predictor(
        version="sam3",
        async_loading_frames=False,
    )
    session = predictor.handle_request(
        {"type": "start_session", "resource_path": str(image_dir),
         "offload_video_to_cpu": offload_to_cpu,
         "offload_state_to_cpu": offload_to_cpu}
    )
    session_id = session["session_id"]

    # A text prompt on the first frame asks SAM3 to find and track all persons.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    processed_frames = set()
    raw_objects = 0
    try:
        prompt_response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            }
        )

        print("Prompt output keys:", list(prompt_response["outputs"]))
        # Propagate the prompted objects through every frame.
        for output in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id,
             "propagation_direction": "forward", "start_frame_index": 0}
        ):
            frame_index = int(output["frame_index"])
            frame_number = frame_numbers[frame_index]
            processed_frames.add(frame_index)

            # Predictor wraps the per-frame arrays in response["outputs"].
            objects = output["outputs"]
            boxes = to_numpy(objects["out_boxes_xywh"])
            scores = to_numpy(objects["out_probs"])
            object_ids = to_numpy(objects["out_obj_ids"])
            if frame_index == 0:
                print("Frame output keys:", list(objects))

            if boxes.size == 0:
                continue
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            object_ids = object_ids.reshape(-1)
            if not (len(boxes) == len(scores) == len(object_ids)):
                raise ValueError("SAM3 boxes, scores and IDs have different lengths")
            raw_objects += len(boxes)
            # SAM3 video boxes are normalized xywh; MOT17 requires pixel xywh.
            boxes = boxes * np.array([image_width, image_height, image_width, image_height])

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

    if len(processed_frames) != len(image_paths):
        raise RuntimeError(f"Only {len(processed_frames)}/{len(image_paths)} frames processed")
    print(f"Processed frames: {len(processed_frames)}, raw objects: {raw_objects}")
    if not rows:
        raise RuntimeError("No exported tracks. Check raw object count and score threshold; result was not written.")
    output_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} tracking rows to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--offload-to-cpu", action="store_true",
                        help="Store video/tracker state on CPU to reduce GPU memory use")
    args = parser.parse_args()
    run(args.sequence_dir, args.output, args.prompt, args.score_threshold, args.offload_to_cpu)
