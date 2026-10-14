"""Evaluate a SAM3 MOTChallenge result against MOT17 ground truth."""

from pathlib import Path
import argparse
import csv

import motmetrics as mm


def read_mot_file(path: Path, ground_truth: bool):
    """Read MOT rows and keep pedestrian annotations only."""
    by_frame = {}
    with path.open(newline="") as file:
        for row in csv.reader(file):
            if len(row) < 7:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            x, y, width, height = map(float, row[2:6])

            # In MOT17 GT, class 1 is pedestrian and confidence 1 is valid GT.
            if ground_truth:
                confidence = int(float(row[6]))
                class_id = int(float(row[7])) if len(row) > 7 else 1
                if class_id != 1 or confidence != 1:
                    continue

            by_frame.setdefault(frame, {})[track_id] = (x, y, width, height)
    return by_frame


def iou_distance_matrix(gt_boxes, pred_boxes):
    """Return 1-IoU distances; values above the threshold are rejected."""
    import numpy as np

    if not gt_boxes or not pred_boxes:
        return np.empty((len(gt_boxes), len(pred_boxes)))

    matrix = np.ones((len(gt_boxes), len(pred_boxes)), dtype=float)
    for i, gt in enumerate(gt_boxes):
        gx, gy, gw, gh = gt
        for j, pred in enumerate(pred_boxes):
            px, py, pw, ph = pred
            ix1, iy1 = max(gx, px), max(gy, py)
            ix2, iy2 = min(gx + gw, px + pw), min(gy + gh, py + ph)
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = gw * gh + pw * ph - intersection
            iou = intersection / union if union > 0 else 0.0
            matrix[i, j] = 1.0 - iou
    return matrix


def evaluate(gt_file: Path, result_file: Path, iou_threshold: float):
    # motmetrics uses frame-wise assignments to calculate CLEAR MOT and ID metrics.
    gt = read_mot_file(gt_file, ground_truth=True)
    pred = read_mot_file(result_file, ground_truth=False)
    accumulator = mm.MOTAccumulator(auto_id=False)

    for frame in sorted(set(gt) | set(pred)):
        gt_ids = list(gt.get(frame, {}))
        pred_ids = list(pred.get(frame, {}))
        distances = iou_distance_matrix(
            [gt[frame][track_id] for track_id in gt_ids],
            [pred[frame][track_id] for track_id in pred_ids],
        )
        # Distances above 1-IoU threshold are treated as no match.
        distances[distances > 1.0 - iou_threshold] = float("nan")
        accumulator.update(gt_ids, pred_ids, distances)

    metrics = [
        "num_frames", "num_objects", "num_predictions", "num_matches",
        "num_misses", "num_false_positives", "num_switches",
        "mota", "motp", "idf1", "idp", "idr",
    ]
    summary = mm.metrics.create().compute(accumulator, metrics=metrics, name="SAM3")
    print(summary.to_string(float_format="%.4f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.gt, args.result, args.iou_threshold)
