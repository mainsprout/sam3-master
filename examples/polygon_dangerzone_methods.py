"""Shared helpers for the polygon danger-zone experiment notebooks."""

import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
from sam3.visualization_utils import load_frame, render_masklet_frame


def load_video_frames(video_path):
    video_path = str(video_path)
    if video_path.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    frames = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        frames.sort(key=lambda path: int(Path(path).stem))
    except ValueError:
        frames.sort()
    return frames


def get_video_fps(video_path, default_fps=10.0):
    video_path = str(video_path)
    if video_path.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return float(fps)
    return float(default_fps)


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def select_polygon_on_frame(frame_rgb, max_display_width=1280):
    if frame_rgb is None or frame_rgb.size == 0:
        raise ValueError("The polygon editor received an empty frame.")

    height, width = frame_rgb.shape[:2]
    scale = min(1.0, max_display_width / width)
    display_size = (int(width * scale), int(height * scale))
    display_base = cv2.resize(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        display_size,
        interpolation=cv2.INTER_AREA,
    )
    points = []
    window_name = "Draw danger zone - Enter: save, R: reset, Esc: cancel"

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *display_size)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            canvas = display_base.copy()
            if len(points) >= 2:
                cv2.polylines(
                    canvas,
                    [np.asarray(points, dtype=np.int32)],
                    len(points) >= 3,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            for index, point in enumerate(points):
                cv2.circle(canvas, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    str(index + 1),
                    (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                canvas,
                "Left:add  Right/Backspace:undo  Enter/S:save  R:reset  Esc:cancel",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10, ord("s")):
                if len(points) < 3:
                    print("A polygon needs at least three points.")
                    continue
                if abs(cv2.contourArea(np.asarray(points, dtype=np.float32))) < 10:
                    print("The polygon is too small.")
                    continue
                break
            if key in (8, 127) and points:
                points.pop()
            elif key == ord("r"):
                points.clear()
            elif key == 27:
                raise KeyboardInterrupt("Polygon drawing was cancelled.")
    finally:
        cv2.destroyWindow(window_name)

    display_width, display_height = display_size
    return [[x / display_width, y / display_height] for x, y in points]


def save_zone_config(config_path, normalized_points, source_video):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 1,
        "camera_mode": "fixed",
        "source_video": str(Path(source_video).resolve()),
        "points_normalized": normalized_points,
    }
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved danger-zone config: {config_path.resolve()}")


def load_zone_config(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    points = config["points_normalized"]
    if len(points) < 3:
        raise ValueError("The saved danger zone has fewer than three points.")
    return points


def normalized_polygon_to_pixels(normalized_points, width, height):
    polygon = np.asarray(
        [[round(x * width), round(y * height)] for x, y in normalized_points],
        dtype=np.int32,
    )
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return polygon


def preview_polygon(frame_rgb, normalized_polygon, alpha=0.2):
    frame_rgb = frame_rgb.copy()
    height, width = frame_rgb.shape[:2]
    polygon = normalized_polygon_to_pixels(normalized_polygon, width, height)
    overlay = frame_rgb.copy()
    cv2.fillPoly(overlay, [polygon], (255, 215, 0))
    result = cv2.addWeighted(overlay, alpha, frame_rgb, 1.0 - alpha, 0)
    cv2.polylines(result, [polygon], True, (255, 215, 0), 4, cv2.LINE_AA)
    return result


def _mask_as_bool(mask):
    return np.squeeze(mask).astype(bool)


def _bottom_center(component_x, component_y, bottom_band):
    bottom_y = int(component_y.max())
    band_x = component_x[component_y >= bottom_y - (bottom_band - 1)]
    if len(band_x) == 0:
        return None
    return int(np.median(band_x)), bottom_y


def extract_support_candidates(
    mask,
    lower_ratio=0.30,
    bottom_band=5,
    min_component_pixels=12,
    split_width_ratio=0.28,
):
    """Return one or two image-space support candidates from a person mask.

    These are geometric left/right candidates, not anatomical foot labels.
    Separate lower-mask components are preferred. A single wide component is
    split around its horizontal median; a narrow component remains one point.
    """
    mask = _mask_as_bool(mask)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    person_height = y_max - y_min + 1
    person_width = x_max - x_min + 1
    lower_start = max(y_min, int(y_max - person_height * lower_ratio))

    lower_mask = np.zeros_like(mask, dtype=np.uint8)
    lower_mask[lower_start : y_max + 1] = mask[lower_start : y_max + 1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        lower_mask, connectivity=8
    )

    components = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_component_pixels:
            continue
        component_y, component_x = np.where(labels == label_id)
        components.append((component_x, component_y, area))

    if not components:
        return []

    if len(components) >= 2:
        components.sort(
            key=lambda item: (int(item[1].max()), item[2]), reverse=True
        )
        points = [
            _bottom_center(component_x, component_y, bottom_band)
            for component_x, component_y, _ in components[:2]
        ]
        return sorted((point for point in points if point), key=lambda p: p[0])

    component_x, component_y, _ = components[0]
    component_width = int(component_x.max() - component_x.min() + 1)
    split_x = float(np.median(component_x))
    left = component_x < split_x
    right = component_x >= split_x
    enough_on_both_sides = (
        int(left.sum()) >= min_component_pixels
        and int(right.sum()) >= min_component_pixels
    )
    looks_like_joined_supports = (
        component_width >= person_width * split_width_ratio and enough_on_both_sides
    )

    if looks_like_joined_supports:
        points = [
            _bottom_center(component_x[left], component_y[left], bottom_band),
            _bottom_center(component_x[right], component_y[right], bottom_band),
        ]
        return [point for point in points if point]

    point = _bottom_center(component_x, component_y, bottom_band)
    return [point] if point else []


def extract_contact_region(mask, lower_ratio=0.35, contact_band=5):
    """Return a thin mask following the lower envelope of the person mask."""
    mask = _mask_as_bool(mask)
    ys, xs = np.where(mask)
    contact = np.zeros_like(mask, dtype=bool)
    if len(xs) == 0:
        return contact

    y_min, y_max = int(ys.min()), int(ys.max())
    person_height = y_max - y_min + 1
    lower_start = max(y_min, int(y_max - person_height * lower_ratio))

    for x in np.unique(xs):
        column_y = np.where(mask[:, x])[0]
        if len(column_y) == 0:
            continue
        bottom_y = int(column_y.max())
        if bottom_y < lower_start:
            continue
        band_start = max(lower_start, bottom_y - (contact_band - 1))
        contact[band_start : bottom_y + 1, x] = mask[
            band_start : bottom_y + 1, x
        ]
    return contact


def _create_writer(output_path, fps, width, height):
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output_path}")
    return writer


def _new_state():
    return {
        "inside": False,
        "inside_frames": 0,
        "outside_frames": 0,
    }


def _update_state(
    state,
    evidence,
    frame_idx,
    fps,
    object_id,
    confirm_frames,
    exit_confirm_frames,
    events,
):
    entered = False
    if evidence == "inside":
        state["inside_frames"] += 1
        state["outside_frames"] = 0
        if not state["inside"] and state["inside_frames"] >= confirm_frames:
            state["inside"] = True
            entered = True
    elif evidence == "outside":
        state["outside_frames"] += 1
        state["inside_frames"] = 0
        if state["outside_frames"] >= exit_confirm_frames:
            state["inside"] = False
    else:
        state["inside_frames"] = 0
        state["outside_frames"] = 0

    if entered:
        event = {
            "frame_idx": frame_idx,
            "time_seconds": frame_idx / fps,
            "object_id": object_id,
        }
        events.append(event)
        print(
            f"[DANGER ZONE ENTRY] frame={frame_idx}, "
            f"time={event['time_seconds']:.2f}s, person_id={object_id}"
        )


def _draw_zone(frame, polygon, active_danger, zone_alpha):
    color = (255, 0, 0) if active_danger else (255, 215, 0)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    frame = cv2.addWeighted(overlay, zone_alpha, frame, 1.0 - zone_alpha, 0)
    cv2.polylines(frame, [polygon], True, color, 4, cv2.LINE_AA)
    label_point = polygon[np.argmin(polygon[:, 1])]
    cv2.putText(
        frame,
        "DANGER ZONE",
        (int(label_point[0]), max(35, int(label_point[1]) - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def _draw_status_banner(frame, danger_ids, warning_ids):
    height, width = frame.shape[:2]
    if danger_ids:
        text = "DANGER! PERSON " + ", ".join(map(str, sorted(danger_ids)))
        color = (255, 0, 0)
    elif warning_ids:
        text = "UNCERTAIN / BOUNDARY: PERSON " + ", ".join(
            map(str, sorted(warning_ids))
        )
        color = (255, 165, 0)
    else:
        return frame

    cv2.rectangle(frame, (15, 15), (width - 15, 75), color, -1)
    cv2.putText(
        frame,
        text,
        (30, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def export_with_support_candidates(
    video_frames,
    outputs_per_frame,
    normalized_polygon,
    output_path,
    fps,
    lower_ratio=0.30,
    bottom_band=5,
    min_component_pixels=12,
    split_width_ratio=0.28,
    zone_margin=8.0,
    confirm_frames=3,
    exit_confirm_frames=3,
    mask_alpha=0.5,
    zone_alpha=0.18,
):
    """Export using one or two independently tested lower-mask points."""
    if not video_frames:
        raise ValueError("No video frames were loaded.")
    first_frame = load_frame(video_frames[0])
    height, width = first_frame.shape[:2]
    polygon = normalized_polygon_to_pixels(normalized_polygon, width, height)
    contour = polygon.astype(np.float32).reshape((-1, 1, 2))
    states = {}
    events = []
    writer = _create_writer(output_path, fps, width, height)

    try:
        for frame_idx in sorted(outputs_per_frame):
            frame_output = outputs_per_frame[frame_idx]
            rendered = render_masklet_frame(
                load_frame(video_frames[frame_idx]),
                frame_output,
                frame_idx=frame_idx,
                alpha=mask_alpha,
            )
            danger_ids = []
            warning_ids = []

            for object_id, mask in zip(
                frame_output["out_obj_ids"], frame_output["out_binary_masks"]
            ):
                object_id = int(object_id)
                points = extract_support_candidates(
                    mask,
                    lower_ratio=lower_ratio,
                    bottom_band=bottom_band,
                    min_component_pixels=min_component_pixels,
                    split_width_ratio=split_width_ratio,
                )
                distances = [
                    cv2.pointPolygonTest(contour, point, True) for point in points
                ]

                if not points:
                    evidence = "uncertain"
                elif any(distance > zone_margin for distance in distances):
                    evidence = "inside"
                elif all(distance < -zone_margin for distance in distances):
                    evidence = "outside"
                else:
                    evidence = "boundary"

                state = states.setdefault(object_id, _new_state())
                _update_state(
                    state,
                    evidence,
                    frame_idx,
                    fps,
                    object_id,
                    confirm_frames,
                    exit_confirm_frames,
                    events,
                )

                for candidate_index, point in enumerate(points, start=1):
                    inside = distances[candidate_index - 1] > zone_margin
                    color = (255, 0, 0) if inside else (0, 255, 0)
                    cv2.circle(rendered, point, 8, color, -1, cv2.LINE_AA)
                    cv2.putText(
                        rendered,
                        f"C{candidate_index}",
                        (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                if state["inside"]:
                    danger_ids.append(object_id)
                elif evidence in ("uncertain", "boundary"):
                    warning_ids.append(object_id)

            rendered = _draw_zone(rendered, polygon, bool(danger_ids), zone_alpha)
            rendered = _draw_status_banner(rendered, danger_ids, warning_ids)
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print(f"Saved support-candidate video to: {Path(output_path).resolve()}")
    print(f"Detected {len(events)} danger-zone entries.")
    return events


def export_with_contact_area(
    video_frames,
    outputs_per_frame,
    normalized_polygon,
    output_path,
    fps,
    lower_ratio=0.35,
    contact_band=5,
    min_overlap_pixels=8,
    min_overlap_ratio=0.03,
    warning_radius=12,
    confirm_frames=3,
    exit_confirm_frames=3,
    mask_alpha=0.5,
    zone_alpha=0.18,
):
    """Export using overlap between the contact-region mask and danger zone."""
    if not video_frames:
        raise ValueError("No video frames were loaded.")
    first_frame = load_frame(video_frames[0])
    height, width = first_frame.shape[:2]
    polygon = normalized_polygon_to_pixels(normalized_polygon, width, height)
    zone_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(zone_mask, [polygon], 1)
    kernel_size = max(1, warning_radius * 2 + 1)
    warning_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    warning_mask = cv2.dilate(zone_mask, warning_kernel).astype(bool)
    zone_mask = zone_mask.astype(bool)
    states = {}
    events = []
    writer = _create_writer(output_path, fps, width, height)

    try:
        for frame_idx in sorted(outputs_per_frame):
            frame_output = outputs_per_frame[frame_idx]
            rendered = render_masklet_frame(
                load_frame(video_frames[frame_idx]),
                frame_output,
                frame_idx=frame_idx,
                alpha=mask_alpha,
            )
            danger_ids = []
            warning_ids = []

            for object_id, mask in zip(
                frame_output["out_obj_ids"], frame_output["out_binary_masks"]
            ):
                object_id = int(object_id)
                contact = extract_contact_region(
                    mask, lower_ratio=lower_ratio, contact_band=contact_band
                )
                contact_pixels = int(np.count_nonzero(contact))
                overlap = contact & zone_mask
                overlap_pixels = int(np.count_nonzero(overlap))
                overlap_ratio = (
                    overlap_pixels / contact_pixels if contact_pixels else 0.0
                )
                near_zone = bool(np.any(contact & warning_mask))

                if contact_pixels == 0:
                    evidence = "uncertain"
                elif (
                    overlap_pixels >= min_overlap_pixels
                    and overlap_ratio >= min_overlap_ratio
                ):
                    evidence = "inside"
                elif near_zone:
                    evidence = "boundary"
                else:
                    evidence = "outside"

                state = states.setdefault(object_id, _new_state())
                _update_state(
                    state,
                    evidence,
                    frame_idx,
                    fps,
                    object_id,
                    confirm_frames,
                    exit_confirm_frames,
                    events,
                )

                rendered[contact] = (
                    0.45 * rendered[contact] + 0.55 * np.array([0, 255, 255])
                ).astype(np.uint8)
                rendered[overlap] = np.array([255, 0, 0], dtype=np.uint8)

                ys, xs = np.where(contact)
                if len(xs):
                    label_point = (int(np.median(xs)), int(ys.min()))
                    cv2.putText(
                        rendered,
                        f"overlap={overlap_ratio:.2f}",
                        label_point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if state["inside"]:
                    danger_ids.append(object_id)
                elif evidence in ("uncertain", "boundary"):
                    warning_ids.append(object_id)

            rendered = _draw_zone(rendered, polygon, bool(danger_ids), zone_alpha)
            rendered = _draw_status_banner(rendered, danger_ids, warning_ids)
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print(f"Saved contact-area video to: {Path(output_path).resolve()}")
    print(f"Detected {len(events)} danger-zone entries.")
    return events
