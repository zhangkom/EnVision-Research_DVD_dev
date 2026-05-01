import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_frame_indices(value):
    if not value:
        return []
    indices = []
    for part in value.replace(",", " ").split():
        indices.append(int(part))
    return indices


def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = frame_index + 1
    safe_index = max(0, min(frame_index, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame {safe_index} from {video_path}")
    return frame, safe_index


def fit_frame(frame, width, height):
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def put_text(img, text, origin, scale=0.55, color=(235, 235, 235), thickness=1):
    cv2.putText(
        img,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def label_cell(label, width, height):
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    lines = [label[i : i + 22] for i in range(0, len(label), 22)] or [label]
    y = 28
    for line in lines[:4]:
        put_text(canvas, line, (10, y), scale=0.52, color=(245, 245, 245))
        y += 24
    return canvas


def header_cell(text, width, height):
    canvas = np.full((height, width, 3), 36, dtype=np.uint8)
    put_text(canvas, text, (10, max(24, height // 2 + 6)), scale=0.6)
    return canvas


def build_contact_sheet(videos, labels, frame_indices, cell_width, cell_height):
    label_width = 180
    header_height = 44

    rows = []
    header = [header_cell("source/preset", label_width, header_height)]
    for idx in frame_indices:
        header.append(header_cell(f"frame {idx}", cell_width, header_height))
    rows.append(np.concatenate(header, axis=1))

    for video_path, label in zip(videos, labels):
        cells = [label_cell(label, label_width, cell_height)]
        for requested_idx in frame_indices:
            frame, actual_idx = read_frame(video_path, requested_idx)
            cell = fit_frame(frame, cell_width, cell_height)
            if actual_idx != requested_idx:
                put_text(cell, f"actual {actual_idx}", (8, cell_height - 10), scale=0.5)
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))

    return np.concatenate(rows, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a stage-1 visual contact sheet from source/depth videos."
    )
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--frame_indices", default="0 30 60 90 120")
    parser.add_argument("--cell_width", type=int, default=360)
    parser.add_argument("--cell_height", type=int, default=112)
    parser.add_argument("--output", default="output/stage1_contact_sheet.png")
    return parser.parse_args()


def main():
    args = parse_args()
    videos = [Path(item) for item in args.videos]
    labels = args.labels or [path.stem for path in videos]
    if len(labels) != len(videos):
        raise ValueError("--labels must have the same number of entries as --videos")
    frame_indices = parse_frame_indices(args.frame_indices)
    if not frame_indices:
        raise ValueError("--frame_indices cannot be empty")

    sheet = build_contact_sheet(
        videos, labels, frame_indices, args.cell_width, args.cell_height
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    print(output)


if __name__ == "__main__":
    main()
