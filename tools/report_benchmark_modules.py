import argparse
import json
from pathlib import Path


MODULE_KEYS = [
    ("model_load_s", "model load"),
    ("video_decode_s", "video decode"),
    ("video_resize_s", "video resize"),
    ("video_read_resize_s", "video read/resize"),
    ("inference_s", "inference total"),
    ("resize_back_s", "resize depth back"),
    ("save_s", "save visualization"),
    ("save_depth_npy_s", "save depth npy"),
    ("save_depth_png16_s", "save depth png16"),
]


def load_metrics(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def module_rows(metrics, include_model_load):
    total = metrics["total_s"] if include_model_load else metrics["total_s"] - metrics[
        "model_load_s"
    ]
    rows = []
    seen_video_split = "video_decode_s" in metrics or "video_resize_s" in metrics
    for key, label in MODULE_KEYS:
        if key == "model_load_s" and not include_model_load:
            continue
        if key == "video_read_resize_s" and seen_video_split:
            continue
        if key not in metrics:
            continue
        seconds = float(metrics.get(key) or 0.0)
        if seconds == 0:
            continue
        rows.append((label, seconds, seconds / total * 100.0))

    profile = metrics.get("inference_profile") or {}
    if profile:
        inference_total = float(metrics["inference_s"])
        rows.append(
            (
                "  inference: model.pipe windows (of inference)",
                float(profile.get("model_pipe_s", 0.0)),
                float(profile.get("model_pipe_s", 0.0)) / inference_total * 100.0,
            )
        )
        rows.append(
            (
                "  inference: overlap alignment (of inference)",
                float(profile.get("overlap_alignment_s", 0.0)),
                float(profile.get("overlap_alignment_s", 0.0))
                / inference_total
                * 100.0,
            )
        )
        rows.append(
            (
                "  inference: window prepare (of inference)",
                float(profile.get("window_prepare_s", 0.0)),
                float(profile.get("window_prepare_s", 0.0)) / inference_total * 100.0,
            )
        )
    return rows


def summarize_one(path, include_model_load):
    metrics = load_metrics(path)
    rows = module_rows(metrics, include_model_load)
    title = (
        f"{Path(path).name}: {metrics.get('resized_height')}x"
        f"{metrics.get('resized_width')}, {metrics.get('bench_frames')} frames"
    )
    lines = [
        f"## {title}",
        "",
        f"- preset: `{metrics.get('preset') or '-'}`",
        f"- decode_resize: `{metrics.get('decode_resize', False)}`",
        f"- total_s: `{metrics['total_s']:.3f}`",
        f"- end_to_end_fps_excluding_model_load: `{metrics.get('end_to_end_fps_excluding_model_load', 0):.2f}`",
        "",
        "| module | seconds | percent |",
        "| --- | ---: | ---: |",
    ]
    for label, seconds, percent in rows:
        lines.append(f"| {label} | {seconds:.3f} | {percent:.1f}% |")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Report DVD benchmark module timings.")
    parser.add_argument("benchmarks", nargs="+")
    parser.add_argument("--include_model_load", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    sections = [
        summarize_one(path, include_model_load=args.include_model_load)
        for path in args.benchmarks
    ]
    report = "# Benchmark Module Timing\n\n" + "\n\n".join(sections) + "\n"
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
