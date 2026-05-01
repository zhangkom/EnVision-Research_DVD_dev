import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESETS = (
    "balanced",
    "throughput",
    "realtime",
    "realtime-preview",
    "speed-floor",
)


def parse_metrics(stdout):
    marker = '{\n  "timestamp"'
    start = stdout.rfind(marker)
    if start < 0:
        raise ValueError("Cannot find benchmark JSON in stdout.")
    return json.loads(stdout[start:])


def run_sample(args, preset):
    run_name = f"stage1_{preset.replace('-', '_')}"
    cmd = [
        args.python,
        str(REPO_ROOT / "tools" / "benchmark_single_video.py"),
        "--weights",
        args.weights,
        "--local_model_path",
        args.local_model_path,
        "--input_video",
        args.input_video,
        "--output_dir",
        args.output_dir,
        "--preset",
        preset,
        "--dtype",
        args.dtype,
        "--target_fps",
        str(args.target_fps),
        "--run_name",
        run_name,
        "--decode_resize",
        "--fast_video_save",
        "--save_depth_npy",
        "--save_depth_png16",
    ]
    if args.max_frames is not None:
        cmd.extend(["--max_frames", str(args.max_frames)])
    if args.no_resize_back:
        cmd.append("--no_resize_back")

    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)

    metrics = parse_metrics(result.stdout)
    print(
        (
            f"{preset}: e2e {metrics['end_to_end_fps_excluding_model_load']:.2f} FPS, "
            f"depth {metrics['output_depth_height']}x{metrics['output_depth_width']}"
        ),
        flush=True,
    )
    return metrics


def write_summary(results, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    json_path = output_path / f"stage1_quality_samples_{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Stage 1 Quality Samples",
        "",
        "| preset | frames | resized | output depth | e2e FPS | video | npy | png16 |",
        "| --- | ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for item in results:
        resized = f"{item['resized_height']}x{item['resized_width']}"
        output_depth = f"{item['output_depth_height']}x{item['output_depth_width']}"
        lines.append(
            (
                f"| {item.get('preset') or '-'} | {item['bench_frames']} | "
                f"{resized} | {output_depth} | "
                f"{item['end_to_end_fps_excluding_model_load']:.2f} | "
                f"{item.get('output_path', '')} | "
                f"{item.get('depth_npy_path', '')} | "
                f"{item.get('depth_png16_dir', '')} |"
            )
        )

    md_path = output_path / f"stage1_quality_samples_{timestamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stage-1 quality samples.")
    parser.add_argument("--weights", default="ckpt/model.safetensors")
    parser.add_argument("--local_model_path", default="models")
    parser.add_argument("--input_video", default="test_video/depth_full_50frame.mp4")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--target_fps", type=float, default=25.0)
    parser.add_argument("--max_frames", type=int, default=121)
    parser.add_argument("--presets", nargs="+", default=list(DEFAULT_PRESETS))
    parser.add_argument("--no_resize_back", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    results = [run_sample(args, preset) for preset in args.presets]
    json_path, md_path = write_summary(results, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
