import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESETS = ("balanced", "throughput", "realtime-preview", "speed-floor")


def parse_metrics(stdout):
    marker = '{\n  "timestamp"'
    start = stdout.rfind(marker)
    if start < 0:
        raise ValueError("Cannot find benchmark JSON in stdout.")
    return json.loads(stdout[start:])


def run_benchmark(args, preset):
    run_name = f"sweep_{preset.replace('-', '_')}"
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
    ]
    if args.max_frames is not None:
        cmd.extend(["--max_frames", str(args.max_frames)])
    if args.decode_resize:
        cmd.append("--decode_resize")
    if args.save_outputs:
        cmd.append("--fast_video_save")
    else:
        cmd.append("--no_save")

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
            f"{preset}: inference {metrics['inference_fps']:.2f} FPS, "
            f"e2e {metrics['end_to_end_fps_excluding_model_load']:.2f} FPS, "
            f"speedup needed {metrics['required_end_to_end_speedup_to_target']:.2f}x"
        ),
        flush=True,
    )
    return metrics


def write_reports(results, output_dir, target_fps):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target_fps": target_fps,
        "results": results,
    }
    json_path = output_path / f"realtime_sweep_{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Realtime Sweep",
        "",
        f"Target FPS: {target_fps:.2f}",
        "",
        "| preset | resized | frames | inference FPS | e2e FPS | e2e speedup needed | peak reserved GB | met realtime |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        resized = f"{item['resized_height']}x{item['resized_width']}"
        lines.append(
            (
                f"| {item.get('preset') or '-'} | {resized} | {item['bench_frames']} | "
                f"{item['inference_fps']:.2f} | "
                f"{item['end_to_end_fps_excluding_model_load']:.2f} | "
                f"{item['required_end_to_end_speedup_to_target']:.2f}x | "
                f"{item.get('cuda_peak_reserved_gb', 0):.2f} | "
                f"{'yes' if item['realtime_met_excluding_model_load'] else 'no'} |"
            )
        )

    md_path = output_path / f"realtime_sweep_{timestamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run DVD realtime preset sweeps.")
    parser.add_argument("--weights", default="ckpt/model.safetensors")
    parser.add_argument("--local_model_path", default="models")
    parser.add_argument("--input_video", default="test_video/depth_full_50frame.mp4")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--target_fps", type=float, default=25.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--presets", nargs="+", default=list(DEFAULT_PRESETS))
    parser.add_argument("--decode_resize", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    results = [run_benchmark(args, preset) for preset in args.presets]
    json_path, md_path = write_reports(results, args.output_dir, args.target_fps)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
