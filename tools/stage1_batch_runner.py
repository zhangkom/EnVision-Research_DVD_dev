import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.benchmark_single_video import (  # noqa: E402
    DTYPES,
    PRESETS,
    apply_stage1_optimizations,
    generate_depth_sliced,
    load_model,
    read_video_limited,
    read_video_limited_resized,
    resize_depth_back,
    resize_for_training_scale,
    save_depth_npy,
    save_depth_png16,
    save_depth_video_fast,
    timed,
)


DEFAULT_PRESETS = ("realtime", "realtime-preview", "speed-floor")


def sanitize_name(value):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def cuda_memory_metrics(metrics):
    if not torch.cuda.is_available():
        return
    metrics["cuda_peak_allocated_gb"] = round(
        torch.cuda.max_memory_allocated() / (1024**3), 2
    )
    metrics["cuda_peak_reserved_gb"] = round(
        torch.cuda.max_memory_reserved() / (1024**3), 2
    )


def preset_config(preset):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choices: {sorted(PRESETS)}")
    return PRESETS[preset]


def process_one_video(model, args, video_path, preset):
    config = preset_config(preset)
    height = config["height"]
    width = config["width"]
    window_size = config["window_size"]
    overlap = config["overlap"]
    video_path = str(video_path)
    stem = sanitize_name(Path(video_path).stem)
    preset_stem = sanitize_name(preset)
    run_stem = f"{args.run_prefix}_{stem}_{preset_stem}"

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_video": video_path,
        "preset": preset,
        "height": height,
        "width": width,
        "window_size": window_size,
        "overlap": overlap,
        "decode_resize": args.decode_resize,
        "no_resize_back": args.no_resize_back,
        "target_fps": args.target_fps,
        "run_stem": run_stem,
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.decode_resize:
        with timed(metrics, "video_decode_s"):
            input_tensor, origin_fps, original_frames, orig_size = (
                read_video_limited_resized(video_path, height, width, args.max_frames)
            )
        metrics["video_resize_s"] = 0.0
    else:
        with timed(metrics, "video_decode_s"):
            input_tensor, origin_fps, original_frames = read_video_limited(
                video_path, args.max_frames
            )
        with timed(metrics, "video_resize_s"):
            input_tensor, orig_size = resize_for_training_scale(
                input_tensor, height, width
            )

    metrics["video_read_resize_s"] = metrics["video_decode_s"] + metrics[
        "video_resize_s"
    ]
    frames = int(input_tensor.shape[1])
    metrics.update(
        {
            "original_frames": int(original_frames),
            "bench_frames": frames,
            "origin_fps": float(origin_fps),
            "orig_height": int(orig_size[0]),
            "orig_width": int(orig_size[1]),
            "resized_height": int(input_tensor.shape[-2]),
            "resized_width": int(input_tensor.shape[-1]),
        }
    )

    metrics["warmup_inference_s"] = 0.0
    if args.warmup_inference_runs > 0:
        with timed(metrics, "warmup_inference_s"):
            with torch.inference_mode():
                for _ in range(args.warmup_inference_runs):
                    _ = generate_depth_sliced(model, input_tensor, window_size, overlap)

    with timed(metrics, "inference_s"):
        with torch.inference_mode():
            depth = generate_depth_sliced(model, input_tensor, window_size, overlap)[0]

    if args.no_resize_back:
        metrics["resize_back_s"] = 0.0
        metrics["resize_back_skipped"] = True
    else:
        with timed(metrics, "resize_back_s"):
            depth = resize_depth_back(depth, orig_size)
        metrics["resize_back_skipped"] = False

    metrics["output_depth_height"] = int(depth.shape[1])
    metrics["output_depth_width"] = int(depth.shape[2])

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_video:
        with timed(metrics, "save_s"):
            metrics["output_path"] = save_depth_video_fast(
                depth, args.output_dir, run_stem, origin_fps, args.grayscale
            )
    else:
        metrics["save_s"] = 0.0

    if args.save_depth_npy:
        with timed(metrics, "save_depth_npy_s"):
            metrics["depth_npy_path"] = save_depth_npy(
                depth, args.output_dir, run_stem
            )
    else:
        metrics["save_depth_npy_s"] = 0.0

    if args.save_depth_png16:
        with timed(metrics, "save_depth_png16_s"):
            depth_dir, metadata_path = save_depth_png16(
                depth, args.output_dir, run_stem
            )
        metrics["depth_png16_dir"] = depth_dir
        metrics["depth_png16_metadata"] = metadata_path
    else:
        metrics["save_depth_png16_s"] = 0.0

    metrics["job_total_s"] = sum(
        metrics[k]
        for k in (
            "video_read_resize_s",
            "inference_s",
            "resize_back_s",
            "save_s",
            "save_depth_npy_s",
            "save_depth_png16_s",
        )
    )
    metrics["job_total_s_excluding_save"] = sum(
        metrics[k]
        for k in (
            "video_read_resize_s",
            "inference_s",
            "resize_back_s",
        )
    )
    metrics["inference_fps"] = frames / metrics["inference_s"]
    metrics["runtime_fps"] = frames / metrics["job_total_s"]
    metrics["runtime_fps_excluding_save"] = frames / metrics[
        "job_total_s_excluding_save"
    ]
    metrics["required_runtime_speedup_to_target"] = (
        args.target_fps / metrics["runtime_fps"]
    )
    metrics["required_runtime_no_save_speedup_to_target"] = (
        args.target_fps / metrics["runtime_fps_excluding_save"]
    )
    metrics["realtime_met_runtime"] = metrics["runtime_fps"] >= args.target_fps
    metrics["realtime_met_runtime_excluding_save"] = (
        metrics["runtime_fps_excluding_save"] >= args.target_fps
    )
    cuda_memory_metrics(metrics)

    if args.write_job_json:
        job_path = Path(args.output_dir) / (
            f"batch_job_{run_stem}_{frames}f_"
            f"{metrics['resized_height']}x{metrics['resized_width']}.json"
        )
        job_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics["job_json_path"] = str(job_path)

    del input_tensor, depth
    if torch.cuda.is_available() and args.empty_cache_between_jobs:
        torch.cuda.empty_cache()

    return metrics


def write_reports(summary, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_path / f"stage1_batch_{timestamp}.json"
    md_path = output_path / f"stage1_batch_{timestamp}.md"
    summary["summary_json_path"] = str(json_path)
    summary["summary_md_path"] = str(md_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Stage 1 Batch Runner",
        "",
        f"Model load: {summary['model_load_s']:.2f}s",
        f"Model setup: {summary['stage1_optimize_model_s']:.2f}s",
        f"Target FPS: {summary['target_fps']:.2f}",
        "",
        "| video | preset | frames | depth | inference FPS | runtime FPS | no-save FPS | realtime | outputs |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in summary["results"]:
        depth_size = f"{item['output_depth_height']}x{item['output_depth_width']}"
        outputs = []
        for key in ("output_path", "depth_npy_path", "depth_png16_dir"):
            if item.get(key):
                outputs.append(item[key])
        lines.append(
            (
                f"| {Path(item['input_video']).name} | {item['preset']} | "
                f"{item['bench_frames']} | {depth_size} | "
                f"{item['inference_fps']:.2f} | {item['runtime_fps']:.2f} | "
                f"{item['runtime_fps_excluding_save']:.2f} | "
                f"{'yes' if item['realtime_met_runtime_excluding_save'] else 'no'} | "
                f"{'<br>'.join(outputs)} |"
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stage-1 DVD inference with one persistent model load."
    )
    parser.add_argument("--weights", default="ckpt/model.safetensors")
    parser.add_argument("--model_config", default="ckpt/model_config.yaml")
    parser.add_argument("--local_model_path", default="models")
    parser.add_argument(
        "--input_videos",
        nargs="+",
        default=["test_video/depth_full_50frame.mp4"],
    )
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--presets", nargs="+", default=list(DEFAULT_PRESETS))
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_fps", type=float, default=25.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--decode_resize", action="store_true")
    parser.add_argument("--no_resize_back", action="store_true")
    parser.add_argument("--compile_dit", action="store_true")
    parser.add_argument("--compile_backend", default="inductor")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--vae_channels_last_3d", action="store_true")
    parser.add_argument("--warmup_inference_runs", type=int, default=0)
    parser.add_argument("--run_prefix", default="stage1_batch")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--save_depth_npy", action="store_true")
    parser.add_argument("--save_depth_png16", action="store_true")
    parser.add_argument("--write_job_json", action="store_true")
    parser.add_argument("--empty_cache_between_jobs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)

    yaml_args = OmegaConf.load(args.model_config)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "weights": args.weights,
        "local_model_path": args.local_model_path,
        "model_config": args.model_config,
        "input_videos": args.input_videos,
        "presets": args.presets,
        "dtype_requested": args.dtype,
        "device_requested": args.device,
        "decode_resize": args.decode_resize,
        "no_resize_back": args.no_resize_back,
        "target_fps": args.target_fps,
        "results": [],
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        summary.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_runtime": torch.version.cuda,
                "compute_capability": f"{props.major}.{props.minor}",
                "gpu_memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )

    with timed(summary, "model_load_s"):
        model = load_model(
            args.weights,
            yaml_args,
            args.device,
            DTYPES[args.dtype],
            args.local_model_path,
        )

    with timed(summary, "stage1_optimize_model_s"):
        model = apply_stage1_optimizations(model, args, summary)

    for video_path in args.input_videos:
        for preset in args.presets:
            print(f"Running {video_path} preset={preset}", flush=True)
            metrics = process_one_video(model, args, video_path, preset)
            summary["results"].append(metrics)
            print(
                (
                    f"{Path(video_path).name} {preset}: "
                    f"inference {metrics['inference_fps']:.2f} FPS, "
                    f"runtime-no-save {metrics['runtime_fps_excluding_save']:.2f} FPS"
                ),
                flush=True,
            )

    summary["jobs"] = len(summary["results"])
    summary["batch_runtime_s_excluding_model_load"] = sum(
        item["job_total_s"] for item in summary["results"]
    )
    summary["batch_frames"] = sum(item["bench_frames"] for item in summary["results"])
    if summary["batch_runtime_s_excluding_model_load"] > 0:
        summary["batch_runtime_fps"] = (
            summary["batch_frames"] / summary["batch_runtime_s_excluding_model_load"]
        )

    json_path, md_path = write_reports(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
