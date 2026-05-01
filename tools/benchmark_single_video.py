import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file

from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule
from test_script.test_single_video import (
    compute_scale_and_shift,
    generate_depth_sliced,
    get_window_index,
    pad_time_mod4,
    resize_depth_back,
    resize_for_training_scale,
    save_results,
)


DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


PRESETS = {
    "quality": {
        "height": 480,
        "width": 640,
        "window_size": 81,
        "overlap": 21,
    },
    "balanced": {
        "height": 256,
        "width": 640,
        "window_size": 81,
        "overlap": 21,
    },
    "throughput": {
        "height": 192,
        "width": 640,
        "window_size": 81,
        "overlap": 9,
    },
    "realtime-preview": {
        "height": 128,
        "width": 512,
        "window_size": 81,
        "overlap": 9,
    },
    "speed-floor": {
        "height": 96,
        "width": 384,
        "window_size": 81,
        "overlap": 9,
    },
}


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed(metrics, name):
    cuda_sync()
    start = time.perf_counter()
    yield
    cuda_sync()
    metrics[name] = time.perf_counter() - start


def read_video_limited(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0), fps, total_frames


def training_resize_shape(height, width, target_h, target_w):
    ratio = max(target_h / height, target_w / width)
    new_height = int(np.ceil(height * ratio))
    new_width = int(np.ceil(width * ratio))
    new_height = (new_height + 15) // 16 * 16
    new_width = (new_width + 15) // 16 * 16
    return new_height, new_width


def read_video_limited_resized(video_path, target_h, target_w, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    orig_size = None
    resized_size = None
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        if orig_size is None:
            height, width = frame.shape[:2]
            orig_size = (height, width)
            resized_size = training_resize_shape(height, width, target_h, target_w)

        if resized_size != orig_size:
            frame = cv2.resize(
                frame,
                (resized_size[1], resized_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0), fps, total_frames, orig_size


def depth_to_single_channel(depth):
    if depth.ndim == 4 and depth.shape[-1] > 1:
        return depth.mean(axis=-1)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        return depth[..., 0]
    return depth


def save_depth_npy(depth, output_dir, stem):
    depth_single = depth_to_single_channel(depth).astype(np.float16)
    path = os.path.join(output_dir, f"{stem}_depth.npy")
    np.save(path, depth_single)
    return path


def save_depth_png16(depth, output_dir, stem):
    depth_single = depth_to_single_channel(depth).astype(np.float32)
    depth_dir = os.path.join(output_dir, f"{stem}_depth_png16")
    os.makedirs(depth_dir, exist_ok=True)

    d_min = float(np.nanmin(depth_single))
    d_max = float(np.nanmax(depth_single))
    denom = max(d_max - d_min, 1e-8)
    depth_u16 = ((depth_single - d_min) / denom * 65535.0).clip(0, 65535).astype(
        np.uint16
    )

    for idx, frame in enumerate(depth_u16):
        cv2.imwrite(os.path.join(depth_dir, f"{idx:06d}.png"), frame)

    metadata_path = os.path.join(depth_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"min": d_min, "max": d_max, "dtype": "uint16"}, f, indent=2)
    return depth_dir, metadata_path


def save_depth_video_fast(depth, output_dir, stem, fps, grayscale=False):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{stem}_{'gray' if grayscale else 'color'}_depth_vis.mp4"
    )

    depth_single = depth_to_single_channel(depth).astype(np.float32)
    d_min = float(np.nanmin(depth_single))
    d_max = float(np.nanmax(depth_single))
    denom = max(d_max - d_min, 1e-8)
    depth_u8 = ((depth_single - d_min) / denom * 255.0).clip(0, 255).astype(np.uint8)

    height, width = depth_u8.shape[1:3]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    for frame in depth_u8:
        if grayscale:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = cv2.applyColorMap(frame, cv2.COLORMAP_TURBO)
        writer.write(frame_bgr)
    writer.release()
    return output_path


def load_model(weights, yaml_args, device, dtype, local_model_path):
    accelerator = Accelerator()
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=yaml_args.lora_rank,
        lora_base_model=yaml_args.lora_base_model,
        args=yaml_args,
        torch_dtype=dtype,
        skip_download=True,
        local_model_path=local_model_path,
    )

    state_dict = load_file(weights, device="cpu")
    dit_state_dict = {
        k.replace("pipe.dit.", ""): v
        for k, v in state_dict.items()
        if k.startswith("pipe.dit.")
    }
    if not dit_state_dict:
        dit_state_dict = state_dict

    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()
    model = model.to(device=device, dtype=dtype)
    model.pipe.torch_dtype = dtype
    model.eval()
    return model


def apply_stage1_optimizations(model, args, metrics):
    metrics["compile_dit_requested"] = bool(args.compile_dit)
    metrics["vae_channels_last_3d_requested"] = bool(args.vae_channels_last_3d)

    if args.vae_channels_last_3d:
        try:
            model.pipe.vae = model.pipe.vae.to(memory_format=torch.channels_last_3d)
            metrics["vae_channels_last_3d_applied"] = True
        except Exception as exc:
            metrics["vae_channels_last_3d_applied"] = False
            metrics["vae_channels_last_3d_error"] = repr(exc)

    if args.compile_dit:
        if not hasattr(torch, "compile"):
            metrics["compile_dit_applied"] = False
            metrics["compile_dit_error"] = "torch.compile is not available."
            return model
        try:
            compile_kwargs = {
                "mode": args.compile_mode,
                "backend": args.compile_backend,
            }
            model.pipe.dit = torch.compile(model.pipe.dit, **compile_kwargs)
            metrics["compile_dit_applied"] = True
            metrics["compile_dit_backend"] = args.compile_backend
            metrics["compile_dit_mode"] = args.compile_mode
        except Exception as exc:
            metrics["compile_dit_applied"] = False
            metrics["compile_dit_error"] = repr(exc)

    return model


def generate_depth_sliced_profiled(
    model, input_rgb, window_size=45, overlap=9, scale_only=False
):
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    print(f"depth_windows {depth_windows}")

    profile = {
        "window_count": len(depth_windows),
        "window_prepare_s": 0.0,
        "model_pipe_s": 0.0,
        "overlap_alignment_s": 0.0,
        "window_pipe_s": [],
    }
    depth_res_list = []

    for start, end in depth_windows:
        prepare_start = time.perf_counter()
        input_rgb_slice = input_rgb[:, start:end]
        input_rgb_slice, origin_T = pad_time_mod4(input_rgb_slice)
        input_frame = input_rgb_slice.shape[1]
        input_height, input_width = input_rgb_slice.shape[-2:]
        profile["window_prepare_s"] += time.perf_counter() - prepare_start

        cuda_sync()
        pipe_start = time.perf_counter()
        extra_image_frame_index = torch.ones([B, input_frame]).to(model.pipe.device)
        outputs = model.pipe(
            prompt=[""] * B,
            negative_prompt=[""] * B,
            mode=model.args.mode,
            height=input_height,
            width=input_width,
            num_frames=input_frame,
            batch_size=B,
            input_image=input_rgb_slice[:, 0],
            extra_images=input_rgb_slice,
            extra_image_frame_index=extra_image_frame_index,
            input_video=input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        cuda_sync()
        pipe_s = time.perf_counter() - pipe_start
        profile["model_pipe_s"] += pipe_s
        profile["window_pipe_s"].append(
            {"start": int(start), "end": int(end), "seconds": pipe_s}
        )
        depth_res_list.append(outputs["depth"][:, :origin_T])

    align_start = time.perf_counter()
    depth_list_aligned = None
    prev_end = None
    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue

        curr_start = start
        real_overlap = prev_end - curr_start

        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]

            if scale_only:
                scale = np.sum(curr_frames * ref_frames) / (
                    np.sum(curr_frames * curr_frames) + 1e-6
                )
                shift = 0.0
            else:
                scale, shift = compute_scale_and_shift(curr_frames, ref_frames)

            scale = np.clip(scale, 0.7, 1.5)
            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0

            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(
                1, real_overlap, 1, 1, 1
            )
            smooth_overlap = (1 - alpha) * ref_frames + alpha * aligned_t[
                :, :real_overlap
            ]

            depth_list_aligned = np.concatenate(
                [
                    depth_list_aligned[:, :-real_overlap],
                    smooth_overlap,
                    aligned_t[:, real_overlap:],
                ],
                axis=1,
            )
        else:
            depth_list_aligned = np.concatenate([depth_list_aligned, t], axis=1)

        prev_end = end

    profile["overlap_alignment_s"] = time.perf_counter() - align_start
    if profile["window_count"]:
        profile["model_pipe_avg_s"] = profile["model_pipe_s"] / profile[
            "window_count"
        ]
    return depth_list_aligned[:, :T], profile


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DVD single-video inference.")
    parser.add_argument("--ckpt", default="ckpt")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--model_config", default="ckpt/model_config.yaml")
    parser.add_argument("--local_model_path", default="models")
    parser.add_argument("--input_video", default="test_video/depth_full_50frame.mp4")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=21)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--target_fps", type=float, default=None)
    parser.add_argument("--decode_resize", action="store_true")
    parser.add_argument("--no_resize_back", action="store_true")
    parser.add_argument("--profile_modules", action="store_true")
    parser.add_argument("--compile_dit", action="store_true")
    parser.add_argument("--compile_backend", default="inductor")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--vae_channels_last_3d", action="store_true")
    parser.add_argument("--warmup_inference_runs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--fast_video_save", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_depth_npy", action="store_true")
    parser.add_argument("--save_depth_png16", action="store_true")
    args = parser.parse_args()
    if args.preset:
        explicit_flags = {
            arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for key, value in PRESETS[args.preset].items():
            if f"--{key}" not in explicit_flags:
                setattr(args, key, value)
    return args


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)

    weights = args.weights or os.path.join(args.ckpt, "model.safetensors")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Missing weights: {weights}. Download DVD weights into ckpt/ or pass --weights."
        )

    yaml_args = OmegaConf.load(args.model_config)
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_video": args.input_video,
        "height": args.height,
        "width": args.width,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "preset": args.preset,
        "decode_resize": args.decode_resize,
        "no_resize_back": args.no_resize_back,
        "profile_modules": args.profile_modules,
        "warmup_inference_runs": args.warmup_inference_runs,
        "weights": weights,
        "local_model_path": args.local_model_path,
        "device_requested": args.device,
        "dtype_requested": args.dtype,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metrics.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_runtime": torch.version.cuda,
                "compute_capability": f"{props.major}.{props.minor}",
                "gpu_memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )
        torch.cuda.reset_peak_memory_stats()

    with timed(metrics, "model_load_s"):
        model = load_model(
            weights,
            yaml_args,
            args.device,
            DTYPES[args.dtype],
            args.local_model_path,
        )

    with timed(metrics, "stage1_optimize_model_s"):
        model = apply_stage1_optimizations(model, args, metrics)

    if args.decode_resize:
        with timed(metrics, "video_decode_s"):
            input_tensor, origin_fps, original_frames, orig_size = (
                read_video_limited_resized(
                    args.input_video, args.height, args.width, args.max_frames
                )
            )
        metrics["video_resize_s"] = 0.0
    else:
        with timed(metrics, "video_decode_s"):
            input_tensor, origin_fps, original_frames = read_video_limited(
                args.input_video, args.max_frames
            )
        with timed(metrics, "video_resize_s"):
            input_tensor, orig_size = resize_for_training_scale(
                input_tensor, args.height, args.width
            )
    metrics["video_read_resize_s"] = metrics["video_decode_s"] + metrics[
        "video_resize_s"
    ]

    frames = int(input_tensor.shape[1])
    metrics.update(
        {
            "original_frames": original_frames,
            "bench_frames": frames,
            "origin_fps": float(origin_fps),
            "target_fps": float(args.target_fps or origin_fps),
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
                    _ = generate_depth_sliced(
                        model, input_tensor, args.window_size, args.overlap
                    )

    with timed(metrics, "inference_s"):
        with torch.inference_mode():
            if args.profile_modules:
                depth_profiled, profile_modules = generate_depth_sliced_profiled(
                    model, input_tensor, args.window_size, args.overlap
                )
                depth = depth_profiled[0]
                metrics["inference_profile"] = profile_modules
            else:
                depth = generate_depth_sliced(
                    model, input_tensor, args.window_size, args.overlap
                )[0]

    if args.no_resize_back:
        metrics["resize_back_s"] = 0.0
        metrics["resize_back_skipped"] = True
    else:
        with timed(metrics, "resize_back_s"):
            depth = resize_depth_back(depth, orig_size)
        metrics["resize_back_skipped"] = False

    metrics["output_depth_height"] = int(depth.shape[1])
    metrics["output_depth_width"] = int(depth.shape[2])

    stem = args.run_name or Path(args.input_video).stem
    if not args.no_save:
        with timed(metrics, "save_s"):
            if args.fast_video_save:
                output_path = save_depth_video_fast(
                    depth, args.output_dir, stem, origin_fps, args.grayscale
                )
            else:
                save_args = argparse.Namespace(**vars(args))
                if args.run_name:
                    save_args.input_video = f"{args.run_name}.mp4"
                output_path = save_results(depth, origin_fps, save_args)
        metrics["output_path"] = output_path
    else:
        metrics["save_s"] = 0.0

    if args.save_depth_npy:
        with timed(metrics, "save_depth_npy_s"):
            metrics["depth_npy_path"] = save_depth_npy(depth, args.output_dir, stem)
    else:
        metrics["save_depth_npy_s"] = 0.0

    if args.save_depth_png16:
        with timed(metrics, "save_depth_png16_s"):
            depth_dir, metadata_path = save_depth_png16(depth, args.output_dir, stem)
        metrics["depth_png16_dir"] = depth_dir
        metrics["depth_png16_metadata"] = metadata_path
    else:
        metrics["save_depth_png16_s"] = 0.0

    metrics["total_s"] = sum(
        metrics[k]
        for k in (
            "model_load_s",
            "stage1_optimize_model_s",
            "video_read_resize_s",
            "inference_s",
            "resize_back_s",
            "save_s",
            "save_depth_npy_s",
            "save_depth_png16_s",
        )
    )
    metrics["inference_fps"] = frames / metrics["inference_s"]
    metrics["end_to_end_fps_excluding_model_load"] = frames / (
        metrics["total_s"] - metrics["model_load_s"]
    )
    runtime_s_excluding_setup = (
        metrics["total_s"]
        - metrics["model_load_s"]
        - metrics.get("stage1_optimize_model_s", 0.0)
    )
    metrics["runtime_s_excluding_model_load_and_setup"] = runtime_s_excluding_setup
    metrics["runtime_fps_excluding_model_load_and_setup"] = (
        frames / runtime_s_excluding_setup
    )
    target_fps = metrics["target_fps"]
    metrics["inference_realtime_ratio"] = metrics["inference_fps"] / target_fps
    metrics["end_to_end_realtime_ratio_excluding_model_load"] = (
        metrics["end_to_end_fps_excluding_model_load"] / target_fps
    )
    metrics["runtime_realtime_ratio_excluding_model_load_and_setup"] = (
        metrics["runtime_fps_excluding_model_load_and_setup"] / target_fps
    )
    metrics["required_inference_speedup_to_target"] = target_fps / metrics[
        "inference_fps"
    ]
    metrics["required_end_to_end_speedup_to_target"] = target_fps / metrics[
        "end_to_end_fps_excluding_model_load"
    ]
    metrics["required_runtime_speedup_to_target"] = target_fps / metrics[
        "runtime_fps_excluding_model_load_and_setup"
    ]
    metrics["realtime_met_inference_only"] = metrics["inference_fps"] >= target_fps
    metrics["realtime_met_excluding_model_load"] = (
        metrics["end_to_end_fps_excluding_model_load"] >= target_fps
    )
    metrics["realtime_met_excluding_model_load_and_setup"] = (
        metrics["runtime_fps_excluding_model_load_and_setup"] >= target_fps
    )
    if torch.cuda.is_available():
        metrics["cuda_peak_allocated_gb"] = round(
            torch.cuda.max_memory_allocated() / (1024**3), 2
        )
        metrics["cuda_peak_reserved_gb"] = round(
            torch.cuda.max_memory_reserved() / (1024**3), 2
        )

    os.makedirs(args.output_dir, exist_ok=True)
    benchmark_path = os.path.join(
        args.output_dir,
        (
            f"benchmark_{stem}_{frames}f_"
            f"{metrics['resized_height']}x{metrics['resized_width']}_"
            f"win{args.window_size}_ov{args.overlap}_{args.dtype}.json"
        ),
    )
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    metrics["benchmark_path"] = benchmark_path
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
