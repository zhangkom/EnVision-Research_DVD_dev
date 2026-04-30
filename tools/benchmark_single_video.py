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
    generate_depth_sliced,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DVD single-video inference.")
    parser.add_argument("--ckpt", default="ckpt")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--model_config", default="ckpt/model_config.yaml")
    parser.add_argument("--local_model_path", default="models")
    parser.add_argument("--input_video", default="test_video/depth_full_50frame.mp4")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=21)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_depth_npy", action="store_true")
    parser.add_argument("--save_depth_png16", action="store_true")
    return parser.parse_args()


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

    with timed(metrics, "video_read_resize_s"):
        input_tensor, origin_fps, original_frames = read_video_limited(
            args.input_video, args.max_frames
        )
        input_tensor, orig_size = resize_for_training_scale(
            input_tensor, args.height, args.width
        )

    frames = int(input_tensor.shape[1])
    metrics.update(
        {
            "original_frames": original_frames,
            "bench_frames": frames,
            "origin_fps": float(origin_fps),
            "orig_height": int(orig_size[0]),
            "orig_width": int(orig_size[1]),
            "resized_height": int(input_tensor.shape[-2]),
            "resized_width": int(input_tensor.shape[-1]),
        }
    )

    with timed(metrics, "inference_s"):
        depth = generate_depth_sliced(
            model, input_tensor, args.window_size, args.overlap
        )[0]

    with timed(metrics, "resize_back_s"):
        depth = resize_depth_back(depth, orig_size)

    if not args.no_save:
        with timed(metrics, "save_s"):
            output_path = save_results(depth, origin_fps, args)
        metrics["output_path"] = output_path
    else:
        metrics["save_s"] = 0.0

    stem = Path(args.input_video).stem
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
        f"benchmark_{stem}_{frames}f_{args.dtype}.json",
    )
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    metrics["benchmark_path"] = benchmark_path
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
