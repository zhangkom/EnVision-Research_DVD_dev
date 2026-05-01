import json

import torch


def cuda_capability():
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "recommendations": ["CUDA is not available. Use a CUDA GPU for realtime work."],
        }

    props = torch.cuda.get_device_properties(0)
    major = int(props.major)
    minor = int(props.minor)
    compute = major + minor / 10
    native_fp8_candidate = compute >= 8.9
    bf16_supported = bool(torch.cuda.is_bf16_supported())

    recommendations = []
    if compute <= 7.5:
        recommendations.extend(
            [
                "Use FP16 as the default inference dtype.",
                "Do not prioritize FP8 on this GPU; Turing-class devices do not expose a native FP8 fast path.",
                "For aggressive acceleration, test TensorRT FP16 first and TensorRT INT8 with calibration second.",
            ]
        )
    elif native_fp8_candidate:
        recommendations.extend(
            [
                "FP8 may be worth a separate experiment if the TensorRT/PyTorch stack supports it on this machine.",
                "Still validate depth stability against an FP16 teacher output before accepting FP8.",
            ]
        )
    else:
        recommendations.extend(
            [
                "Use FP16/BF16 based on measured speed and quality.",
                "Prioritize TensorRT FP16 or INT8 before FP8 unless the deployment stack proves native FP8 support.",
            ]
        )

    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": f"{major}.{minor}",
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "bf16_supported_by_torch": bf16_supported,
        "native_fp8_candidate": native_fp8_candidate,
        "recommended_realtime_check": (
            "python tools/realtime_sweep.py --target_fps 25 "
            "--presets balanced throughput realtime-preview"
        ),
        "recommendations": recommendations,
    }


def main():
    print(json.dumps(cuda_capability(), indent=2))


if __name__ == "__main__":
    main()
