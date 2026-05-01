import importlib.util
import json

import torch


def has_module(name):
    return importlib.util.find_spec(name) is not None


def main():
    cuda_available = torch.cuda.is_available()
    info = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "torch_compile_available": hasattr(torch, "compile"),
        "torch_float8_e4m3fn_available": hasattr(torch, "float8_e4m3fn"),
        "torch_float8_e5m2_available": hasattr(torch, "float8_e5m2"),
        "tensorrt_python_available": has_module("tensorrt"),
        "torch_tensorrt_available": has_module("torch_tensorrt"),
        "onnx_available": has_module("onnx"),
        "onnxruntime_available": has_module("onnxruntime"),
        "xformers_available": has_module("xformers"),
        "flash_attn_available": has_module("flash_attn"),
    }

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "bf16_supported_by_torch": bool(torch.cuda.is_bf16_supported()),
            }
        )

    recommendations = []
    if info["torch_compile_available"]:
        recommendations.append("torch.compile can be tested on fixed 4090 shapes.")
    if info["tensorrt_python_available"] or info["torch_tensorrt_available"]:
        recommendations.append("TensorRT path is available for FP16/FP8 experiments.")
    else:
        recommendations.append(
            "TensorRT Python packages are not visible in this conda env; install them before TensorRT FP16/FP8 engine work."
        )
    if info["torch_float8_e4m3fn_available"] or info["torch_float8_e5m2_available"]:
        recommendations.append(
            "Torch exposes float8 dtypes, but DVD still needs a real quantization/export path before FP8 can be accepted."
        )
    if not info["xformers_available"]:
        recommendations.append("xFormers is not visible; keep using the current attention path unless installed.")

    info["recommendations"] = recommendations
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
