# Conda Inference Setup

Use this path for local inference and benchmarking. It avoids training-only dependencies such as Deepspeed.

## Create Environment

```powershell
conda create -n dvd python=3.10 -y
```

If GitHub/Hugging Face/PyPI need the local proxy:

```powershell
$env:HTTP_PROXY='http://127.0.0.1:25378'
$env:HTTPS_PROXY='http://127.0.0.1:25378'
```

Install PyTorch first. This example matches the tested Windows RTX 4090 environment:

```powershell
conda run -n dvd python -m pip install --upgrade pip
conda run -n dvd python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install inference dependencies and the editable project:

```powershell
conda run -n dvd python -m pip install -r requirements-inference.txt
conda run -n dvd python -m pip install -e . --no-deps
```

## Download Weights

DVD weights:

```powershell
conda run -n dvd huggingface-cli download FayeHongfeiZhang/DVD --revision main --local-dir ckpt --max-workers 4
```

Wan2.1 base DiT, VAE, and tokenizer files:

```powershell
conda run -n dvd huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B `
  --include "diffusion_pytorch_model*.safetensors" "Wan2.1_VAE.pth" "google/*" `
  --local-dir "models\Wan-AI\Wan2.1-T2V-1.3B" `
  --max-workers 4
```

Expected local files:

```text
ckpt/model_config.yaml
ckpt/model.safetensors
models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/*
```

## Check CUDA

```powershell
conda run -n dvd python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Quadro RTX 6000 Turing Notes

Use `--dtype fp16`. Do not use BF16 as the default runtime choice on Turing.

Start with the lower-memory speed profile:

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --height 256 `
  --width 640 `
  --window_size 81 `
  --overlap 21 `
  --dtype fp16 `
  --fast_video_save
```

If quality is acceptable, test `--overlap 9` for higher throughput.
