# DVD Inference Benchmark

This repository keeps model weights, local test videos, and generated outputs out of Git.

Recommended local layout:

```text
EnVision-Research_DVD_dev/
├── ckpt/
│   ├── model_config.yaml
│   └── model.safetensors
├── models/
│   └── Wan-AI/Wan2.1-T2V-1.3B/
├── test_video/
│   └── depth_full_50frame.mp4
└── output/
```

For Quadro RTX 6000 Turing, use FP16 rather than BF16:

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --height 480 `
  --width 640 `
  --window_size 81 `
  --overlap 21 `
  --dtype fp16
```

If the weights live outside this repository, pass explicit paths:

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --dtype fp16
```

The script writes a depth visualization video and a JSON timing report into `output/`.
The JSON includes model loading time, video decode/resize time, inference time, resize-back time, save time, FPS, and CUDA peak memory.

For downstream 2D-to-3D processing, save raw relative depth instead of only the color visualization:

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --dtype fp16 `
  --save_depth_npy `
  --save_depth_png16
```

`--save_depth_npy` writes a single-channel float16 relative-depth tensor with shape `T,H,W`.
`--save_depth_png16` writes globally normalized 16-bit PNG frames plus `metadata.json` containing the original min/max used for normalization.

On the local RTX 4090 test machine, `robot_navi.mp4` at resized `480x880` measured:

```text
81 frames: inference 12.58s, 6.44 FPS, peak allocated 8.63GB
1210 frames: inference 301.17s, 4.02 FPS, no output video save
```

On the provided wide test video `test_video/depth_full_50frame.mp4` at resized `480x1728`:

```text
Baseline 81 frames: inference 27.04s, 3.00 FPS, peak reserved 16.75GB
After preprocessing/VAE concat optimization: inference 25.85s, 3.13 FPS, peak reserved 17.08GB
```

For a faster preview/throughput profile on the same wide video, lowering the target height reduces the actual processed size:

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --height 256 `
  --width 640 `
  --window_size 81 `
  --overlap 21 `
  --dtype fp16 `
  --no_save
```

This processes at `256x928` and measured `6.10s` for 81 frames, `13.28 FPS`, with peak reserved memory `7.81GB` on RTX 4090. This is a speed/quality tradeoff and should be visually checked before production use.

Quadro RTX 6000 Turing has 24GB VRAM, so the default 480p window should fit, but it is expected to be slower than RTX 4090.
