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

Common speed/quality presets are available through `--preset`:

```text
quality          480x640 target, window 81, overlap 21
balanced         256x640 target, window 81, overlap 21
throughput       192x640 target, window 81, overlap 9
realtime-preview 128x512 target, window 81, overlap 9
```

See `doc/optimization_strategy.md` for the GGUF/FP8 analysis and the real-time benchmark notes.

For the real-time goal, run the sweep helper on the target GPU:

```powershell
conda run -n dvd python tools\check_runtime_capability.py
```

Then run the preset sweep:

```powershell
conda run -n dvd python tools\realtime_sweep.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --target_fps 25 `
  --decode_resize `
  --presets balanced throughput realtime-preview
```

The sweep writes JSON and Markdown summaries with the measured FPS and the remaining speedup needed to hit 25 FPS.
`--decode_resize` resizes each frame during decoding so the real-time path avoids an extra full-resolution tensor resize step.

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
For long videos, add `--fast_video_save` to write the visualization with OpenCV instead of the slower ImageIO/Matplotlib path:

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --height 256 `
  --width 640 `
  --dtype fp16 `
  --run_name depth_full_50frame_h256 `
  --fast_video_save
```

The fast visualization uses OpenCV's Turbo colormap. It is intended for review/playback; raw depth outputs are unaffected.

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

The complete 1000-frame test video at `256x928` measured `106.18s` pure inference, `9.42 FPS`, peak reserved memory `7.81GB`. The legacy visualization save path took `84.47s`; `--fast_video_save` reduced visualization saving to `11.61s`, bringing end-to-end throughput excluding model load to `8.09 FPS`.

Reducing overlap can improve throughput further:

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --height 256 `
  --width 640 `
  --window_size 81 `
  --overlap 9 `
  --dtype fp16 `
  --no_save
```

On the same 1000-frame video, `overlap=9` reduced the window count from 17 to 14 and measured `85.74s` pure inference, `11.66 FPS`. This is a quality/speed tradeoff: the overlap-alignment MAE showed larger spikes, so visually compare against `overlap=21` before using it for production.

Quadro RTX 6000 Turing has 24GB VRAM, so the default 480p window should fit, but it is expected to be slower than RTX 4090.
