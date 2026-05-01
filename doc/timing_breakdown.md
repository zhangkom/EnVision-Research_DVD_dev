# DVD 耗时模块拆分

本文用于记录当前推理链路的模块耗时，方便判断优化优先级。

## 结论

当前最耗时的模块是 **inference/model.pipe**。在 4090 的 1000 帧测试里，即使是已经超过 25 FPS 的极限预览档，去掉一次性模型加载后，推理本体仍占端到端耗时约 90%。

用 `--profile_modules` 深拆后，1000 帧极限预览档里：

- `model.pipe` 窗口推理：`23.767s`，占 inference `93.4%`
- overlap 对齐拼接：`1.640s`，占 inference `6.4%`
- 窗口准备：约 `0.001s`

因此优化优先级是：

1. 先压低 `model.pipe` 的单窗口耗时：TensorRT FP16、INT8、轻量模型/蒸馏。
2. 再处理输入输出链路：`decode_resize`、跳过不必要的 `resize_back`、减少视频/PNG 保存。
3. 模型加载只影响离线批处理首条视频；服务常驻后不应计入实时帧率。

## 当前已知拆分

测试视频：`test_video/depth_full_50frame.mp4`，1000 frames，25 FPS。

### 4090 极限预览档，144x512，decode_resize，profile_modules

来源：`output/benchmark_profile_realtime_preview_1000f_1000f_144x512_win81_ov9_fp16.json`

| 模块 | 耗时 | 占总耗时，不含模型加载 |
| --- | ---: | ---: |
| inference | 25.445s | 89.3% |
| video decode + resize | 1.605s | 5.6% |
| resize depth back | 1.429s | 5.0% |
| save | 0.000s | 0.0% |

端到端不含模型加载：`28.478s`，`35.11 FPS`。

inference 内部：

| 推理内部模块 | 耗时 | 占 inference |
| --- | ---: | ---: |
| model.pipe windows | 23.767s | 93.4% |
| overlap alignment | 1.640s | 6.4% |
| window prepare | 0.001s | 0.0% |

共 14 个窗口，单窗口 `model.pipe` 平均 `1.698s`。第一个窗口约 `2.040s`，后续窗口约 `1.66-1.69s`。

如果把一次性模型加载也算进去：

| 模块 | 耗时 | 占总耗时 |
| --- | ---: | ---: |
| inference | 25.445s | 65.4% |
| model load | 10.400s | 26.8% |
| video decode + resize | 1.605s | 4.1% |
| resize depth back | 1.429s | 3.7% |

### 4090 预览档，192x704

来源：`output/benchmark_depth_full_50frame_h192_ov9_1000f_192x704_win81_ov9_fp16.json`

| 模块 | 耗时 | 占总耗时，不含模型加载 |
| --- | ---: | ---: |
| inference | 50.272s | 87.3% |
| video read/resize | 5.111s | 8.9% |
| resize depth back | 2.225s | 3.9% |

端到端不含模型加载：`57.609s`，`17.36 FPS`。

### 4090 吞吐档，256x928，overlap=9

来源：`output/benchmark_depth_full_50frame_h256_ov9_1000f_256x928_win81_ov9_fp16.json`

| 模块 | 耗时 | 占总耗时，不含模型加载 |
| --- | ---: | ---: |
| inference | 85.736s | 93.6% |
| video read/resize | 4.188s | 4.6% |
| resize depth back | 1.672s | 1.8% |

端到端不含模型加载：`91.597s`，`10.92 FPS`。

### 4090 平衡档，256x928，overlap=21，fast_video_save

来源：`output/benchmark_depth_full_50frame_h256_fast_1000f_256x928_win81_ov21_fp16.json`

| 模块 | 耗时 | 占总耗时，不含模型加载 |
| --- | ---: | ---: |
| inference | 106.183s | 85.9% |
| save visualization | 11.614s | 9.4% |
| video read/resize | 4.193s | 3.4% |
| resize depth back | 1.677s | 1.4% |

端到端不含模型加载：`123.667s`，`8.09 FPS`。

## 怎么生成报告

单个 benchmark：

```powershell
conda run -n dvd python tools\report_benchmark_modules.py `
  output\benchmark_depth_full_50frame_realtime_preview_decode_resize_1000f_144x512_win81_ov9_fp16.json
```

包含模型加载占比：

```powershell
conda run -n dvd python tools\report_benchmark_modules.py `
  --include_model_load `
  output\benchmark_depth_full_50frame_realtime_preview_decode_resize_1000f_144x512_win81_ov9_fp16.json
```

深度拆分推理内部：

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --preset realtime-preview `
  --target_fps 25 `
  --decode_resize `
  --profile_modules `
  --max_frames 81 `
  --no_save
```

`--profile_modules` 会额外记录：

- `inference_profile.model_pipe_s`
- `inference_profile.overlap_alignment_s`
- `inference_profile.window_prepare_s`
- `inference_profile.window_pipe_s`

这样可以确认推理阶段内部到底是模型窗口推理，还是 overlap 拼接在耗时。
