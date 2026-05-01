# 第一阶段 RTX 4090 实现结果

第一阶段目标：先以本机 RTX 4090 作为交付目标，不考虑 Quadro RTX 6000 Turing 限制，先把实时链路跑通并形成可评估样本。

## 已实现

### 1. 跳过回原分辨率后处理

`tools/benchmark_single_video.py` 新增：

- `--no_resize_back`
- `output_depth_height`
- `output_depth_width`
- `resize_back_skipped`

打开 `--no_resize_back` 后，深度图保持推理尺寸输出，例如：

- `realtime-preview`: `144x512`
- `speed-floor`: `112x384`

这可以减少实时链路里的上采样开销，也更符合“下游统一处理低分辨率深度”的方案。

### 2. 阶段一完整 preset 矩阵

命令：

```powershell
conda run -n dvd python tools\realtime_sweep.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --target_fps 25 `
  --decode_resize `
  --no_resize_back `
  --presets balanced throughput realtime-preview speed-floor
```

结果文件：`output/realtime_sweep_20260501_090924.md`

| preset | resized/output depth | frames | inference FPS | e2e FPS，不含模型加载 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| balanced | 256x928 | 1000 | 8.59 | 8.44 | 不实时，质量候选 |
| throughput | 192x704 | 1000 | 20.06 | 19.36 | 接近实时，还差约 1.29x |
| realtime | 160x576 | 1000 | 29.85 | 28.47 | 实时，当前优先质量候选 |
| realtime-preview | 144x512 | 1000 | 38.79 | 36.53 | 实时，需质量评估 |
| speed-floor | 112x384 | 1000 | 67.34 | 61.38 | 实时，质量风险最高 |

第一阶段速度结论：

- `realtime` 已经满足 25 FPS，而且比 `realtime-preview` 多约 25% 像素。
- `realtime-preview` 速度余量更大。
- `speed-floor` 有较大速度余量。
- `throughput` 还差约 1.29x，后续可通过 TensorRT/量化/关键帧传播继续冲。
- `balanced` 当前离实时较远，更适合作为质量参考或 teacher。

### 3. 阶段一质量样本

命令：

```powershell
conda run -n dvd python tools\stage1_quality_samples.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --target_fps 25 `
  --max_frames 121 `
  --no_resize_back `
  --presets balanced throughput realtime-preview speed-floor
```

结果文件：`output/stage1_quality_samples_20260501_091114.md`

| preset | frames | depth size | e2e FPS，含保存，不含模型加载 | 样本 |
| --- | ---: | --- | ---: | --- |
| balanced | 121 | 256x928 | 8.28 | `output/stage1_balanced_color_depth_vis.mp4` |
| throughput | 121 | 192x704 | 14.79 | `output/stage1_throughput_color_depth_vis.mp4` |
| realtime | 121 | 160x576 | 20.18 | `output/stage1_realtime_color_depth_vis.mp4` |
| realtime-preview | 121 | 144x512 | 26.56 | `output/stage1_realtime_preview_color_depth_vis.mp4` |
| speed-floor | 121 | 112x384 | 40.69 | `output/stage1_speed_floor_color_depth_vis.mp4` |

每个样本还同时输出：

- `*_depth.npy`
- `*_depth_png16/`
- benchmark JSON

已生成快速审片拼图：

- `output/stage1_contact_sheet.png`

这张图把源视频、`balanced`、`throughput`、`realtime`、`realtime-preview`、`speed-floor` 在相同帧位拼到一起，适合先快速判断低分辨率深度是否有明显失真、闪烁或边缘问题。

下一步需要人工检查 `realtime`、`realtime-preview` 和 `speed-floor` 的 2D 转 3D 效果。如果 `160x576` 能接受，第一阶段优先用 `realtime`；如果质量不够但 `144x512` 能接受，则用 `realtime-preview`；如果两者都不够，则优先冲 `throughput`。

### 4. 常驻 batch runner

已新增：

- `tools/stage1_batch_runner.py`

用途：

- 只加载一次模型。
- 连续处理多个视频或多个 preset。
- 分开记录 `model_load_s`、每个 job 的 decode/inference/save 耗时。
- 更接近后续服务化/批处理形态。

命令：

```powershell
conda run -n dvd python tools\stage1_batch_runner.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_videos test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --target_fps 25 `
  --decode_resize `
  --no_resize_back `
  --presets realtime-preview speed-floor `
  --write_job_json
```

结果文件：

- `output/stage1_batch_20260501_093536.md`
- `output/stage1_batch_20260501_092751.md`

| preset | frames | depth size | inference FPS | runtime FPS，不含模型加载 |
| --- | ---: | --- | ---: | ---: |
| realtime | 1000 | 160x576 | 29.12 | 27.49 |
| realtime-preview | 1000 | 144x512 | 38.97 | 36.28 |
| speed-floor | 1000 | 112x384 | 69.35 | 61.76 |

结论：

- 模型加载约 `11.33s`，属于一次性启动成本。
- 常驻形态下 `realtime-preview` 和 `speed-floor` 都稳定超过 25 FPS。
- 现在第一阶段瓶颈已经从“能否实时”转为“`144x512` 或 `112x384` 的深度质量是否满足 2D 转 3D”。

### 5. 第一阶段加速环境检测

命令：

```powershell
conda run -n dvd python tools\check_stage1_acceleration.py
```

本机结果：

| 项目 | 状态 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| compute capability | 8.9 |
| PyTorch | 2.8.0+cu128 |
| torch.compile | available |
| torch float8 dtype | available |
| TensorRT Python | not available |
| torch_tensorrt | not available |
| ONNX / ONNX Runtime | not available |
| xFormers | not available |
| flash-attn | not available |

结论：

- PyTorch compile 可以试。
- FP8 dtype 可见，但还没有可接受的 FP8 量化/导出链路。
- TensorRT FP16/FP8 当前环境缺依赖，下一步需要先安装 TensorRT/ONNX 相关包。

### 6. PyTorch compile 初测

已新增：

- `--compile_dit`
- `--compile_backend`
- `--compile_mode`
- `--warmup_inference_runs`
- `runtime_fps_excluding_model_load_and_setup`

`speed-floor + torch.compile + warmup`：

- inference FPS: `69.19`
- runtime FPS excluding model load/setup: `63.00`
- 未 compile 的 `speed-floor` e2e FPS: `61.38`
- 约 `2.6%` 提升

`throughput + torch.compile + warmup`：

- inference FPS: `20.26`
- runtime FPS excluding model load/setup: `19.55`
- 未 compile 的 `throughput` e2e FPS: `19.36`
- 约 `1%` 提升

结论：当前 `torch.compile` 对这个链路收益很小，不应作为第一阶段主优化。

### 7. VAE channels-last 初测

`--vae_channels_last_3d` 已加为实验项，但当前模型直接应用会失败：

```text
RuntimeError('required rank 5 tensor to use channels_last_3d format')
```

结论：不能用简单 `module.to(memory_format=torch.channels_last_3d)` 方式优化 VAE；如果要继续做，需要更细粒度地只处理 Conv3D 输入/权重，不作为当前优先项。

## 第一阶段当前判断

第一阶段已经有一个更优先的可跑通实时档：

- `realtime`
- `160x576`
- `27.49 FPS` 常驻 runtime，不含模型加载
- 不回原分辨率输出深度

是否可交付取决于质量：

- 如果 `160x576` 深度对 2D 转 3D 效果可接受，第一阶段主线就是补齐服务化和输出协议。
- 如果 `160x576` 质量不够，下一优先级是把 `throughput 192x704` 从 `19.36 FPS` 提升到 25 FPS。

## 下一步

1. 人工检查 `output/stage1_contact_sheet.png`、`output/stage1_realtime_preview_color_depth_vis.mp4` 和下游 2D 转 3D 效果。
2. 基于 `tools/stage1_batch_runner.py` 固定第一阶段输入输出协议。
3. 安装 TensorRT/ONNX 环境，尝试固定 shape TensorRT FP16。
4. 如果 `throughput` 质量明显更好，优先针对 `192x704` 做 TensorRT/INT8/关键帧传播。
5. 如果低分辨率都不满足质量，启动 DVD teacher 蒸馏轻量模型。
