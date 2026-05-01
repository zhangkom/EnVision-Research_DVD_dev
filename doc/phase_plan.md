# DVD 两阶段实时化计划

## 阶段定义

### 第一阶段：RTX 4090 达到可交付实时

第一阶段只以本机 RTX 4090 作为优化目标，不再受 Quadro RTX 6000 Turing 的限制。

验收目标：

- 输入视频 25 FPS。
- 模型服务常驻，不计一次性模型加载。
- `end_to_end_fps_excluding_model_load >= 25`。
- 输出深度图能通过 2D 转 3D 的人工质量检查。
- 每个方案都记录 FPS、显存峰值、输出路径和质量结论。

当前 4090 已知结果：

- `realtime-preview + decode_resize`，实际处理尺寸 `144x512`，1000 帧端到端不含模型加载 `35.11 FPS`。
- `realtime + decode_resize + no_resize_back`，实际处理尺寸 `160x576`，1000 帧常驻 batch runtime `27.49 FPS`。
- `realtime-preview + decode_resize + no_resize_back`，常驻 batch runtime `36.28 FPS`。
- `speed-floor + decode_resize + no_resize_back`，常驻 batch runtime `61.76 FPS`。
- 深拆后，`model.pipe` 窗口推理占 inference `93.4%`，是第一瓶颈。
- overlap 对齐约占 inference `6.4%`，不是主瓶颈。

第一阶段优化重点：

1. 先确认 `160x576` 和 `144x512` 实时档的 2D 转 3D 质量是否能接受。
2. 增加 `--no_resize_back`，让下游直接吃低分辨率深度或统一后处理。
3. 跑完整分辨率矩阵：`balanced`、`throughput`、`realtime-preview`、`speed-floor`。
4. 尝试 PyTorch 层优化：固定 shape、`torch.compile`、SDPA/xFormers、VAE channels-last。
5. 尝试 TensorRT FP16 固定 shape engine。
6. 在 4090 上把 FP8 作为实验项评估，但只走 TensorRT/PyTorch FP8，不走 GGUF。
7. 如果画质不够或速度余量不够，启动 DVD teacher 蒸馏轻量实时模型。

第一阶段不做：

- 不以 Quadro RTX 6000 Turing 的能力作为限制。
- 不把 GGUF 当成主路线。
- 不为 6000 的 FP8 不支持问题牺牲 4090 阶段的实验空间。

### 第二阶段：迁移到 Quadro RTX 6000 Turing

第二阶段在第一阶段方案稳定后，再评估迁移到 Quadro RTX 6000 Turing。

第二阶段目标：

- 复用第一阶段的输入输出协议、质量门槛和 benchmark 工具。
- 跑相同 preset，确认真实速度差距。
- 对不能迁移的优化做降级策略。

第二阶段注意点：

- Quadro RTX 6000 Turing 不支持原生 FP8 GEMM 快路径。
- 优先使用 FP16、TensorRT FP16、INT8 calibration。
- 如果第一阶段依赖 4090 FP8，第二阶段需要准备 FP16/INT8 替代 engine。
- 如果 6000 无法达到实时，则采用轻量蒸馏模型或降级 preset。

## 第一阶段推荐执行顺序

### 1. 固定 4090 基线

```powershell
conda run -n dvd python tools\realtime_sweep.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --target_fps 25 `
  --decode_resize `
  --presets balanced throughput realtime realtime-preview speed-floor
```

### 2. 输出质量样本

每个候选 preset 至少输出：

- 深度可视化视频。
- raw depth `.npy`。
- 16-bit PNG 深度帧。
- benchmark JSON。

用于比较：

- 闪烁。
- 边缘。
- 前后景排序。
- 下游 2D 转 3D 效果。

已补充 `tools/stage1_contact_sheet.py`，可把源视频和多个深度结果按固定帧位拼成 `output/stage1_contact_sheet.png` 做快速审片。

### 3. 去掉不必要后处理

已实现：

- `--no_resize_back`
- 可选只输出推理尺寸深度。
- 可选跳过可视化视频，只输出 raw depth。

目标是把已经超过 25 FPS 的低分辨率档位留出更多速度余量。

当前结果见 `doc/stage1_results.md`。

### 4. 常驻 batch 形态

已实现：

- `tools/stage1_batch_runner.py`
- 一次加载模型，连续处理多个视频或 preset。
- per-job 记录 decode、inference、resize-back、save、runtime FPS。
- 支持 `--save_video`、`--save_depth_npy`、`--save_depth_png16` 输出下游所需深度。

当前常驻测试：

| preset | depth size | runtime FPS |
| --- | --- | ---: |
| realtime | 160x576 | 27.49 |
| realtime-preview | 144x512 | 36.28 |
| speed-floor | 112x384 | 61.76 |

### 5. 编译和量化

优先顺序：

1. PyTorch FP16 固定 shape。
2. TensorRT FP16。
3. 4090 FP8 实验。
4. TensorRT INT8。
5. 蒸馏轻量模型。

FP8 只作为 4090 第一阶段实验项。若启用，必须和 FP16 teacher 对比深度稳定性，不能只看速度。

当前环境检测结果：

- `torch.compile` 可用，但初测收益很小。
- TensorRT / torch_tensorrt / ONNX 当前 conda 环境不可见，需要安装后再做 FP16/FP8 engine 实验。
- torch float8 dtype 可见，但还缺真实 DVD FP8 量化/导出路径。

## 第二阶段迁移检查表

第一阶段完成后，在 Quadro RTX 6000 Turing 上跑：

```powershell
conda run -n dvd python tools\check_runtime_capability.py
```

```powershell
conda run -n dvd python tools\realtime_sweep.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --target_fps 25 `
  --decode_resize `
  --presets balanced throughput realtime-preview speed-floor
```

迁移判定：

- 如果 6000 FP16 低分辨率档能过 25 FPS：沿用第一阶段方案。
- 如果 6000 接近 25 FPS：优先做 TensorRT FP16/INT8。
- 如果 6000 明显低于 25 FPS：部署蒸馏轻量模型，DVD 保留为离线高质量 teacher。
