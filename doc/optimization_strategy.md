# DVD 实时化与量化优化路线

本文记录 DVD 在 2D 转 3D 深度图生产里的优化判断，重点回答：还能不能继续优化、能不能实时、GGUF 和 FP8 是否值得投入。

## 结论

当前最终目标调整为：**在允许量化、编译、降分辨率、蒸馏或替换实时子模型的前提下，让 2D 转 3D 的深度图生成达到 25 FPS 实时能力**。模型服务常驻时，实时指标按端到端处理速度计算，不把一次性模型加载时间计入每条视频。

当前分两个阶段推进：第一阶段只以本机 **RTX 4090** 作为优化和交付目标；第二阶段在第一阶段稳定后，再迁移到 **Quadro RTX 6000 Turing**。详细计划见 `doc/phase_plan.md`。

当前 DVD 模型基于 Wan2.1 1.3B 微调，主要计算在视频 DiT 和 VAE 上。第一阶段不受 Turing 限制，可以把 4090 FP8 作为实验项；但 GGUF 仍不适合作为主路线，因为它不能直接承接当前 PyTorch 视频扩散链路。

更现实的路径是：

1. 第一阶段用 4090 固定实时基线和质量门槛。
2. 通过分辨率、窗口重叠和保存路径控制吞吐。
3. 优先做 TensorRT FP16；4090 上可实验 FP8；INT8 用校准集验证。
4. 如果原始 DVD 1.3B 在目标质量下无法 25 FPS，则把 DVD 作为 teacher / 高质量离线基线，用轻量模型、蒸馏模型或 TensorRT INT8 引擎承接实时路径。
5. 第二阶段再评估 Quadro RTX 6000 Turing，并准备 FP16/INT8 降级方案。

## 为什么不优先做 GGUF

GGUF 是面向 GGML 及其执行器的模型存储格式，官方说明里也把它定义为用于 GGML-based executors 的推理格式：<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

DVD 当前不是一个单纯的 LLM 权重加载问题，而是 PyTorch 里的视频扩散推理链路：

- Wan Video DiT
- Wan VAE / Conv3D 编解码
- LoRA merge
- 视频窗口切片与 overlap 对齐
- 深度后处理和视频/PNG/NPY 输出

把 safetensors 换成 GGUF 并不会自动让这些算子在 ggml/llama.cpp 里运行。真正要做 GGUF 加速，需要重写或适配整套 DiT、VAE、调度、窗口拼接和后处理执行器，成本很高，而且 Turing 上没有证据表明这会比 PyTorch/TensorRT FP16 更快。

## 为什么不把 FP8 作为跨阶段主线

第一阶段 4090 可以把 FP8 作为 TensorRT/PyTorch 实验项，但不能把 FP8 和 GGUF 绑定。第二阶段目标卡 Quadro RTX 6000 是 Turing 架构。NVIDIA 官方页面说明 Quadro RTX 6000 powered by Turing，并列出 576 Tensor Cores、4608 CUDA cores、24 GB GDDR6：<https://www.nvidia.com/en-sg/products/workstations/quadro/rtx-6000/>

TensorRT-RTX 官方支持矩阵显示，Turing / compute capability 7.5 支持 FP32、FP16，但 BF16、INT8 WoQ GEMM、INT4 WoQ GEMM、FP8 GEMM、FP4 GEMM 都是 No：<https://docs.nvidia.com/deeplearning/tensorrt-rtx/v1.1/getting-started/support-matrix.html>

所以在 Quadro RTX 6000 Turing 上：

- FP8 不会变成硬件原生 Tensor Core 快路径。
- FP8 很可能需要反量化或走兼容路径，反而拖慢。
- DVD 这种视频扩散/深度模型也需要量化校准，盲目 FP8 风险高。

FP8 更适合在有原生 FP8 支持的新架构上评估，例如第一阶段的 4090/Ada 或 Hopper/Blackwell，并且要配合 TensorRT、PyTorch FP8 或专门量化库做精度校准。

## 当前实测基线

测试视频：`test_video/depth_full_50frame.mp4`

实际视频信息：

- 1000 frames
- 25 FPS
- 1858x518

本机 RTX 4090 实测结果：

| 档位 | 实际处理尺寸 | overlap | 保存 | 纯推理 FPS | 端到端 FPS，不含模型加载 | 备注 |
| --- | --- | --- | --- | ---: | ---: | --- |
| 质量档 | 480x1728 | 21 | no_save | 3.13 | - | 画质优先，离实时很远 |
| 平衡档 | 256x928 | 21 | fast_video_save | 9.42 | 8.09 | 完整视频可用吞吐档 |
| 吞吐档 | 256x928 | 9 | no_save | 11.66 | - | overlap 降低，需检查闪烁/跳变 |
| 预览档 | 192x704 | 9 | no_save | 19.89 | 17.36 | 接近实时，但 4090 仍未到 25 FPS |
| 极限预览档 | 144x512 | 9 | no_save | 38.71 | 32.27 | 4090 可实时，但质量需要重点确认 |
| 极限预览档 + decode_resize | 144x512 | 9 | no_save | 38.56 | 34.77 | 解码时缩放，4090 端到端超过 25 FPS |

第一阶段以 RTX 4090 为准：极限预览档已经超过 25 FPS，下一步重点是质量评估、减少后处理、TensorRT/FP8/INT8 实验。第二阶段再单独评估 Quadro RTX 6000 Turing。

`decode_resize` 后，极限预览档的视频读入/缩放从约 `3.77s` 降到 `1.61s`，端到端不含模型加载从 `32.27 FPS` 提升到 `34.77 FPS`。

## 推荐 preset

`tools/benchmark_single_video.py` 增加了 `--preset`，用于固定常用档位。

质量档：

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --preset quality `
  --dtype fp16 `
  --fast_video_save
```

平衡档：

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --preset balanced `
  --dtype fp16 `
  --fast_video_save
```

吞吐档：

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --preset throughput `
  --dtype fp16 `
  --no_save
```

极限预览档：

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --preset realtime-preview `
  --dtype fp16 `
  --no_save
```

如果要覆盖 preset 的某个参数，直接额外传参数即可，例如：

```powershell
conda run -n dvd python tools\benchmark_single_video.py `
  --preset throughput `
  --overlap 21 `
  --dtype fp16 `
  --no_save
```

## 实时 sweep

先在目标机上检查 CUDA/GPU 能力：

```powershell
conda run -n dvd python tools\check_runtime_capability.py
```

`tools/realtime_sweep.py` 可以一次跑多个 preset，并输出 realtime ratio、达到 25 FPS 还需要的加速倍数、显存峰值和 Markdown 汇总。

4090 第一阶段建议先跑：

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

快速冒烟测试可以只跑 9 帧：

```powershell
conda run -n dvd python tools\realtime_sweep.py `
  --weights C:\work\workspace_own\workspace_dvd\ckpt\model.safetensors `
  --local_model_path C:\work\workspace_own\workspace_dvd\models `
  --input_video test_video\depth_full_50frame.mp4 `
  --output_dir output `
  --target_fps 25 `
  --max_frames 9 `
  --decode_resize `
  --presets realtime-preview
```

`--decode_resize` 会在视频解码时直接缩放到推理尺寸，避免先构建完整原分辨率 tensor 再 resize。低分辨率实时档推荐打开它。

`benchmark_single_video.py` 的 JSON 里现在会直接记录：

- `target_fps`
- `inference_realtime_ratio`
- `end_to_end_realtime_ratio_excluding_model_load`
- `required_inference_speedup_to_target`
- `required_end_to_end_speedup_to_target`
- `realtime_met_inference_only`
- `realtime_met_excluding_model_load`

## 后续优化优先级

### P0：在 RTX 4090 上固定阶段一基线

先跑 `balanced`、`throughput`、`realtime-preview`、`speed-floor` 四个档位，拿到 4090 的真实 FPS、显存峰值和质量样本。阶段一先不考虑 Quadro RTX 6000 Turing。

### P1：定义质量门槛

实时方案不能只看 FPS。建议选 3-5 条代表性视频，保存原始 DVD 质量档深度作为 teacher，再比较实时档输出：

- 闪烁和跳变：相邻帧深度变化是否稳定。
- 边缘质量：人物、前景、道具边界是否糊掉。
- 深度排序：前后景是否反转。
- 下游 2D 转 3D 效果：立体感、遮挡、变形是否可接受。

只要实时模型能通过这些业务门槛，它不必和原始 DVD 每个像素完全一致。

### P2：批处理时避免重复加载模型

如果后续是批量视频处理，模型加载约 10 秒可以摊掉。需要新增一个 batch runner：单次加载模型，循环处理多个视频。

### P3：TensorRT FP16 / 4090 FP8

比 GGUF 更值得投入。优先尝试把 DiT 和 VAE 的稳定子图导出/编译为 TensorRT FP16。4090 阶段可以增加 FP8 实验，但必须以 FP16 teacher 做质量对比。

### P4：INT8 量化

INT8 需要校准集和质量评估。可以先从 VAE 或部分线性层做 PTQ 实验，但 DiT 里的 attention、norm、时间条件分支对量化更敏感，需要逐段验证深度图稳定性。

### P5：蒸馏或换轻量实时模型

如果业务的硬要求是 25 FPS、原分辨率或接近原分辨率实时深度，当前 DVD 1.3B 原模型可能不合适。建议架构上拆成：

- 实时路径：轻量单目/视频深度模型，或用 DVD 生成数据后蒸馏出的学生模型。
- 离线路径：DVD 生成高一致性深度图，作为高质量输出和 teacher。

这样能让项目在交互端可实时，同时保留 DVD 的高质量输出能力。
