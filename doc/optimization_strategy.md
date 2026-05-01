# DVD 实时化与量化优化路线

本文记录 DVD 在 2D 转 3D 深度图生产里的优化判断，重点回答：还能不能继续优化、能不能实时、GGUF 和 FP8 是否值得投入。

## 结论

当前 DVD 模型基于 Wan2.1 1.3B 微调，主要计算在视频 DiT 和 VAE 上。它可以继续优化，但在目标卡 **Quadro RTX 6000 Turing** 上，不建议把主方向放在 GGUF 或 FP8。

更现实的路径是：

1. 用 FP16 跑通 Quadro RTX 6000 Turing 的基线。
2. 通过分辨率、窗口重叠和保存路径控制吞吐。
3. 如果需要进一步加速，优先做 TensorRT FP16 / INT8，而不是 GGUF / FP8。
4. 如果业务必须 25 FPS 实时，建议把 DVD 定位为离线高质量深度生成，同时另选一个轻量级深度模型做实时预览。

## 为什么不优先做 GGUF

GGUF 是面向 GGML 及其执行器的模型存储格式，官方说明里也把它定义为用于 GGML-based executors 的推理格式：<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

DVD 当前不是一个单纯的 LLM 权重加载问题，而是 PyTorch 里的视频扩散推理链路：

- Wan Video DiT
- Wan VAE / Conv3D 编解码
- LoRA merge
- 视频窗口切片与 overlap 对齐
- 深度后处理和视频/PNG/NPY 输出

把 safetensors 换成 GGUF 并不会自动让这些算子在 ggml/llama.cpp 里运行。真正要做 GGUF 加速，需要重写或适配整套 DiT、VAE、调度、窗口拼接和后处理执行器，成本很高，而且 Turing 上没有证据表明这会比 PyTorch/TensorRT FP16 更快。

## 为什么不优先做 FP8

目标卡 Quadro RTX 6000 是 Turing 架构。NVIDIA 官方页面说明 Quadro RTX 6000 powered by Turing，并列出 576 Tensor Cores、4608 CUDA cores、24 GB GDDR6：<https://www.nvidia.com/en-sg/products/workstations/quadro/rtx-6000/>

TensorRT-RTX 官方支持矩阵显示，Turing / compute capability 7.5 支持 FP32、FP16，但 BF16、INT8 WoQ GEMM、INT4 WoQ GEMM、FP8 GEMM、FP4 GEMM 都是 No：<https://docs.nvidia.com/deeplearning/tensorrt-rtx/v1.1/getting-started/support-matrix.html>

所以在 Quadro RTX 6000 Turing 上：

- FP8 不会变成硬件原生 Tensor Core 快路径。
- FP8 很可能需要反量化或走兼容路径，反而拖慢。
- DVD 这种视频扩散/深度模型也需要量化校准，盲目 FP8 风险高。

FP8 更适合在有原生 FP8 支持的新架构上评估，例如 Ada / Hopper / Blackwell，并且要配合 TensorRT、PyTorch FP8 或专门量化库做精度校准。

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

Quadro RTX 6000 Turing 比 RTX 4090 早两代，且没有 Ada 的第四代 Tensor Cores。即使极限预览档在 4090 上超过 25 FPS，也不能直接推断 Quadro 上能实时。实际结论需要在目标机上跑下面的 preset。

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

## 后续优化优先级

### P0：在 Quadro 目标机复测

先跑 `balanced`、`throughput`、`realtime-preview` 三个档位，拿到真实 FPS 和显存峰值。判断是否能实时必须以目标机实测为准。

### P1：批处理时避免重复加载模型

如果后续是批量视频处理，模型加载约 10 秒可以摊掉。需要新增一个 batch runner：单次加载模型，循环处理多个视频。

### P2：TensorRT FP16

比 GGUF 更值得投入。优先尝试把 DiT 和 VAE 的稳定子图导出/编译为 TensorRT FP16。Turing 支持 FP16，但要注意 TensorRT-RTX 文档提到 Turing 需要构建针对设备的 engine。

### P3：INT8 量化

INT8 需要校准集和质量评估。可以先从 VAE 或部分线性层做 PTQ 实验，但 DiT 里的 attention、norm、时间条件分支对量化更敏感，需要逐段验证深度图稳定性。

### P4：换轻量实时模型

如果业务的硬要求是 25 FPS、原分辨率或接近原分辨率实时深度，当前 DVD 1.3B 路线不合适。建议架构上拆成：

- 实时预览：轻量单目/视频深度模型。
- 离线高质量：DVD 生成高一致性深度图。

这样能让项目在交互端可实时，同时保留 DVD 的高质量输出能力。
