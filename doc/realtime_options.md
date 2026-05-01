# DVD 实时化方案清单

目标：在允许量化、编译、降分辨率、混合推理、蒸馏或替换实时子模型的前提下，让 2D 转 3D 深度图生产达到 25 FPS。

当前分两个阶段推进：

- 第一阶段：只以本机 RTX 4090 作为优化和交付目标。
- 第二阶段：第一阶段稳定后，再迁移到 Quadro RTX 6000 Turing。

详细阶段计划见 `doc/phase_plan.md`。

实时指标建议统一为：

- 模型服务常驻，不计一次性模型加载时间。
- 端到端处理速度 >= 输入视频 FPS，当前测试视频为 25 FPS。
- 记录 `end_to_end_fps_excluding_model_load`、显存峰值、输出深度质量和下游 2D 转 3D 视觉效果。

## 方案总览

| 优先级 | 方案 | 预期收益 | 工程成本 | 质量风险 | 备注 |
| --- | --- | --- | --- | --- | --- |
| P0 | 低分辨率实时档 + decode_resize | 中到高 | 低 | 中到高 | 已跑通，4090 上极限预览档超过 25 FPS |
| P1 | 减少非必要后处理和输出 | 低到中 | 低 | 低 | 如果下游可直接吃低分辨率深度，可省 resize/save |
| P2 | 常驻服务 + CPU/GPU 流水线 | 低到中 | 中 | 低 | 避免重复加载，重叠解码、推理、保存 |
| P3 | PyTorch 层优化 | 低到中 | 中 | 低到中 | `torch.compile`、SDPA/xFormers、算子替换 |
| P4 | TensorRT FP16 | 中到高 | 高 | 中 | 比 GGUF 更适合当前链路 |
| P5 | 4090 FP8 / TensorRT INT8 | 高 | 高 | 中到高 | 4090 可试 FP8；INT8 需要校准集和质量评估 |
| P6 | DVD 关键帧 + 深度插帧/传播 | 高 | 高 | 中 | 让 DVD 不必每帧都跑 |
| P7 | DVD teacher 蒸馏轻量实时模型 | 最高 | 很高 | 可控 | 最像最终产品路线 |
| P8 | 更换/升级目标 GPU | 高 | 采购成本 | 低 | 如果硬件可变，这是最直接的性能杠杆 |

## 为什么不把 GGUF + FP8 绑定作为主方案

这个方向不是完全不能研究，而是**不适合作为第一阶段 4090 实时化的主路线，也不适合作为第二阶段 Quadro RTX 6000 Turing 的迁移路线**。原因分四层：

### 1. GGUF 只是格式，不是 DVD 的执行器

GGUF 是面向 GGML-based executors 的模型格式，适合 GGML/llama.cpp 这类执行链路：<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

DVD 不是单个 LLM，也不是只包含 dense matmul 的文本模型。它当前是 PyTorch 视频扩散/深度估计链路，包含：

- Wan Video DiT
- Wan VAE / Conv3D 编解码
- LoRA merge
- 视频窗口切片
- overlap 深度对齐和拼接
- 深度 resize、保存和下游输出

把 `safetensors` 转成 GGUF 并不会让这些 PyTorch module 自动在 GGML 里运行。真要走 GGUF，需要重写或适配 DiT、VAE、attention、Conv3D、调度、窗口拼接和后处理执行器，工作量接近重新做一套推理框架。

### 2. FP8 可以在 4090 阶段试，但不能和 GGUF 绑定

第一阶段的 4090 不受 Turing 限制，FP8 可以作为 TensorRT/PyTorch 的实验项。但这和 GGUF 是两回事：

- FP8 是低精度计算/权重量化方向。
- GGUF 是 GGML 执行器使用的模型格式。
- DVD 当前的执行瓶颈是 PyTorch/TensorRT 能否高效跑 DiT/VAE，而不是权重文件扩展名。

所以第一阶段可以试 FP8，但应作为 TensorRT/PyTorch FP8 实验，不能预设为“GGUF + FP8”。

### 3. Quadro RTX 6000 Turing 没有原生 FP8 快路径

第二阶段目标卡 Quadro RTX 6000 是 Turing 架构。NVIDIA 官方规格说明它是 Turing GPU：<https://www.nvidia.com/en-sg/products/workstations/quadro/rtx-6000/>

TensorRT-RTX 官方支持矩阵显示，Turing / compute capability 7.5 支持 FP32、FP16，但 `FP8 GEMM` 是 `No`：<https://docs.nvidia.com/deeplearning/tensorrt-rtx/v1.1/getting-started/support-matrix.html>

因此在这张卡上把底层从 FP16 换成 FP8，通常不会得到 FP8 Tensor Core 加速。实际执行很可能需要反量化、转换或回退到 FP16/FP32 路径，速度不一定提升，甚至可能变慢。

### 4. FP8 不是无损压缩，会影响深度稳定性

DVD 的输出不是分类标签，而是连续深度图，还要求视频时间一致性。FP8/低比特量化会影响：

- 深度尺度稳定性
- overlap 区域对齐
- 相邻帧闪烁
- 前后景深度排序
- 下游 2D 转 3D 的遮挡和形变

所以即使换到支持 FP8 的新 GPU，也需要校准集、teacher 对比和下游视觉评估，不能直接把 FP16 权重替换成 FP8 就上线。

### 5. 更适合本项目的低精度路线是 TensorRT FP16 / INT8

第一阶段 4090 优先顺序：

1. PyTorch FP16 基线。
2. 固定 shape + TensorRT FP16。
3. TensorRT/PyTorch FP8 实验。
4. 有校准集的 TensorRT INT8。
5. 如果仍不够，再做 DVD teacher 蒸馏轻量实时模型。

第二阶段 Quadro RTX 6000 Turing 优先顺序：

1. PyTorch FP16 基线。
2. 固定 shape + TensorRT FP16。
3. 有校准集的 TensorRT INT8。
4. 如果仍不够，再做 DVD teacher 蒸馏轻量实时模型。

GGUF + FP8 至少踩中“执行器不匹配”这个问题；迁移到 6000 时还会额外踩中“目标硬件不支持原生 FP8”问题。因此它不适合作为主方案。

## 推荐试验顺序

### 1. 在 RTX 4090 上固定第一阶段基线

第一阶段先不要考虑 Quadro RTX 6000 Turing，先确认 4090 上的速度/质量边界。

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

判定：

- 如果 `realtime-preview` >= 25 FPS，先进入质量评估。
- 如果 `throughput` 或更高质量档接近 25 FPS，优先尝试链路优化和 TensorRT FP16。
- 如果低分辨率质量不够，直接启动 teacher 蒸馏或关键帧传播路线。

### 2. 做分辨率和 overlap 矩阵

当前最敏感的速度旋钮是空间分辨率和 overlap。

建议矩阵：

| 档位 | height | width | overlap | 用途 |
| --- | ---: | ---: | ---: | --- |
| 质量底线 | 256 | 640 | 21 | 保留更多细节 |
| 吞吐 | 192 | 640 | 9 | 速度/质量折中 |
| 极限预览 | 128 | 512 | 9 | 冲实时 |
| 极限速度 | 96 | 384 | 9 | 只用于判断速度天花板 |

如果 `overlap=9` 出现明显深度跳变，试 `overlap=13` 或 `overlap=21`。

### 3. 减少实时链路里的非必要工作

如果下游 2D 转 3D 可以接受低分辨率深度，再统一上采样，实时链路可以跳过部分后处理。

待试改动：

- 增加 `--no_resize_back`：直接输出推理尺寸深度。
- 增加 raw depth mmap/NPY 批量输出，减少逐帧 PNG 和视频编码。
- 推理服务常驻：一次加载模型，连续处理多个视频或流式片段。
- CPU 解码、GPU 推理、输出保存做 producer/consumer 流水线。

这类优化通常不能把 10 FPS 变成 25 FPS，但能把已经接近实时的档位推过线。

### 4. PyTorch 编译和注意力优化

目标：在不改变模型精度和输出的前提下提高吞吐。

待试项：

- `torch.inference_mode()`：已加入。
- `torch.compile`：优先在 Linux 上试；Windows 上可能受 PyTorch/Triton 支持限制。
- 检查 DiT attention 是否能走 PyTorch SDPA 或 xFormers。
- 检查 VAE Conv3D 是否存在 channels-last、固定 shape、静态图优化空间。

判定：

- 如果收益 < 10%，不要在这条线上消耗太久。
- 如果某个固定 shape 能明显加速，再把实时 preset 固定成少数几个 engine/profile。

### 5. TensorRT FP16

这是第一阶段 4090 和第二阶段 6000 都值得试的编译路线。

试验目标：

- 先导出或 trace VAE 子模块，验证 TensorRT FP16 可行性。
- 再尝试 DiT 的固定 shape engine。
- 固定实时档 shape，例如 `144x512`、`192x704`，减少动态 shape 复杂度。

风险：

- 当前 pipeline 不是标准 diffusers 单模型，导出工作量可能较大。
- 动态时间长度、窗口切片、额外条件输入需要固定化。
- Windows 上 TensorRT 环境和 engine 构建可能更麻烦，Linux 更适合。

### 6. 4090 FP8 与 TensorRT INT8 量化

第一阶段可以把 FP8 加入实验，但它必须走 TensorRT/PyTorch 路线，不能绑定 GGUF。INT8 仍然是更通用的迁移路线，因为第二阶段 6000 可以继续评估 INT8。

建议顺序：

1. 准备 3-5 条代表性视频作为校准集。
2. 用 FP16 DVD 质量档生成 teacher depth。
3. 在 4090 上尝试 TensorRT/PyTorch FP8，并和 FP16 teacher 对比。
4. 对 VAE 或部分线性层先做 INT8 PTQ 试验。
5. 对 DiT 做 INT8 calibration。
6. 对比深度稳定性和下游 2D 转 3D 效果。

不建议：

- 把 FP8 和 GGUF 绑定。
- 只做 PyTorch 动态量化就期待 CUDA 推理显著变快。
- 没有校准集就直接接受 INT8 输出。

### 7. DVD 关键帧 + 深度插帧/传播

如果全帧 DVD 太慢，可以让 DVD 只处理关键帧或低 FPS 深度，再把深度传播到中间帧。

候选路线：

- DVD 每 2 帧跑一次，深度插帧到 25 FPS。
- DVD 每 3-4 帧跑一次，用光流或视频运动估计传播深度。
- 轻量模型每帧跑，DVD 关键帧修正尺度和结构。

适合场景：

- 镜头运动平滑。
- 画面中快速遮挡不多。
- 下游 2D 转 3D 对偶发局部误差容忍度较高。

### 8. DVD teacher 蒸馏轻量实时模型

这是最可能形成稳定实时产品的路线。

做法：

1. 用 DVD 质量档或平衡档批量生成训练视频的深度 teacher。
2. 训练轻量学生模型，输入 RGB 帧或短视频片段，输出深度。
3. 加 temporal consistency loss，约束闪烁。
4. 导出 ONNX/TensorRT FP16。
5. 做 INT8 calibration。

优点：

- 实时概率最高。
- 可以针对你的 2D 转 3D 视频分布优化。
- 可以保留 DVD 作为离线高质量模式。

代价：

- 需要训练数据和训练流程。
- 需要定义质量门槛。
- 一开始不能保证泛化，需要持续补数据。

## 第一轮建议

第一轮只做 4 件事：

1. 在 4090 上跑 `realtime_sweep.py --decode_resize`。
2. 增加 `96x384` 极限速度档，确认目标卡速度天花板。
3. 增加 `--no_resize_back`，测试下游能否吃低分辨率深度。
4. 保存每个档位的深度视频和 raw depth，人工判断质量底线。

第一轮结束后再决定：

- 如果 4090 的 `realtime-preview` 画质可接受：继续做链路优化和质量修补。
- 如果 4090 的低分辨率画质不可接受：并行启动 TensorRT FP16、FP8 实验和蒸馏轻量模型。
- 第一阶段稳定后，再进入 Quadro RTX 6000 Turing 迁移。
