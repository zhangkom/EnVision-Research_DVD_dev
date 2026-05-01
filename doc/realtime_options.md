# DVD 实时化方案清单

目标：在允许量化、编译、降分辨率、混合推理、蒸馏或替换实时子模型的前提下，让 2D 转 3D 深度图生产达到 25 FPS。

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
| P5 | TensorRT INT8 PTQ/QAT | 高 | 高 | 中到高 | 需要校准集和质量评估 |
| P6 | DVD 关键帧 + 深度插帧/传播 | 高 | 高 | 中 | 让 DVD 不必每帧都跑 |
| P7 | DVD teacher 蒸馏轻量实时模型 | 最高 | 很高 | 可控 | 最像最终产品路线 |
| P8 | 更换/升级目标 GPU | 高 | 采购成本 | 低 | 如果硬件可变，这是最直接的性能杠杆 |

## 推荐试验顺序

### 1. 在 Quadro RTX 6000 Turing 上做真实基线

先不要量化，先确认目标机的真实差距。

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
  --presets balanced throughput realtime-preview
```

判定：

- 如果 `realtime-preview` >= 25 FPS，先进入质量评估。
- 如果 `realtime-preview` < 25 FPS，记录还差几倍，再试更激进的尺寸或进入 TensorRT/蒸馏路线。

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

这是 Quadro RTX 6000 Turing 上比 GGUF/FP8 更值得试的编译路线。

试验目标：

- 先导出或 trace VAE 子模块，验证 TensorRT FP16 可行性。
- 再尝试 DiT 的固定 shape engine。
- 固定实时档 shape，例如 `144x512`、`192x704`，减少动态 shape 复杂度。

风险：

- 当前 pipeline 不是标准 diffusers 单模型，导出工作量可能较大。
- 动态时间长度、窗口切片、额外条件输入需要固定化。
- Windows 上 TensorRT 环境和 engine 构建可能更麻烦，Linux 更适合。

### 6. TensorRT INT8 量化

这是原始 DVD 路线里最可能带来大幅加速的量化方向，但也最需要质量评估。

建议顺序：

1. 准备 3-5 条代表性视频作为校准集。
2. 用 FP16 DVD 质量档生成 teacher depth。
3. 对 VAE 或部分线性层先做 PTQ 试验。
4. 对 DiT 做 INT8 calibration。
5. 对比深度稳定性和下游 2D 转 3D 效果。

不建议：

- 在 Quadro RTX 6000 Turing 上优先 FP8。
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

1. 在 Quadro 上跑 `realtime_sweep.py --decode_resize`。
2. 增加 `96x384` 极限速度档，确认目标卡速度天花板。
3. 增加 `--no_resize_back`，测试下游能否吃低分辨率深度。
4. 保存每个档位的深度视频和 raw depth，人工判断质量底线。

第一轮结束后再决定：

- 如果 Quadro 的极限预览档接近或超过 25 FPS：继续做链路优化和质量修补。
- 如果 Quadro 明显低于 25 FPS：直接并行启动 TensorRT FP16 和蒸馏轻量模型。
