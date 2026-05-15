# 车道线二分类感知项目上下文

## 1. 项目目标

本项目面向自动驾驶基础感知验证，目标不是精确检测车道线形状，而是判断车辆当前位置或假设位置是否处于有效车道区域内。

任务定义：

```text
输入：车载前视图像中的车辆前方 ROI
输出：lane_out / lane_in 二分类
```

类别定义：

```text
0 = lane_out
1 = lane_in
```

模型主线采用轻量 CNN 分类器 `LaneCNN`。车道线/地图信息只用于离线生成标签和构造样本，不作为模型推理输入。

## 2. 当前数据方案

当前实现使用 `nuScenes v1.0-mini`，原因是数据量比 TuSimple、CULane 等完整车道线数据集小，更适合面试项目快速开发和演示。

nuScenes mini 不是专门的车道线检测数据集，但提供：

- `CAM_FRONT` 前视图像
- ego pose
- HD map / map expansion layers

因此当前标签生成方式是：

```text
CAM_FRONT 图像 + ego pose + HD map
-> 自动生成 lane_in / lane_out ROI 样本
```

需要说明的是，这与 TuSimple/CULane 的“图像车道线点标注 -> 左右边界 -> 二分类”不同。当前实现更偏自动驾驶工程中的地图辅助标签生成。

## 3. 标签生成设计

数据处理脚本：

```text
scripts/prepare_nuscenes_binary.py
```

输出：

```text
data/train.csv
data/val.csv
data/summary.txt
```

CSV 字段：

```csv
image_path,roi_x1,roi_y1,roi_x2,roi_y2,label,lateral_offset_m,location,scene_token,sample_token
```

### 3.1 Ego Footprint

`footprint` 指车辆在地面上的虚拟占用矩形。

当前默认车辆尺寸：

```text
length = 4.6m
width  = 1.9m
```

处理流程：

```text
1. 在 ego 坐标系中构造车辆矩形 footprint
2. 根据 ego pose 将 footprint 转到 global/map 坐标系
3. 将 nuScenes map 中 lane / lane_connector / drivable_area 提取为 polygon geometry
4. 计算 footprint polygon 与 map polygon 的面积重叠比例
```

标签规则：

```text
overlap_ratio = area(footprint ∩ map_polygon) / area(footprint)

overlap_ratio >= 0.8 -> lane_in
overlap_ratio <= 0.5 -> lane_out
其他情况             -> ignore
```

使用面积法而不是单点查询的原因：

- 车辆不是一个点，车身可能已经压线但 ego center 仍在车道内
- 面积重叠比单点判断更稳定
- `ignore` 区间可以过滤边界附近模糊样本

### 3.2 偏移 ROI 与负样本

真实自动驾驶数据多数是正常驾驶，中心 ROI 大概率都是 `lane_in`。如果只用真实 ego pose，会导致 `lane_out` 样本不足。

当前通过横向偏移构造候选样本：

```text
0m
±0.8m
±1.6m
±2.4m
```

对每个偏移：

```text
偏移 ego footprint
-> 查询 map overlap_ratio 得到标签
-> 同步偏移图像 ROI
```

注意：不能只偏移地图标签而不偏移图像 ROI，否则会产生“图像看起来在车道内，但标签是 lane_out”的错配样本。

### 3.3 前方 Corridor 过滤

模型输入是车辆前方 ROI，而 footprint 描述的是车辆当前地面占用区域。两者存在轻微语义错位。

因此脚本支持可选：

```bash
--check-front-corridor
```

它会检查车辆前方若干点是否仍在可行驶区域内。该逻辑只用于过滤 `lane_in` 样本，不作为主标签来源。

## 4. 模型设计

当前主模型：

```text
LaneCNN
```

代码位置：

```text
src/lane_binary/model/lane_cnn.py
```

结构：

```text
Input: RGB ROI, 3 x 224 x 224

Conv-BN-ReLU-MaxPool: 3   -> 32
Conv-BN-ReLU-MaxPool: 32  -> 64
Conv-BN-ReLU-MaxPool: 64  -> 128
Conv-BN-ReLU:         128 -> 256
AdaptiveAvgPool2d
Dropout
Linear: 256 -> 2
```

输出：

```text
logits = [lane_out_logit, lane_in_logit]
```

`logits` 是模型未归一化的类别分数。训练时直接传给 `CrossEntropyLoss`，推理时再做 `softmax` 得到概率。

### 4.1 为什么不用 U-Net

U-Net 是语义分割网络，适合输出像素级 mask：

```text
input image -> H x W mask
```

本项目目标是 ROI 级二分类：

```text
input ROI -> lane_out / lane_in
```

因此轻量 CNN 分类器更直接。U-Net 会引入像素级 mask 标注、分割 loss 和后处理规则，对这个“简单二分类模型”任务来说过度设计。

U-Net 可以作为未来扩展：

```text
分割可行驶区域 / 车道线 mask -> 后处理得到 lane_in / lane_out
```

但当前主线不采用。

## 5. Loss 与指标

当前支持：

```text
ce
weighted_ce
focal
```

实现位置：

```text
src/lane_binary/utils/losses.py
```

默认建议：

```text
Weighted CrossEntropyLoss
```

原因是 `lane_in / lane_out` 通常不均衡，尤其真实行驶数据中 `lane_in` 更多。

Loss 选择不做自动切换，而是通过参数配置：

```bash
--loss ce
--loss weighted_ce
--loss focal --focal-gamma 2.0
```

类别权重可根据训练集统计自动计算：

```text
w_class = total_samples / (num_classes * class_count)
```

评估指标：

```text
accuracy
lane_out precision
lane_out recall
lane_out F1
lane_in precision
lane_in recall
lane_in F1
confusion matrix
```

其中 `lane_out recall` 最关键，因为该任务更关注车辆已经偏出车道时是否被漏检。

当前 best checkpoint 的选择分数：

```text
score = 0.7 * lane_out_recall + 0.3 * lane_out_f1
```

## 6. 数据增强

数据增强分为两类。

### 6.1 离线样本增强

由 `prepare_nuscenes_binary.py` 完成：

```text
中心 ROI
左偏移 ROI
右偏移 ROI
边界附近 hard negative
```

这是本任务最重要的增强方式，因为它直接解决 `lane_out` 样本不足的问题。

### 6.2 在线图像增强

实现位置：

```text
src/lane_binary/dataset/transforms.py
```

训练增强：

```text
Resize 224x224
ColorJitter: brightness / contrast / saturation
RandomHorizontalFlip
轻微 RandomAffine: 小角度旋转、小幅平移、轻微缩放
GaussianBlur
RandomErasing
Normalize
```

验证预处理：

```text
Resize 224x224
ToTensor
Normalize
```

不建议默认使用：

```text
大幅随机裁剪
大幅平移
大角度旋转
强透视变换
垂直翻转
MixUp / CutMix
```

原因是这些增强可能改变 ROI 与车道区域的几何关系，导致标签不可靠。

## 7. 当前代码结构

```text
.
├── CODEX.md
├── README.md
├── requirements.txt
├── scripts/
│   ├── prepare_nuscenes_binary.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── infer_onnx.py
└── src/
    └── lane_binary/
        ├── dataset/
        │   ├── lane_roi_dataset.py
        │   ├── transforms.py
        │   └── dataloader.py
        ├── model/
        │   └── lane_cnn.py
        └── utils/
            ├── losses.py
            ├── metrics.py
            ├── inference.py
            └── seed.py
```

## 8. 训练与评估

安装依赖：

```bash
pip install -r requirements.txt
```

生成数据：

```bash
python scripts/prepare_nuscenes_binary.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --output-dir data
```

训练：

```bash
python scripts/train.py \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --image-root /path/to/nuscenes \
  --output-dir checkpoints \
  --run-dir runs/lane_cnn \
  --epochs 30 \
  --batch-size 32 \
  --loss weighted_ce
```

TensorBoard：

```bash
tensorboard --logdir runs
```

评估：

```bash
python scripts/evaluate.py \
  --csv data/val.csv \
  --image-root /path/to/nuscenes \
  --checkpoint checkpoints/best.pt
```

## 9. ONNX 导出与推理

导出：

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/best.pt \
  --output onnx/lane_cnn.onnx
```

ONNX 输入：

```text
name: image
shape: [batch, 3, 224, 224]
```

ONNX 输出：

```text
name: logits
shape: [batch, 2]
```

ONNX Runtime 单图推理：

```bash
python scripts/infer_onnx.py \
  --onnx onnx/lane_cnn.onnx \
  --image /path/to/nuscenes/samples/CAM_FRONT/xxx.jpg \
  --roi 430 390 850 720
```

输出示例：

```json
{
  "class_id": 1,
  "class_name": "lane_in",
  "confidence": 0.93,
  "probabilities": {
    "lane_out": 0.07,
    "lane_in": 0.93
  }
}
```

## 10. FP32、FP16 与 INT8

当前默认交付：

```text
训练：FP32
ONNX 导出：FP32
ONNX Runtime 推理：FP32
```

这是最稳的基础版本，便于复现和评估。

### 10.1 FP16

FP16 是半精度浮点，不是 INT8 量化的中间步骤。

适用场景：

```text
GPU 推理
TensorRT
CUDAExecutionProvider
GPU 训练混合精度 AMP
```

典型流程：

```text
FP32 checkpoint -> FP32 ONNX -> FP16 ONNX / TensorRT FP16 engine
```

本项目的 `LaneCNN` 很小，FP16 收益不一定明显，因此不作为默认实现。

### 10.2 INT8

INT8 才是更典型的量化推理方案。

常见流程：

```text
FP32 checkpoint -> FP32 ONNX -> INT8 ONNX
```

通常不需要：

```text
FP32 -> FP16 -> INT8
```

INT8 静态量化需要代表性 calibration ROI 数据，并且必须回归评估：

```text
accuracy
lane_out recall
F1
confusion matrix
```

当前项目不默认实现 INT8，因为它会引入校准数据、量化策略和部署硬件适配，容易偏离面试题主线。可以作为后续部署优化加分项。

## 11. 当前工程状态

已完成：

```text
nuScenes mini 数据处理
ego footprint 面积重叠标签生成
偏移 ROI / hard negative 构造
LaneRoiDataset / DataLoader
训练与验证 transforms
LaneCNN
CE / Weighted CE / Focal Loss
Binary metrics
TensorBoard 训练
checkpoint 保存
evaluate.py
ONNX 导出
ONNX Runtime 单图推理
requirements.txt
README.md
```

尚未在本地完整跑通训练，原因是当前工作区没有 nuScenes mini 数据，且当前环境没有安装 PyTorch / ONNX Runtime 等依赖。

已做的静态验证：

```bash
python3 -m compileall -q src scripts
python3 scripts/export_onnx.py --help
python3 scripts/infer_onnx.py --help
```

## 12. 面试答辩要点

可以重点说明：

1. 这个项目不是车道线检测，而是将地图/车道信息转化为 ROI 二分类监督。
2. `lane_in / lane_out` 的标签不是人工拍脑袋标注，而是通过 ego footprint 与 HD map polygon 的面积重叠比例自动生成。
3. 使用偏移 footprint 和偏移 ROI 构造 `lane_out` 与 hard negative，解决正常驾驶数据负样本不足的问题。
4. 模型采用 `LaneCNN`，因为任务输出是 ROI 级类别，不需要 U-Net 这类像素级分割网络。
5. 默认使用 Weighted CE，并重点关注 `lane_out recall`，因为漏检车道外更重要。
6. 增强策略分为离线 ROI 采样增强和在线图像增强。
7. 默认导出 FP32 ONNX，FP16 和 INT8 都是后续部署优化，不是主线要求。
