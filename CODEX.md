# 车道线二分类感知项目方案

## 1. 项目目标

在自动驾驶感知中，有些基础决策并不需要精确输出车道线形状，只需要快速判断当前车辆前方区域是否处于有效车道内。因此，本项目设计一个轻量级深度学习二分类模型：

- 输入：车载前视图像中的车辆前方 ROI 区域
- 输出：`lane_in` 或 `lane_out`
- 作用：快速验证车辆当前位置是否处于车道区域内，为基础决策或感知状态检查提供参考

本项目不直接做车道线检测、分割或矢量化建图，而是利用开源车道线数据集的标注，将车道线检测问题转换为车道内/车道外二分类问题。

## 2. 数据集选择

推荐使用开源车道线检测数据集：

- TuSimple Lane Detection
- CULane
- BDD100K Lane Marking

基础实现优先选择 TuSimple，原因是标注格式相对简单，适合快速构建面试项目。

数据使用方式：

- 原始图像作为输入来源
- 车道线标注用于自动生成二分类标签
- 训练阶段不要求模型输出车道线，只训练 ROI 分类结果

## 3. 从车道线检测到二分类

这是本项目的关键设计。

核心流程：

```text
车道线标注
-> 解析左右车道边界
-> 定义车辆前方 ROI
-> 判断 ROI 对应的车辆假设位置是否位于左右车道线之间
-> 生成 lane_in / lane_out 标签
```

### 3.1 车道内定义

假设车辆默认位于图像底部中心，车辆前方 ROI 表示当前行驶方向上的局部区域。

在图像高度方向选取多个采样位置：

```text
y = {0.65H, 0.75H, 0.85H, 0.95H}
```

对每个采样高度，根据车道线标注插值得到：

```text
x_left(y)
x_right(y)
```

如果 ROI 中心横坐标 `x_roi` 在多数采样位置上满足：

```text
x_left(y) + margin < x_roi < x_right(y) - margin
```

则认为该 ROI 对应 `lane_in`。

其中 `margin` 用于过滤贴近车道线边界的模糊样本，可以设为：

```text
margin = 0.05 * lane_width(y)
```

或固定为 10-30 像素。

### 3.2 左右车道边界提取

以 TuSimple 标注为例，每条 lane 是一组 `(x, y)` 点。

步骤：

1. 选取参考高度：

```text
y_ref = 0.9H
```

2. 对每条车道线插值得到该高度处的横坐标：

```text
x_i(y_ref)
```

3. 以图像中心 `x_center = 0.5W` 为参考，选择距离中心最近的左右两条线：

```text
left_lane  = max{x_i | x_i < x_center}
right_lane = min{x_i | x_i > x_center}
```

4. 如果无法找到有效左右边界，该样本可标记为 `ignore`，不参与训练。

### 3.3 标签生成规则

对一个候选 ROI：

```text
valid_count = 0
inside_count = 0

for y in sample_ys:
    if left_lane 和 right_lane 在该 y 有效:
        valid_count += 1
        if x_left(y) + margin < x_roi < x_right(y) - margin:
            inside_count += 1

inside_ratio = inside_count / valid_count
```

标签规则：

```text
inside_ratio >= 0.8 -> lane_in
inside_ratio <= 0.2 -> lane_out
其他情况          -> ignore
```

引入 `ignore` 的原因是车道边界附近本身存在语义模糊，强行标注可能引入噪声。

### 3.4 负样本构造

真实车道线数据集中，车辆通常位于当前车道内。如果只使用中心 ROI，`lane_out` 样本会不足。因此需要主动构造负样本。

推荐方式：

```text
中心 ROI                    -> 通常为 lane_in
向左横向偏移的 ROI           -> lane_out 或 hard sample
向右横向偏移的 ROI           -> lane_out 或 hard sample
靠近左车道线外侧的 ROI        -> hard negative
靠近右车道线外侧的 ROI        -> hard negative
无有效车道边界的复杂样本       -> ignore 或 lane_out
```

重点是构造 hard negative，而不是只使用明显不在道路上的区域。这样模型学习到的是当前位置与车道区域的关系，而不是简单判断图像里有没有车道线。

最终可生成训练清单：

```csv
image_path,roi_x1,roi_y1,roi_x2,roi_y2,label
clips/0313-1/6040/20.jpg,430,380,850,720,1
clips/0313-1/6040/20.jpg,120,380,540,720,0
clips/0313-1/6040/20.jpg,760,380,1180,720,0
```

## 4. 模型设计

本项目采用简单轻量 CNN 作为主模型，符合题目中“简单深度学习模型”的要求。

### 4.1 主模型：SimpleLaneCNN

输入：

```text
RGB ROI image: 3 x 224 x 224
```

网络结构：

```text
Conv-BN-ReLU-MaxPool: 3   -> 32
Conv-BN-ReLU-MaxPool: 32  -> 64
Conv-BN-ReLU-MaxPool: 64  -> 128
Conv-BN-ReLU:         128 -> 256
Global Average Pooling
Dropout
Linear: 256 -> 2
```

输出：

```text
logits: [lane_out, lane_in]
```

选择该模型的原因：

- 任务是 ROI 二分类，不需要复杂检测或分割网络
- CNN 足以提取车道线边缘、道路纹理、透视结构等局部特征
- 模型参数少，训练和推理成本低
- ONNX 导出和部署简单
- 结构清晰，适合面试答辩

### 4.2 MobileNet 作为可选对照

MobileNet 也是 CNN，准确说是轻量级卷积神经网络。它使用深度可分离卷积降低计算量。

本项目可以保留 MobileNetV3-Small 作为扩展对照：

```text
SimpleLaneCNN       -> 基础方案，满足简单模型要求
MobileNetV3-Small   -> 工程增强方案，用于对比泛化能力
```

但主方案建议使用自定义 SimpleLaneCNN。因为本任务目标较简单，且面试题强调设计一个简单模型，直接使用复杂 backbone 可能显得过度设计。

## 5. 借鉴车道线检测方法的策略

本项目不直接使用 MapTR、MapQR 等复杂方法，但可以借鉴它们的结构化建模思想。

可借鉴的点：

- 利用车道几何结构，而不是把问题当作普通图片分类
- 使用车道线标注自动生成二分类标签
- 构造边界附近 hard negative 样本
- 对近车区域给予更高关注
- 可选地加入辅助监督，例如车道线 mask 或边缘 heatmap

不建议直接引入：

- Transformer query
- BEV encoder
- Hungarian matching
- polyline point set regression
- 矢量化地图元素预测
- 多摄像头融合

原因是这些方法面向车道线检测、HD Map 或矢量化建图，对当前二分类任务过重。

## 6. 数据增强设计

数据增强原则：

```text
模拟真实驾驶环境变化，但不能破坏 lane_in / lane_out 标签语义。
```

本项目增强分为两类。

### 6.1 离线 ROI 采样增强

这是本任务最重要的增强方式。

从同一张标注图像中生成多个 ROI：

```text
中心 ROI
左偏移 ROI
右偏移 ROI
边界外侧 ROI
边界附近 hard negative ROI
```

每个 ROI 都根据车道线几何关系重新计算标签。

### 6.2 在线图像增强

训练时可使用：

```text
亮度变化
对比度变化
水平翻转
轻微仿射变换
轻微模糊
随机遮挡
```

推荐配置：

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=3,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
        shear=0
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.1,
        scale=(0.02, 0.08),
        ratio=(0.3, 3.3)
    ),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

验证集只做确定性预处理：

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

不推荐的增强：

```text
大角度旋转
大幅随机裁剪
大幅平移
强透视变换
垂直翻转
MixUp / CutMix
随机改变 ROI 位置但不重新计算标签
```

这些增强可能改变 ROI 与车道边界之间的几何关系，导致标签错误。

### 6.3 仿射变换说明

仿射变换是一类几何变换，包括：

```text
平移 translation
旋转 rotation
缩放 scaling
错切 shear
```

在本任务中只使用轻微仿射变换，用于模拟摄像头轻微抖动、安装误差或车身姿态变化。

推荐范围：

```text
旋转 <= 3 度
平移 <= 3%-5%
缩放 0.95-1.05
不使用或少使用错切
```

## 7. 训练方案

损失函数：

```text
CrossEntropyLoss
```

优化器：

```text
AdamW
```

推荐超参数：

```text
batch_size: 32
epochs: 20-50
learning_rate: 1e-3
weight_decay: 1e-4
scheduler: CosineAnnealingLR 或 StepLR
```

如果正负样本不均衡：

```text
Weighted CrossEntropyLoss
或 WeightedRandomSampler
```

关注指标：

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
lane_out recall
```

其中 `lane_out recall` 尤其重要，因为该任务更关注车辆已经偏离车道时能否被识别出来。

## 8. TensorBoard 记录

训练过程中使用 TensorBoard 记录：

```text
train/loss
val/loss
val/accuracy
val/f1
val/lane_out_recall
learning_rate
sample_predictions
```

启动方式：

```bash
tensorboard --logdir runs
```

## 9. ONNX 导出与推理

训练完成后导出：

```text
best_model.pth -> lane_binary.onnx
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

ONNX Runtime 推理流程：

```text
读取前视图像
根据 ROI 坐标裁剪车辆前方区域
Resize 到 224 x 224
Normalize
ONNX Runtime 推理
Softmax
输出 lane_in / lane_out 及置信度
```

示例输出：

```json
{
  "class": "lane_in",
  "confidence": 0.93
}
```

## 10. 项目目录建议

```text
lane-binary-classification/
├── CODEX.md
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── train.csv
│   └── val.csv
├── scripts/
│   ├── prepare_tusimple_binary.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── infer_onnx.py
├── src/
│   └── lane_binary/
│       ├── dataset.py
│       ├── model.py
│       ├── transforms.py
│       ├── metrics.py
│       └── utils.py
├── runs/
├── checkpoints/
└── onnx/
```

## 11. 面试答辩要点

可以重点说明：

1. 本项目不是直接做车道线检测，而是利用车道线标注派生二分类标签。
2. 车道内/车道外通过 ROI 中心与左右车道边界的几何关系定义。
3. 通过横向偏移 ROI 和边界附近 ROI 构造负样本与 hard negative。
4. 主模型采用轻量 CNN，满足简单、快速、可部署的要求。
5. 图像增强重点模拟光照、抖动、阴影和车道线磨损，但不破坏标签语义。
6. 使用 TensorBoard 记录训练过程，便于观察收敛和过拟合。
7. 导出 ONNX 后可用 ONNX Runtime 推理，方便后续接入工程环境。

推荐总结表述：

```text
本项目将车道线检测数据转化为车道内/车道外二分类任务。通过解析开源数据集中的车道线标注，插值得到车辆前方多个 y 位置的左右车道边界，再判断候选 ROI 的中心是否处于左右边界之间。训练时使用中心 ROI 作为正样本，并通过横向偏移和边界外侧采样构造 lane_out 与 hard negative 样本。模型采用轻量 CNN，结合亮度、翻转、轻微仿射和遮挡增强，最终导出 ONNX 并使用 ONNX Runtime 完成推理验证。
```
