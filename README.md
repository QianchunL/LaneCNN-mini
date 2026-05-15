# 车道线二分类感知

本项目将车载前视图像中的车辆前方 ROI 转换为一个简单二分类任务：

- `0 = lane_out`
- `1 = lane_in`

目标不是输出精确车道线，而是判断当前车辆假设位置是否处于有效车道区域内。

## 当前数据方案：nuScenes Mini

为了减小数据量，基础版本使用 `nuScenes v1.0-mini`。它不是专门的车道线检测数据集，但提供：

- 前视相机图像 `CAM_FRONT`
- ego pose
- HD map / map expansion layers

因此可以通过 `ego pose + HD map` 自动生成 `lane_in / lane_out` 标签。

## 标签生成逻辑

数据处理脚本位于：

```bash
scripts/prepare_nuscenes_binary.py
```

核心逻辑：

1. 遍历 nuScenes mini 的 sample。
2. 读取 `CAM_FRONT` 图像和对应 ego pose。
3. 根据车辆长宽构造 ego footprint。
4. 将 footprint 从 ego 坐标转换到 global/map 坐标。
5. 将 nuScenes map 中的 `lane`、`lane_connector`、`drivable_area` 提取为 polygon geometry。
6. 计算 ego footprint polygon 与地图可行驶 polygon 的面积重叠比例并生成标签：

```text
overlap_ratio >= 0.8 -> lane_in
overlap_ratio <= 0.5 -> lane_out
其他情况           -> ignore
```

为了构造足够的 `lane_out` 和 hard negative，脚本会对 ego footprint 做横向偏移，并同步裁剪对应偏移后的 ROI：

```text
0m
±0.8m
±1.6m
±2.4m
```

注意：偏移时不能只改地图标签而不改图像 ROI，否则会出现图像内容和标签不一致。

## 安装依赖

建议先创建虚拟环境，然后安装 nuScenes devkit 和面积计算依赖：

```bash
pip install -r requirements.txt
```

## nuScenes Mini 目录要求

假设数据解压在：

```text
/path/to/nuscenes
├── v1.0-mini
├── samples
├── sweeps
└── maps
```

其中 `maps` 需要包含 nuScenes map expansion 数据。

## 生成训练 CSV

运行：

```bash
python scripts/prepare_nuscenes_binary.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --output-dir data
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

字段含义：

- `image_path`：图像路径，默认相对 nuScenes dataroot。
- `roi_x1, roi_y1, roi_x2, roi_y2`：前方 ROI 裁剪坐标。
- `label`：`0` 表示 `lane_out`，`1` 表示 `lane_in`。
- `lateral_offset_m`：假设车辆横向偏移量，正值表示向左。
- `location`：nuScenes map location。
- `scene_token, sample_token`：用于追踪样本来源。

如果希望 CSV 中写入绝对图像路径：

```bash
python scripts/prepare_nuscenes_binary.py \
  --dataroot /path/to/nuscenes \
  --absolute-paths
```

## 可选：前方 corridor 过滤

由于模型输入是车辆前方 ROI，只使用当前 footprint 作为标签可能与图像内容存在轻微错位。可以开启前方 corridor 检查：

```bash
python scripts/prepare_nuscenes_binary.py \
  --dataroot /path/to/nuscenes \
  --check-front-corridor
```

开启后，脚本会额外检查车辆前方若干点是否仍在可行驶区域内。该检查只用于过滤 `lane_in` 样本，不作为主标签来源。

## 重要参数

```bash
--offsets-m "0,-0.8,0.8,-1.6,1.6,-2.4,2.4"
```

横向偏移假设。正值为 ego 坐标左侧。

```bash
--vehicle-length-m 4.6
--vehicle-width-m 1.9
```

车辆 footprint 尺寸。

```bash
--map-layers "lane,lane_connector,drivable_area"
```

用于面积相交的 nuScenes map polygon 层。默认同时使用车道层和可行驶区域层。

```bash
--in-threshold 0.8
--out-threshold 0.5
```

根据 footprint 与地图 polygon 的面积重叠比例决定标签。

```bash
--roi-width-frac 0.42
--roi-y1-frac 0.52
--roi-y2-frac 0.95
```

ROI 在图像中的基础大小。

```bash
--roi-lateral-span-m 10.0
```

用于将横向米制偏移近似映射到图像 ROI 横向偏移。这个值越大，同样的米制偏移对应的图像移动越小。

## 后续训练建议

数据生成后，训练阶段建议使用：

- 主模型：LaneCNN
- 主 loss：Weighted CrossEntropyLoss
- 可选 loss：Focal Loss
- 指标：accuracy、F1、`lane_out recall`
- 增强：亮度、对比度、水平翻转、轻微仿射、轻微模糊、小面积遮挡

数据增强中需要避免大幅几何变换，因为这会改变 ROI 与车道区域的几何关系，导致标签不可靠。

## 代码结构

当前已实现的数据加载、模型和基础工具模块：

```text
src/lane_binary/
├── dataset/
│   ├── lane_roi_dataset.py   # 读取 CSV、加载图像、裁剪 ROI
│   ├── transforms.py         # 训练/验证图像增强
│   └── dataloader.py         # DataLoader 构建函数
├── model/
│   └── lane_cnn.py           # LaneCNN
└── utils/
    ├── losses.py             # CE、Weighted CE、Focal Loss
    ├── metrics.py            # accuracy、precision、recall、F1
    └── seed.py               # 随机种子
```

使用本地源码时建议设置：

```bash
export PYTHONPATH=src
```

## 数据加载示例

```python
from lane_binary.dataset import LaneRoiDataset, build_train_transform

dataset = LaneRoiDataset(
    csv_path="data/train.csv",
    image_root="/path/to/nuscenes",
    transform=build_train_transform(image_size=224),
)

image, label = dataset[0]
print(image.shape, label)
print(dataset.class_counts())
```

也可以直接构建 `DataLoader`：

```python
from lane_binary.dataset import make_lane_dataloader

train_loader = make_lane_dataloader(
    csv_path="data/train.csv",
    image_root="/path/to/nuscenes",
    batch_size=32,
    train=True,
    num_workers=4,
)
```

## 模型与 Loss 示例

```python
from lane_binary.model import create_model
from lane_binary.utils import build_loss, compute_class_weights

model = create_model("lane_cnn", num_classes=2)

class_counts = {0: 320, 1: 680}
class_weights = compute_class_weights(class_counts)
criterion = build_loss("weighted_ce", class_weights=class_weights)
```

模型输出为两个 logits：

```text
[lane_out_logit, lane_in_logit]
```

训练时将 logits 直接传给 `CrossEntropyLoss` 或 `Weighted CrossEntropyLoss`，不需要手动做 softmax。推理时再对 logits 做 softmax 得到置信度。

## 训练

训练脚本：

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

默认配置：

```text
model: lane_cnn
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-4
scheduler: CosineAnnealingLR
loss: weighted_ce
best checkpoint score: 0.7 * lane_out_recall + 0.3 * lane_out_f1
```

训练输出：

```text
checkpoints/best.pt
checkpoints/last.pt
checkpoints/train_config.json
runs/lane_cnn/
```

查看 TensorBoard：

```bash
tensorboard --logdir runs
```

可切换 loss：

```bash
--loss ce
--loss weighted_ce
--loss focal --focal-gamma 2.0
```

## 评估

评估脚本：

```bash
python scripts/evaluate.py \
  --csv data/val.csv \
  --image-root /path/to/nuscenes \
  --checkpoint checkpoints/best.pt
```

输出指标：

```text
loss
accuracy
lane_out_precision
lane_out_recall
lane_out_f1
lane_in_precision
lane_in_recall
lane_in_f1
confusion_matrix
```

如果需要保存 JSON：

```bash
python scripts/evaluate.py \
  --csv data/val.csv \
  --image-root /path/to/nuscenes \
  --checkpoint checkpoints/best.pt \
  --output-json checkpoints/eval_val.json
```

## ONNX 导出

导出脚本：

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/best.pt \
  --output onnx/lane_cnn.onnx
```

ONNX 输入输出：

```text
input:
  name: image
  shape: [batch, 3, 224, 224]

output:
  name: logits
  shape: [batch, 2]
```

默认支持动态 batch。如果需要固定 batch：

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/best.pt \
  --output onnx/lane_cnn.onnx \
  --no-dynamic-batch
```

## ONNX Runtime 推理

单张图像 + ROI 推理：

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

如果使用 GPU 版 ONNX Runtime，可指定 provider：

```bash
python scripts/infer_onnx.py \
  --onnx onnx/lane_cnn.onnx \
  --image /path/to/image.jpg \
  --roi 430 390 850 720 \
  --providers CUDAExecutionProvider,CPUExecutionProvider
```
