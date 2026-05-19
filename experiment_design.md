# 实验设计说明文档

## 面向课堂复杂场景的多目标视觉感知与学习状态分析研究

---

## 1. 实验总体设计

### 1.1 实验目标

本研究的实验设计围绕以下核心目标展开：

1. **验证目标检测改进方案的有效性**：评估YOLOv10在课堂场景下的检测性能，验证Mosaic-9数据增强、BiFPN多尺度特征融合、SE通道注意力机制等改进策略对检测精度的提升效果。
2. **验证ByteTrack在课堂密集遮挡场景的跟踪能力**：评估ByteTrack两轮关联策略在学生密集排列、频繁遮挡条件下的跟踪稳定性，探索运动模型与弱特征关联的改进方向。
3. **验证ArcFace在远距离/遮挡场景的识别鲁棒性**：评估ArcFace特征提取在课堂远距离拍摄条件下的人脸识别性能，探索跟踪辅助和SAM分割辅助两种增强策略。
4. **验证多模态融合状态分类的准确性**：评估头部姿态、视线方向、身体姿态等多模态信息融合对学生课堂状态分类的效果，验证时序融合策略对分类稳定性的改善。
5. **验证端到端系统的实用性**：评估完整流水线在真实课堂场景下的综合性能，包括处理速度、统计准确度和实际可用性。

### 1.2 实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Ubuntu 20.04 LTS / Windows 10+ |
| GPU | NVIDIA RTX 3060 (12GB) / RTX 4090 (24GB) |
| CPU | Intel i7-12700 / AMD Ryzen 9 5900X |
| 内存 | 32 GB DDR4 |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ |

### 1.3 数据集规划

#### 1.3.1 目标检测数据集

- **基础数据集**: COCO 2017 (person类别，约41,000张训练图像，5,000张验证图像)
- **课堂场景数据集**: 自建数据集，包含以下场景：
  - 小教室场景 (20-30人)：近距离拍摄，光照良好
  - 大教室场景 (50-80人)：远距离拍摄，存在遮挡
  - 低光照场景：光线不足的教室环境
  - 在线课堂场景：屏幕共享画面
- **标注格式**: YOLO格式 (class_id cx cy w h)
- **数据增强**: Mosaic-9、MixUp、随机仿射变换、HSV颜色空间增强

#### 1.3.2 多目标跟踪数据集

- **公开数据集**: MOT17 (7个训练序列 + 7个测试序列)
- **课堂跟踪数据集**: 自建数据集，从课堂视频中提取连续帧并标注跟踪ID
- **标注格式**: MOT格式 (frame_id, track_id, x, y, w, h, conf, class_id)

#### 1.3.3 人脸识别数据集

- **注册集**: 学生登记照片，每人1-3张正面照
- **测试集**: 从课堂视频中截取的人脸区域，包含：
  - 正面人脸 (近距离)
  - 侧面人脸 (偏转30°-60°)
  - 遮挡人脸 (口罩、手、书本遮挡)
  - 远距离人脸 (模糊、低分辨率)
- **标注格式**: 每个人脸区域标注对应的学生ID

#### 1.3.4 课堂状态数据集

- **标注方式**: 人工逐帧标注每个学生的课堂状态
- **状态类别**: 专注(focused)、分心(distracted)、低头(head_down)、离座(left_seat)、举手(hand_raising)
- **标注工具**: CVAT / Label Studio
- **标注规范**: 每个学生每秒标注一次状态，由两名标注员独立标注后取一致结果

---

## 2. 实验一：目标检测算法对比

### 2.1 实验目的

对比YOLOv10不同规模模型和RT-DETR在课堂场景下的检测性能，验证改进策略的有效性。

### 2.2 实验设计

#### 2.2.1 实验组设置

| 实验编号 | 模型 | 改进策略 | 说明 |
|---------|------|---------|------|
| Exp1-1 | YOLOv10n | 无 | Nano基线，最快速度 |
| Exp1-2 | YOLOv10s | 无 | Small基线，速度精度平衡 |
| Exp1-3 | YOLOv10m | 无 | Medium基线 |
| Exp1-4 | YOLOv10l | 无 | Large基线，高精度 |
| Exp1-5 | RT-DETR-l | 无 | Transformer对比方案 |
| Exp1-6 | YOLOv10s+BiFPN | BiFPN多尺度融合 | 特征融合改进 |
| Exp1-7 | YOLOv10s+SE | SE通道注意力 | 注意力机制改进 |
| Exp1-8 | YOLOv10s+Mosaic9 | Mosaic-9数据增强 | 数据增强改进 |
| Exp1-9 | YOLOv10s+All | BiFPN+SE+Mosaic9 | 综合改进 |

#### 2.2.2 训练配置

```yaml
# 训练超参数
epochs: 300
batch_size: 16
imgsz: 640
optimizer: SGD
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# 数据增强 (基线)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0  # Mosaic-4 (基线)

# 数据增强 (Mosaic-9改进组)
mosaic: 1.0
mixup: 0.1
copy_paste: 0.1
```

#### 2.2.3 评估流程

```bash
# 1. 在COCO person子集上预训练
python -c "from ultralytics import YOLO; model = YOLO('yolov10s.pt'); model.train(data='coco_person.yaml', epochs=100)"

# 2. 在课堂数据集上微调
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train/weights/best.pt'); model.train(data='classroom.yaml', epochs=300)"

# 3. 评估
python tools/eval_detection.py --model runs/detect/train2/weights/best.pt --data data/annotations/val.yaml
```

### 2.3 预期结果

| 实验编号 | mAP@0.5 | mAP@0.5:0.95 | FPS (RTX3060) |
|---------|---------|--------------|---------------|
| Exp1-1 | ~0.82 | ~0.55 | ~120 |
| Exp1-2 | ~0.87 | ~0.62 | ~80 |
| Exp1-3 | ~0.90 | ~0.67 | ~50 |
| Exp1-4 | ~0.92 | ~0.70 | ~30 |
| Exp1-5 | ~0.89 | ~0.65 | ~40 |
| Exp1-6 | ~0.88 | ~0.64 | ~75 |
| Exp1-7 | ~0.89 | ~0.65 | ~72 |
| Exp1-8 | ~0.88 | ~0.63 | ~78 |
| Exp1-9 | ~0.91 | ~0.68 | ~65 |

### 2.4 消融实验设计

为验证各改进策略的独立贡献，设计以下消融实验：

| 消融组 | BiFPN | SE | Mosaic-9 | 预期mAP@0.5提升 |
|--------|-------|-----|----------|----------------|
| 基线 | ✗ | ✗ | ✗ | - |
| +BiFPN | ✓ | ✗ | ✗ | +1.0% |
| +SE | ✗ | ✓ | ✗ | +1.5% |
| +Mosaic-9 | ✗ | ✗ | ✓ | +0.8% |
| +BiFPN+SE | ✓ | ✓ | ✗ | +2.2% |
| +All | ✓ | ✓ | ✓ | +3.5% |

---

## 3. 实验二：多目标跟踪算法对比

### 3.1 实验目的

评估ByteTrack在课堂密集遮挡场景下的跟踪性能，验证改进关联策略的有效性。

### 3.2 实验设计

#### 3.2.1 实验组设置

| 实验编号 | 算法 | 改进策略 | 说明 |
|---------|------|---------|------|
| Exp2-1 | ByteTrack | 无 | 基线算法 |
| Exp2-2 | ByteTrack+Kalman | 卡尔曼滤波预测 | 运动模型改进 |
| Exp2-3 | ByteTrack+ReID | 弱ReID特征关联 | 外观特征辅助 |
| Exp2-4 | ByteTrack+All | Kalman+ReID | 综合改进 |
| Exp2-5 | DeepSORT | - | 对比方案 |
| Exp2-6 | BoT-SORT | - | 对比方案 |

#### 3.2.2 跟踪参数配置

```yaml
# ByteTrack基线参数
track_thresh: 0.5          # 高置信度阈值
track_buffer: 30           # 轨迹缓冲帧数
match_thresh: 0.8          # 匹配IoU阈值

# Kalman改进参数
use_kalman: true           # 启用卡尔曼预测
kalman_process_noise: 0.01 # 过程噪声
kalman_measure_noise: 0.1  # 测量噪声

# ReID改进参数
use_reid: true             # 启用ReID特征
reid_model: "osnet_x0_25"  # ReID模型
reid_weight: 0.3           # ReID距离权重
```

#### 3.2.3 评估流程

```bash
# 1. 生成跟踪结果
python scripts/run_pipeline.py --source data/videos/classroom.mp4 --output outputs/tracking/

# 2. 转换为MOT格式
python tools/convert_to_mot.py --input outputs/tracking/result.json --output outputs/tracking/result.txt

# 3. 评估
python tools/eval_tracking.py --gt data/annotations/gt_mot.txt --result outputs/tracking/result.txt
```

### 3.3 预期结果

| 实验编号 | MOTA↑ | IDF1↑ | ID Switch↓ | MT↑ | ML↓ | FPS |
|---------|-------|-------|-----------|-----|-----|-----|
| Exp2-1 | 72.5 | 68.3 | 45 | 55% | 12% | 65 |
| Exp2-2 | 74.8 | 70.1 | 38 | 58% | 10% | 60 |
| Exp2-3 | 75.2 | 72.5 | 28 | 60% | 9% | 45 |
| Exp2-4 | 77.0 | 74.8 | 22 | 63% | 7% | 40 |
| Exp2-5 | 70.1 | 71.2 | 35 | 52% | 15% | 35 |
| Exp2-6 | 73.8 | 73.0 | 30 | 57% | 11% | 38 |

---

## 4. 实验三：人脸识别与身份匹配

### 4.1 实验目的

评估ArcFace在课堂远距离/遮挡场景下的识别性能，验证跟踪辅助和SAM分割辅助策略的有效性。

### 4.2 实验设计

#### 4.2.1 实验组设置

| 实验编号 | 方法 | 改进策略 | 说明 |
|---------|------|---------|------|
| Exp3-1 | ArcFace (buffalo_l) | 无 | 基线方案 |
| Exp3-2 | ArcFace + 跟踪辅助 | 利用跟踪减少识别频率 | 跟踪确认后减少重复识别 |
| Exp3-3 | ArcFace + SAM分割 | SAM辅助人脸区域提取 | 精确分割遮挡边界 |
| Exp3-4 | ArcFace + All | 跟踪辅助+SAM分割 | 综合改进 |

#### 4.2.2 识别参数配置

```yaml
# ArcFace基线参数
model: "buffalo_l"         # InsightFace模型
det_size: [640, 640]       # 人脸检测尺寸
threshold: 0.4             # 匹配相似度阈值

# 跟踪辅助参数
use_track_assist: true     # 启用跟踪辅助
confirm_frames: 5          # 确认帧数(连续5帧匹配成功后确认身份)
recheck_interval: 30       # 重新检查间隔(每30帧重新识别一次)

# SAM分割参数
use_sam: true              # 启用SAM分割
sam_model: "sam_vit_b"     # SAM模型
sam_prompt: "face_bbox"    # 使用人脸框作为提示
```

#### 4.2.3 测试场景设计

| 场景 | 距离 | 遮挡程度 | 光照 | 人数 |
|------|------|---------|------|------|
| 近距离正面 | 2-3m | 无遮挡 | 良好 | 20 |
| 中距离侧面 | 5-8m | 轻微遮挡 | 良好 | 40 |
| 远距离模糊 | 10-15m | 中度遮挡 | 一般 | 60 |
| 低光照 | 5-8m | 轻微遮挡 | 较差 | 30 |
| 口罩遮挡 | 3-5m | 口罩遮挡 | 良好 | 20 |

### 4.3 预期结果

| 实验编号 | Accuracy↑ | FAR↓ | FRR↓ | EER↓ | Rank-1↑ |
|---------|-----------|------|------|------|---------|
| Exp3-1 | 85.2% | 2.1% | 12.7% | 5.8% | 88.5% |
| Exp3-2 | 87.5% | 1.8% | 10.7% | 5.2% | 90.2% |
| Exp3-3 | 88.1% | 1.5% | 10.4% | 4.9% | 91.0% |
| Exp3-4 | 90.3% | 1.2% | 8.5% | 4.2% | 93.1% |

---

## 5. 实验四：课堂状态分析

### 5.1 实验目的

评估多模态融合状态分类的准确性，验证时序融合策略对分类稳定性的改善。

### 5.2 实验设计

#### 5.2.1 实验组设置

| 实验编号 | 输入模态 | 分类方法 | 时序处理 | 说明 |
|---------|---------|---------|---------|------|
| Exp4-1 | 头部姿态 | 规则引擎 | 无 | 仅pitch/yaw角度 |
| Exp4-2 | 身体姿态 | 规则引擎 | 无 | 仅关键点空间关系 |
| Exp4-3 | 头部+身体 | 规则引擎 | 无 | 多模态融合 |
| Exp4-4 | 头部+身体 | 规则引擎 | 滑动窗口 | 时序投票 |
| Exp4-5 | 头部+身体 | 规则引擎 | 窗口+状态机 | 完整时序融合 |
| Exp4-6 | 头部+身体 | MLP分类器 | 窗口+状态机 | 学习型分类器 |

#### 5.2.2 状态分类规则

```
分类优先级 (从高到低):
1. 离座检测: 跟踪丢失超过30帧 → LEFT_SEAT
2. 举手检测: 手腕高于肩膀 → HAND_RAISING
3. 低头检测: pitch > 15° → HEAD_DOWN
4. 分心检测: |yaw| > 30° → DISTRACTED
5. 专注状态: |pitch| < 15° 且 |yaw| < 30° → FOCUSED
6. 默认: UNKNOWN
```

#### 5.2.3 时序融合参数

```yaml
# 滑动窗口参数
window_size: 15            # 窗口大小(帧数)
voting_method: "majority"  # 投票方式: majority/weighted

# 状态机参数
transition_frames:
  focused_to_distracted: 3   # 专注→分心需连续3帧
  focused_to_head_down: 5    # 专注→低头需连续5帧
  head_down_to_focused: 3    # 低头→专注需连续3帧
  any_to_left_seat: 30       # 任何→离座需连续30帧丢失
  left_seat_to_any: 1        # 离座→任何状态立即恢复
```

### 5.3 预期结果

| 实验编号 | Accuracy↑ | F1(专注)↑ | F1(低头)↑ | F1(离座)↑ | Kappa↑ |
|---------|-----------|----------|----------|----------|--------|
| Exp4-1 | 72.3% | 0.78 | 0.65 | 0.00 | 0.58 |
| Exp4-2 | 65.1% | 0.55 | 0.48 | 0.72 | 0.45 |
| Exp4-3 | 78.5% | 0.82 | 0.71 | 0.72 | 0.67 |
| Exp4-4 | 82.1% | 0.85 | 0.76 | 0.75 | 0.72 |
| Exp4-5 | 85.7% | 0.88 | 0.80 | 0.82 | 0.78 |
| Exp4-6 | 87.2% | 0.89 | 0.82 | 0.83 | 0.80 |

---

## 6. 实验五：端到端系统评估

### 6.1 实验目的

评估完整流水线在真实课堂场景下的综合性能。

### 6.2 实验设计

#### 6.2.1 测试场景

| 场景编号 | 场景描述 | 人数 | 光照 | 拍摄距离 | 时长 |
|---------|---------|------|------|---------|------|
| S1 | 小教室 | 20 | 良好 | 3-5m | 45min |
| S2 | 大教室 | 60 | 良好 | 8-12m | 45min |
| S3 | 低光照教室 | 30 | 较差 | 5-8m | 45min |
| S4 | 在线课堂 | 25 | 良好 | - | 45min |

#### 6.2.2 评估维度

| 维度 | 指标 | 说明 |
|------|------|------|
| 处理速度 | 端到端FPS | 从输入帧到输出结果的帧率 |
| 模块延迟 | 各模块耗时 | 检测/跟踪/识别/姿态/分类各模块延迟 |
| 统计准确度 | 到课率偏差 | 系统统计与人工统计的偏差 |
| 统计准确度 | 抬头率偏差 | 系统统计与人工统计的偏差 |
| 鲁棒性 | 不同场景性能变化 | 不同场景下性能的稳定性 |

### 6.3 预期结果

| 场景 | FPS | 到课率偏差 | 抬头率偏差 | 检测mAP@0.5 |
|------|-----|----------|----------|-------------|
| S1 | 28 | ±2% | ±5% | 0.91 |
| S2 | 18 | ±5% | ±8% | 0.85 |
| S3 | 22 | ±4% | ±10% | 0.82 |
| S4 | 25 | ±3% | ±6% | 0.88 |

---

## 7. 实验流程总览

```
实验一: 目标检测对比 ──────────────────────────────────────────
  │  数据准备: COCO person + 课堂数据集
  │  训练: YOLOv10n/s/m/l + RT-DETR + 改进方案
  │  评估: mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS
  │  消融: BiFPN / SE / Mosaic-9 独立贡献
  ▼
实验二: 多目标跟踪对比 ────────────────────────────────────────
  │  数据准备: MOT17 + 课堂跟踪标注
  │  测试: ByteTrack + DeepSORT + BoT-SORT
  │  评估: MOTA, IDF1, ID Switch, MT/ML
  ▼
实验三: 人脸识别对比 ──────────────────────────────────────────
  │  数据准备: 注册照片 + 课堂测试集
  │  测试: ArcFace + 跟踪辅助 + SAM分割
  │  评估: Accuracy, FAR, FRR, EER, Rank-1
  ▼
实验四: 状态分析对比 ──────────────────────────────────────────
  │  数据准备: 人工标注状态数据集
  │  测试: 单模态/多模态/时序融合
  │  评估: Accuracy, F1, Kappa, Confusion Matrix
  ▼
实验五: 端到端系统评估 ────────────────────────────────────────
     数据准备: 4种真实课堂场景
     测试: 完整流水线
     评估: FPS, 延迟, 统计准确度, 鲁棒性
```

---

## 8. 数据标注规范

### 8.1 目标检测标注

- **标注工具**: CVAT / LabelImg
- **标注类别**: person (COCO class 0)
- **标注要求**:
  - 可见身体超过30%时标注
  - 严重遮挡(>70%)不标注
  - 边界框紧贴人体边缘
  - 同一人不重复标注

### 8.2 跟踪标注

- **标注工具**: CVAT
- **标注格式**: MOT格式
- **标注要求**:
  - 每个目标分配唯一track_id
  - 目标离开画面后重新出现分配新ID
  - 严重遮挡后重新出现尽量保持原ID

### 8.3 状态标注

- **标注工具**: 自研标注工具
- **标注频率**: 每秒1次
- **标注类别**: focused / distracted / head_down / left_seat / hand_raising
- **标注要求**:
  - 两名标注员独立标注
  - 不一致样本由第三名标注员裁决
  - 计算标注员间一致性(Kappa > 0.8)

---

## 9. 实验结果记录模板

### 9.1 检测实验记录

```json
{
  "experiment_id": "Exp1-2",
  "model": "yolov10s",
  "improvements": [],
  "dataset": "classroom_val",
  "results": {
    "precision": 0.0,
    "recall": 0.0,
    "mAP50": 0.0,
    "mAP50-95": 0.0,
    "fps": 0.0
  },
  "training_info": {
    "epochs": 300,
    "best_epoch": 0,
    "gpu_memory": "0 MB"
  }
}
```

### 9.2 跟踪实验记录

```json
{
  "experiment_id": "Exp2-1",
  "algorithm": "ByteTrack",
  "improvements": [],
  "dataset": "classroom_track_val",
  "results": {
    "MOTA": 0.0,
    "IDF1": 0.0,
    "ID_Switch": 0,
    "MT": 0,
    "ML": 0,
    "FP": 0,
    "FN": 0
  }
}
```

### 9.3 状态分析实验记录

```json
{
  "experiment_id": "Exp4-5",
  "modalities": ["head_pose", "body_pose"],
  "method": "rule_engine",
  "temporal": "window+state_machine",
  "results": {
    "overall_accuracy": 0.0,
    "cohens_kappa": 0.0,
    "per_class": {
      "focused": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
      "distracted": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
      "head_down": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
      "left_seat": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
      "hand_raising": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    },
    "confusion_matrix": []
  }
}
```
