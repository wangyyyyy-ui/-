# 运行说明文档

## 面向课堂复杂场景的多目标视觉感知与学习状态分析系统

---

## 1. 环境准备

### 1.1 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Ubuntu 18.04 / Windows 10 | Ubuntu 20.04+ |
| Python | 3.8 | 3.10+ |
| GPU | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 3060 (12GB)+ |
| CUDA | 11.0 | 11.8+ |
| 内存 | 16 GB | 32 GB |
| 硬盘 | 20 GB | 50 GB+ SSD |

### 1.2 一键环境配置

```bash
# 克隆项目
git clone https://github.com/your-repo/classroom-vision.git
cd classroom-vision

# 运行环境配置脚本
chmod +x setup.sh
./setup.sh
```

### 1.3 手动环境配置

#### 步骤1: 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # Linux
# venv\Scripts\activate   # Windows
```

#### 步骤2: 安装PyTorch

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 步骤3: 安装项目依赖

```bash
pip install -r requirements.txt
```

#### 步骤4: 验证安装

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import cv2
print(f'OpenCV: {cv2.__version__}')

import ultralytics
print(f'Ultralytics: {ultralytics.__version__}')
"
```

预期输出:
```
PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
OpenCV: 4.x.x
Ultralytics: 8.x.x
```

---

## 2. 快速开始

### 2.1 运行Demo (无需真实数据)

```bash
python scripts/run_demo.py
```

Demo将使用合成数据模拟课堂场景，展示系统的完整功能流程。

### 2.2 处理视频文件

```bash
# 基本用法
python scripts/run_pipeline.py --source data/videos/classroom.mp4

# 指定模型和设备
python scripts/run_pipeline.py --source data/videos/classroom.mp4 --model yolov10s --device cuda

# 不显示实时画面（服务器模式）
python scripts/run_pipeline.py --source data/videos/classroom.mp4 --no-display

# 帧采样（每3帧处理1帧，加速处理）
python scripts/run_pipeline.py --source data/videos/classroom.mp4 --sample-interval 3
```

### 2.3 使用摄像头实时分析

```bash
# 默认摄像头
python scripts/run_pipeline.py --source 0

# 指定摄像头
python scripts/run_pipeline.py --source 1

# RTSP流
python scripts/run_pipeline.py --source rtsp://192.168.1.100:554/stream
```

### 2.4 使用配置文件

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

---

## 3. 各模块详细运行说明

### 3.1 视频预处理模块

#### 单独测试预处理

```python
from src.preprocessing.video_loader import VideoLoader
from src.preprocessing.image_enhance import ImageEnhancer

# 加载视频
loader = VideoLoader("data/videos/classroom.mp4", sample_interval=3)
print(f"视频信息: {loader.get_info()}")

# 遍历帧
for frame_info in loader:
    # 自动增强
    enhanced = ImageEnhancer.auto_enhance_for_classroom(frame_info.image)
    # 或手动增强
    enhancer = ImageEnhancer(brightness=1.2, contrast=1.1, denoise_strength=5, clahe_clip=2.5)
    enhanced = enhancer.enhance(frame_info.image)
    break  # 仅处理第一帧作为测试

loader.release()
```

### 3.2 目标检测模块

#### 单独运行检测

```python
from src.detection.yolo_detector import YOLODetector
import cv2

# 加载模型
detector = YOLODetector(model_name="yolov10s", device="cuda")

# 检测单张图像
image = cv2.imread("test.jpg")
result = detector.detect(image)

print(f"检测到 {len(result.detections)} 个目标")
print(f"推理耗时: {result.inference_time:.1f}ms")

for det in result.detections:
    print(f"  类别: {det.class_name}, 置信度: {det.confidence:.2f}, "
          f"位置: {det.bbox}")
```

#### YOLOv10 vs RT-DETR 对比

```python
from src.detection.yolo_detector import YOLODetector
from src.detection.rtdetr_detector import RTDETRDetector
import cv2

image = cv2.imread("test.jpg")

# YOLOv10
yolo = YOLODetector(model_name="yolov10s", device="cuda")
yolo_result = yolo.detect(image)
print(f"YOLOv10: {len(yolo_result.detections)}个目标, {yolo_result.inference_time:.1f}ms")

# RT-DETR
rtdetr = RTDETRDetector(model_name="rtdetr-l", device="cuda")
rtdetr_result = rtdetr.detect(image)
print(f"RT-DETR: {len(rtdetr_result.detections)}个目标, {rtdetr_result.inference_time:.1f}ms")
```

### 3.3 多目标跟踪模块

#### 单独运行跟踪

```python
from src.detection.yolo_detector import YOLODetector
from src.tracking.byte_tracker import ByteTrackWrapper
from src.preprocessing.video_loader import VideoLoader

loader = VideoLoader("data/videos/classroom.mp4")
detector = YOLODetector(model_name="yolov10s", device="cuda")
tracker = ByteTrackWrapper(track_thresh=0.5, track_buffer=30)

for frame_info in loader:
    # 检测
    det_result = detector.detect(frame_info.image)
    # 跟踪
    track_result = tracker.update(det_result, frame_info.frame_id)
    
    print(f"帧 {frame_info.frame_id}: 跟踪到 {len(track_result.tracks)} 个目标")
    for track in track_result.tracks:
        print(f"  ID={track.track_id}, bbox={track.bbox}, conf={track.confidence:.2f}")

loader.release()
```

### 3.4 人脸识别模块

#### 构建人脸特征库

```bash
# 准备照片目录
# data/face_db/photos/
# ├── student_001/
# │   ├── photo1.jpg
# │   └── photo2.jpg
# ├── student_002/
# │   └── photo1.jpg

# 构建特征库
python tools/build_face_db.py --image_dir data/face_db/photos --db_path data/face_db
```

#### 单独运行人脸识别

```python
from src.face_recognition.arcface import ArcFaceRecognizer
from src.face_recognition.face_db import FaceDatabase
import cv2

# 初始化
recognizer = ArcFaceRecognizer(model_name="buffalo_l", threshold=0.4)
face_db = FaceDatabase(db_path="data/face_db", recognizer=recognizer)

# 识别图像中的人脸
image = cv2.imread("test.jpg")
faces = recognizer.detect_faces(image)

for face in faces:
    identity, similarity = recognizer.match_identity(face.embedding, face_db)
    print(f"  身份: {identity}, 相似度: {similarity:.3f}")
```

### 3.5 头部姿态估计模块

```python
from src.pose_estimation.head_pose import HeadPoseEstimator
import cv2

estimator = HeadPoseEstimator()
image = cv2.imread("test.jpg")

poses = estimator.estimate(image)
for pose in poses:
    print(f"  Pitch={pose.pitch:.1f}°, Yaw={pose.yaw:.1f}°, Roll={pose.roll:.1f}°")
    print(f"  头部朝向: {pose.head_direction.value}")
    print(f"  视线方向: {pose.gaze_direction.value}")
```

### 3.6 状态分析模块

```python
from src.state_analysis.state_classifier import StateClassifier
from src.state_analysis.temporal_fusion import TemporalFusion
from src.pose_estimation.head_pose import HeadPoseResult, HeadDirection, GazeDirection

# 初始化
classifier = StateClassifier()
fusion = TemporalFusion(fps=30.0, window_size=15)

# 分类状态
result = classifier.classify(
    track_id=1,
    head_pose=head_pose_result,  # 来自头部姿态估计
    body_pose=body_pose_result,  # 来自身体姿态估计
    is_tracked=True,
    identity="student_001",
)

# 时序融合
smoothed_states, stats = fusion.update(
    state_results=[result],
    frame_id=0,
    timestamp=0.0,
    registered_students=30,
)

print(f"状态: {result.state.value}, 置信度: {result.confidence:.2f}")
print(f"统计: 到课{stats.total_students}人, 抬头率{stats.focus_rate:.1%}")
```

---

## 4. 评估运行说明

### 4.1 完整评估

```bash
# 运行所有评估
python scripts/run_evaluation.py --task all --output outputs/evaluation

# 仅评估检测
python scripts/run_evaluation.py --task detection --model yolov10s

# 仅评估跟踪
python scripts/run_evaluation.py --task tracking

# 仅评估人脸识别
python scripts/run_evaluation.py --task face

# 仅评估状态分析
python scripts/run_evaluation.py --task state
```

### 4.2 单独评估各模块

```bash
# 检测评估
python tools/eval_detection.py --model yolov10s --data data/annotations/val.yaml

# 跟踪评估
python tools/eval_tracking.py --gt data/annotations/gt_mot.txt --result outputs/tracking/result.txt

# 人脸识别评估
python tools/eval_face_recognition.py --db data/face_db --test data/face_db/test

# 状态分析评估
python tools/eval_state.py --gt data/annotations/gt_states.json --pred outputs/pred_states.json
```

### 4.3 评估结果查看

```bash
# 查看评估结果
cat outputs/evaluation/evaluation_results.json | python -m json.tool

# 查看统计图表
# 图表保存在 outputs/ 目录下:
#   - stats_*_rates.png   (抬头率/到课率趋势)
#   - stats_*_counts.png  (各状态人数趋势)
```

---

## 5. 输出文件说明

### 5.1 标注视频

- **路径**: `outputs/annotated_video.avi`
- **格式**: AVI (XVID编码)
- **内容**: 在原始视频上叠加检测框、跟踪ID、身份标签、状态标签、统计面板

### 5.2 统计数据

#### CSV格式

- **路径**: `outputs/stats_YYYYMMDD_HHMMSS.csv`
- **列**: 时间戳, 到课人数, 专注人数, 分心人数, 低头人数, 离座人数, 举手人数, 抬头率, 到课率

#### JSON格式

- **路径**: `outputs/report_YYYYMMDD_HHMMSS.json`
- **内容**: 完整分析报告，包含统计历史、离座事件、汇总信息

### 5.3 评估结果

- **路径**: `outputs/evaluation/evaluation_results.json`
- **内容**: 各模块评估指标的汇总

---

## 6. 常见问题

### Q1: CUDA out of memory

```bash
# 减小batch size或使用更小的模型
python scripts/run_pipeline.py --source video.mp4 --model yolov10n

# 或使用CPU模式（速度较慢）
python scripts/run_pipeline.py --source video.mp4 --device cpu
```

### Q2: InsightFace模型下载失败

```bash
# 手动下载模型
mkdir -p ~/.insightface/models
# 从 https://github.com/deepinsight/insightface/releases 下载 buffalo_l.zip
# 解压到 ~/.insightface/models/buffalo_l/
```

### Q3: 视频无法打开

```bash
# 检查视频编码
ffprobe data/videos/classroom.mp4

# 转码为兼容格式
ffmpeg -i input.mp4 -c:v libx264 -preset medium output.mp4
```

### Q4: 人脸识别准确率低

1. 确保注册照片质量良好（正面、清晰、光线充足）
2. 调整匹配阈值（降低阈值提高召回率，提高阈值降低误识率）
3. 尝试启用SAM分割辅助

### Q5: 处理速度慢

1. 使用更小的模型（yolov10n）
2. 增大帧采样间隔
3. 关闭不需要的模块（如身体姿态估计）
4. 确保使用GPU推理

---

## 7. 配置文件详解

详见 `configs/default.yaml`，包含以下配置组:

| 配置组 | 说明 |
|--------|------|
| video | 视频源、采样间隔、分辨率 |
| preprocessing | 亮度、对比度、去噪、CLAHE |
| detection | 模型、置信度阈值、IoU阈值、设备 |
| tracking | 跟踪阈值、缓冲帧数、匹配阈值 |
| face_recognition | 模型、匹配阈值、特征库路径 |
| head_pose | 估计方法、角度阈值 |
| body_pose | 是否启用、模型复杂度 |
| state_analysis | 状态分类阈值 |
| temporal_fusion | 窗口大小、离座判定帧数 |
| visualization | 显示选项、字体、线条 |
| output | 输出目录、保存选项 |

---

## 8. 开发指南

### 8.1 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_detection.py -v
pytest tests/test_tracking.py -v
pytest tests/test_pipeline.py -v
```

### 8.2 代码风格

```bash
# 格式化代码
black src/ scripts/ tools/

# 检查代码风格
flake8 src/ scripts/ tools/
```

### 8.3 添加新模块

1. 在 `src/` 下创建新模块目录
2. 实现 `__init__.py` 和核心类
3. 在 `src/pipeline.py` 中集成新模块
4. 编写单元测试
5. 更新配置文件和文档
