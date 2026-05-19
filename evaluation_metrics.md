# 评价指标详细说明文档

## 面向课堂复杂场景的多目标视觉感知与学习状态分析研究

---

## 1. 目标检测评价指标

### 1.1 Precision (精确率)

**定义**: 在所有被模型预测为正例的样本中，实际为正例的比例。

**公式**:
```
Precision = TP / (TP + FP)
```

**符号说明**:
- TP (True Positive): 正确检测到的目标数
- FP (False Positive): 误检数（背景被误检为目标，或同一目标被重复检测）

**计算示例**:
假设画面中有10个学生，模型检测出12个目标，其中9个是正确的，3个是误检：
```
TP = 9, FP = 3
Precision = 9 / (9 + 3) = 0.75
```

**课堂场景意义**: Precision衡量检测器的"可信度"。高Precision意味着系统报告的检测结果大部分是真实的，减少误报对后续跟踪和识别的干扰。

### 1.2 Recall (召回率)

**定义**: 在所有实际为正例的样本中，被正确检测为正例的比例。

**公式**:
```
Recall = TP / (TP + FN)
```

**符号说明**:
- FN (False Negative): 漏检数（实际存在但未被检测到的目标）

**计算示例**:
```
TP = 9, FN = 1 (1个学生未被检测到)
Recall = 9 / (9 + 1) = 0.90
```

**课堂场景意义**: Recall衡量检测器的"完整性"。高Recall意味着系统能检测到画面中的大部分学生，减少漏检对到课率统计的影响。

### 1.3 AP (Average Precision)

**定义**: Precision-Recall曲线下的面积，综合反映不同置信度阈值下的检测性能。

**计算方法** (11点插值法):
```
AP = (1/11) * Σ max(Precision|Recall >= t), t ∈ {0, 0.1, 0.2, ..., 1.0}
```

**课堂场景意义**: AP比单一Precision/Recall更全面，因为它考虑了所有可能的置信度阈值下的性能。

### 1.4 mAP@0.5

**定义**: IoU阈值为0.5时，所有类别AP的平均值。

**公式**:
```
mAP@0.5 = (1/C) * Σ AP_c, c = 1, 2, ..., C
```

**IoU (Intersection over Union)**:
```
IoU = Area(B_pred ∩ B_gt) / Area(B_pred ∪ B_gt)
```

当IoU >= 0.5时，认为检测框与真值框匹配成功。

**课堂场景意义**: mAP@0.5是目标检测最常用的评估指标，在课堂场景中，它反映检测器定位学生的整体能力。

### 1.5 mAP@0.5:0.95

**定义**: 在IoU阈值从0.5到0.95（步长0.05）的10个阈值下，mAP的平均值。

**公式**:
```
mAP@0.5:0.95 = (1/10) * Σ mAP@t, t ∈ {0.5, 0.55, 0.60, ..., 0.95}
```

**课堂场景意义**: 更严格的评估指标，要求检测框与真值框高度重合。在课堂场景中，高mAP@0.5:0.95意味着检测框定位更精确，有利于后续的人脸识别和姿态估计。

### 1.6 FPS (推理速度)

**定义**: 每秒处理的帧数。

**测量方法**:
```python
start_time = time.time()
for frame in video:
    result = detector.detect(frame)
end_time = time.time()
fps = num_frames / (end_time - start_time)
```

**课堂场景意义**: 实时性要求。课堂分析系统需要至少15FPS才能保证流畅的实时分析，30FPS以上为理想目标。

---

## 2. 多目标跟踪评价指标

### 2.1 MOTA (Multi-Object Tracking Accuracy)

**定义**: 综合评估漏检、误检和身份切换的跟踪准确度指标。

**公式**:
```
MOTA = 1 - (FN_t + FP_t + IDSW_t) / GT_t
```

**符号说明**:
- FN_t: 第t帧的漏检数
- FP_t: 第t帧的误检数
- IDSW_t: 第t帧发生的身份切换次数
- GT_t: 第t帧的真值目标数

**取值范围**: (-∞, 1]，通常为0到1之间，越高越好

**计算示例**:
```
总GT = 1000, 总FN = 150, 总FP = 80, 总IDSW = 20
MOTA = 1 - (150 + 80 + 20) / 1000 = 0.75
```

**课堂场景意义**: MOTA是MOT领域最核心的指标，综合反映跟踪系统在课堂场景中的整体性能。高MOTA意味着系统能准确跟踪大部分学生，且身份切换少。

### 2.2 IDF1 (ID F1 Score)

**定义**: 基于身份匹配的F1分数，评估跟踪器保持身份一致性的能力。

**公式**:
```
IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

其中:
IDTP: 身份匹配正确的检测数
IDFP: 身份匹配错误的检测数
IDFN: 身份未匹配的真值数
```

**课堂场景意义**: IDF1更关注身份保持的一致性。在课堂场景中，高IDF1意味着系统能长时间保持对同一学生的正确身份标注，这对人脸识别辅助和个体状态追踪至关重要。

### 2.3 ID Switch (身份切换次数)

**定义**: 跟踪过程中，同一真值目标对应的跟踪ID发生变化的次数。

**判定条件**: 当一个真值目标在连续帧中与不同track_id匹配时，记一次ID Switch。

**计算方法**:
```
对于每个真值目标g:
  记录每帧匹配的track_id序列: [id1, id1, id2, id1, id3, ...]
  ID Switch次数 = 相邻帧ID不同的次数
  上述示例: id1→id2 (1次), id2→id1 (1次), id1→id3 (1次) = 3次
```

**课堂场景意义**: ID Switch直接影响人脸识别的准确性。频繁的身份切换会导致同一学生被识别为不同人，影响到课率统计的准确性。

### 2.4 MT (Mostly Tracked)

**定义**: 被成功跟踪超过其生命周期80%的目标数占总目标数的比例。

**公式**:
```
MT_ratio = MT_count / total_GT_tracks
```

**课堂场景意义**: MT反映跟踪的稳定性。在课堂场景中，高MT意味着大部分学生能被持续稳定跟踪，减少因跟踪丢失导致的离座误判。

### 2.5 ML (Mostly Lost)

**定义**: 被成功跟踪不足其生命周期20%的目标数占总目标数的比例。

**公式**:
```
ML_ratio = ML_count / total_GT_tracks
```

**课堂场景意义**: ML反映跟踪的丢失率。高ML意味着大量学生目标频繁丢失，严重影响系统可用性。

---

## 3. 人脸识别评价指标

### 3.1 Accuracy (识别准确率)

**定义**: 正确识别身份的次数占总识别次数的比例。

**公式**:
```
Accuracy = N_correct / N_total
```

**课堂场景意义**: 最直观的识别性能指标。在课堂场景中，Accuracy直接影响到课率统计的准确性。

### 3.2 FAR (False Accept Rate, 误签率)

**定义**: 冒充者被错误接受为合法用户的概率。

**公式**:
```
FAR = FP / (FP + TN)
```

**符号说明**:
- FP: 不同人被错误匹配为同一人的次数
- TN: 不同人被正确拒绝的次数

**计算方法**:
```
1. 对所有非同一人的配对计算相似度
2. 相似度超过阈值T的配对记为FP
3. 相似度低于阈值T的配对记为TN
4. FAR = FP / (FP + TN)
```

**课堂场景意义**: FAR衡量系统的安全性。高FAR意味着系统容易将不同学生混淆，导致到课率统计出现"替签到"的错误。

### 3.3 FRR (False Reject Rate, 漏签率)

**定义**: 合法用户被错误拒绝的概率。

**公式**:
```
FRR = FN / (FN + TP)
```

**符号说明**:
- FN: 同一人被错误拒绝的次数
- TP: 同一人被正确接受的次数

**课堂场景意义**: FRR衡量系统的可用性。高FRR意味着已到课的学生无法被系统识别，导致到课率被低估。

### 3.4 EER (Equal Error Rate, 等错误率)

**定义**: 当FAR = FRR时的错误率。

**计算方法**:
```
1. 计算不同阈值T下的FAR(T)和FRR(T)
2. 绘制DET曲线 (FAR vs FRR)
3. FAR = FRR的交点即为EER
```

**数学表达**:
```
EER = FAR(T*) = FRR(T*)
其中 T* = argmin_T |FAR(T) - FRR(T)|
```

**课堂场景意义**: EER是生物特征识别系统的标准评估指标，它提供了一个与阈值无关的性能度量。EER越低，系统在不同应用场景下的适应性越好。

### 3.5 Rank-1 (首位命中率)

**定义**: 正确身份在所有候选身份中排名第一的比例。

**计算方法**:
```
1. 对每个查询人脸，计算与特征库中所有身份的相似度
2. 按相似度降序排列
3. 如果正确身份排在第一位，记为Rank-1命中
4. Rank-1 = Rank-1命中次数 / 总查询次数
```

**课堂场景意义**: Rank-1衡量识别系统的排序能力。在课堂场景中，即使阈值设定不当导致识别失败，高Rank-1意味着正确答案通常排在最前面，可以通过调整阈值改善。

---

## 4. 课堂状态分析评价指标

### 4.1 Overall Accuracy (总体分类准确率)

**定义**: 所有样本中被正确分类的比例。

**公式**:
```
Accuracy = N_correct / N_total
```

**课堂场景意义**: 最基本的状态分类指标，但在类别不平衡时可能具有误导性。

### 4.2 Per-class Precision / Recall / F1

**定义**: 对每个状态类别分别计算Precision、Recall和F1分数。

**公式**:
```
Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
```

**各状态类别的关注重点**:

| 状态 | 关注指标 | 原因 |
|------|---------|------|
| 专注 | Recall | 不应遗漏专注学生，否则低估课堂质量 |
| 低头 | Precision | 误判低头影响较大，可能错误标记认真记笔记的学生 |
| 离座 | F1 | 离座检测需要同时保证准确和完整 |
| 分心 | F1 | 分心判断需要平衡误判和漏判 |
| 举手 | Precision | 举手检测误报会干扰教师判断 |

### 4.3 Cohen's Kappa (一致性系数)

**定义**: 衡量分类结果与真值之间的一致性，消除了随机分类的影响。

**公式**:
```
Kappa = (Po - Pe) / (1 - Pe)

其中:
Po = 总体分类准确率 (Observed agreement)
Pe = 随机分类的期望一致率 (Expected agreement)

Pe = Σ (a_i * b_i) / N^2

a_i = 第i类真值样本数
b_i = 第i类预测样本数
N = 总样本数
```

**取值范围**: [-1, 1]

**解读标准**:
| Kappa值 | 一致性强度 |
|---------|-----------|
| < 0.00 | 无一致性 |
| 0.00-0.20 | 轻微一致 |
| 0.21-0.40 | 一般一致 |
| 0.41-0.60 | 中度一致 |
| 0.61-0.80 | 高度一致 |
| 0.81-1.00 | 几乎完全一致 |

**课堂场景意义**: Kappa比Accuracy更可靠，因为它考虑了类别分布。在课堂场景中，大部分时间学生处于"专注"状态，如果分类器总是预测"专注"，Accuracy可能很高但Kappa很低。

### 4.4 Confusion Matrix (混淆矩阵)

**定义**: N×N矩阵，展示各类别之间的分类情况。

**结构**:
```
                 预测
              专注  分心  低头  离座  举手
        专注 [  TP   FP   FP   FP   FP ]
真值    分心 [  FN   TP   FP   FP   FP ]
        低头 [  FN   FN   TP   FP   FP ]
        离座 [  FN   FN   FN   TP   FP ]
        举手 [  FN   FN   FN   FN   TP ]
```

**分析要点**:
1. **对角线元素**: 正确分类数，越大越好
2. **非对角线元素**: 误分类数，分析误分类模式
3. **常见误分类对**:
   - 专注↔分心: 视线方向判断边界模糊
   - 低头↔专注: 记笔记vs看手机难以区分
   - 分心↔举手: 举手动作初期可能被误判为侧转

---

## 5. 系统级评价指标

### 5.1 端到端FPS

**定义**: 从输入一帧图像到输出完整分析结果的帧率。

**测量方法**:
```python
total_time = 0
for frame in video:
    start = time.time()
    result = pipeline.process_frame(frame)
    total_time += time.time() - start
fps = num_frames / total_time
```

**性能目标**:
| 场景 | 最低FPS | 目标FPS |
|------|---------|---------|
| 小教室(20人) | 15 | 30 |
| 大教室(60人) | 10 | 20 |

### 5.2 各模块延迟

**定义**: 流水线中各模块的推理耗时。

**测量方法**:
```python
timings = {
    "preprocessing": 0,
    "detection": 0,
    "tracking": 0,
    "face_recognition": 0,
    "pose_estimation": 0,
    "state_analysis": 0,
    "visualization": 0,
}
```

**延迟预算** (30FPS目标，每帧33ms):
| 模块 | 预算(ms) | 占比 |
|------|---------|------|
| 预处理 | 2 | 6% |
| 检测 | 15 | 45% |
| 跟踪 | 3 | 9% |
| 人脸识别 | 5 | 15% |
| 姿态估计 | 5 | 15% |
| 状态分析 | 1 | 3% |
| 可视化 | 2 | 6% |

### 5.3 统计准确度

**定义**: 系统输出的统计指标与人工统计的偏差。

**指标**:
```
到课率偏差 = |系统到课率 - 人工到课率|
抬头率偏差 = |系统抬头率 - 人工抬头率|
离座次数偏差 = |系统离座次数 - 人工离座次数|
```

**可接受范围**:
| 指标 | 可接受偏差 |
|------|----------|
| 到课率 | ±5% |
| 抬头率 | ±10% |
| 离座次数 | ±20% |

---

## 6. 评估工具使用说明

### 6.1 检测评估

```bash
# 使用Ultralytics内置验证
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); model.val(data='classroom.yaml')"

# 使用自定义评估脚本
python tools/eval_detection.py --model best.pt --data data/annotations/val.yaml --output outputs/eval_det.json
```

### 6.2 跟踪评估

```bash
# 使用motmetrics评估
python tools/eval_tracking.py --gt data/annotations/gt_mot.txt --result outputs/tracking/result.txt
```

### 6.3 人脸识别评估

```bash
python tools/eval_face_recognition.py --db data/face_db --test data/face_db/test --output outputs/eval_face.json
```

### 6.4 状态分析评估

```bash
python tools/eval_state.py --gt data/annotations/gt_states.json --pred outputs/pred_states.json --output outputs/eval_state.json
```

### 6.5 完整评估

```bash
python scripts/run_evaluation.py --task all --output outputs/evaluation
```

---

## 7. 结果可视化

### 7.1 检测结果可视化

- **PR曲线**: Precision-Recall曲线
- **F1-Confidence曲线**: 不同置信度阈值下的F1分数
- **混淆矩阵**: 检测结果的混淆分析

### 7.2 跟踪结果可视化

- **轨迹图**: 所有跟踪轨迹在画面中的运动轨迹
- **ID Switch热力图**: 身份切换发生的位置分布

### 7.3 状态分析可视化

- **时序状态图**: 每个学生的状态随时间变化
- **统计趋势图**: 抬头率/到课率随时间变化
- **注意力热力图**: 专注学生的空间分布

### 7.4 系统性能可视化

- **延迟分布图**: 各模块延迟的箱线图
- **FPS趋势图**: 处理帧率随时间的变化
