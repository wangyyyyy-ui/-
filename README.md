# 🎓 面向课堂复杂场景的多目标视觉感知与学习状态分析研究

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Type](https://img.shields.io/badge/Type-系统开发%20%7C%20算法实验-blue)
![Lab](https://img.shields.io/badge/Lab-110_Lab-8A2BE2)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![License](https://img.shields.io/badge/License-MIT-red)

> **110实验室前沿技术分组研究** · 指导学长：周灿宇 / 带教教师：李腊全

本项目旨在研究如何在课堂视频中稳定完成**学生检测、目标跟踪、身份匹配、头部姿态估计、视线方向判断和课堂状态统计**。并非简单的"刷脸签到系统"，而是定位为课堂场景下的多人视觉感知与学习状态分析研究。


---

##  关于本项目

**技术层面** — 解决远距离下人脸模糊识别不清、人脸被遮挡的问题，促进计算机视觉在课堂下的应用研究。

**教育信息化** — 提供教学辅助性统计数据（到课人数、抬头比例、离座次数等），帮助教师了解课堂整体情况。

---

##  核心特性

-  **多人目标检测** — 基于 YOLOv10 / RT-DETR，对比选择最优实时检测方案
-  **多目标跟踪** — 基于 ByteTrack 实现跨帧身份保持与轨迹记录
-  **人脸识别与身份匹配** — 基于 ArcFace/InsightFace 构建人脸特征库并完成匹配
-  **头部姿态与视线估计** — 基于 OpenFace / MediaPipe 计算 pitch/yaw/roll 及视线方向
-  **课堂状态识别** — 综合判定抬头/低头、专注/分心、离座等状态
-  **视觉基础模型探索** — 评估 SAM 2 + DINOv2 在学生区域分割中的辅助作用
-  **可视化统计输出** — 输出到课人数、抬头比例、离座频率等教学辅助数据

---

##  技术架构

---

##  处理流程

### Step 1 · 视频采集与预处理

采集课堂视频数据，进行视频帧采样，以及对图像实现增强效果（亮度调整、去噪），为后续检测提供高质量输入。

### Step 2 · 多人目标检测

采用 YOLOv10 系列模型对每帧图像进行目标检测，对比不同模型在课堂场景下的表现，输出每个学生的检测框。

### Step 3 · 多目标跟踪

采用 ByteTrack 算法执行跨帧目标关联。核心策略是**同时保留高置信度和低置信度检测框**——前者构建可靠轨迹，后者在后续关联中匹配遮挡或模糊目标，减少目标丢失和身份切换。

### Step 4 · 身份匹配

构建学生人脸特征库（基于 ArcFace / InsightFace），对检测到的人脸区域提取特征并与库内特征进行相似度匹配，完成身份确认。

### Step 5 · 行为与状态分析

采用 OpenFace 或 MediaPipe 对每个学生进行头部姿态估计（计算 pitch、yaw、roll）和视线方向估计，结合人体姿态估计检测离座等身体状态，综合判定抬头/低头、专注/分心等学习状态。

### Step 6 · 时序融合与统计输出

设计滑动窗口或状态机进行多帧融合判定以减少误判，汇总分析结果，输出到课人数、抬头比例、离座统计等可视化图表。

---

##  核心算法模块

| 模块 | 技术方案 | 功能说明 |
|:-----|:---------|:---------|
| 目标检测 | YOLOv10 / RT-DETR | 课堂多人实时检测，对比精度与速度 |
| 多目标跟踪 | ByteTrack | 跨帧目标关联，维持遮挡目标轨迹连续性 |
| 人脸识别 | ArcFace / InsightFace | 人脸特征提取与身份匹配 |
| 姿态·视线估计 | OpenFace 3.0 / MediaPipe | 头部朝向、视线方向、动作单元识别 |
| 视觉基础模型 | SAM 2 + DINOv2 | 零样本学生区域分割，鲁棒特征提取 |
| 时序融合 | 滑动窗口 / 状态机 | 多帧融合判定，减少误判 |

---

##  研究综述

<details>
<summary> 点击展开完整研究综述</summary>

### （一）目标检测

YOLO项目自YOLOv1到YOLOv10不断迭代，在检测速度和精度间取得了良好平衡。Zikang等人在2025年提出的改进YOLOv10算法，通过引入Mosaic-9数据增强策略、BiFPN多尺度特征融合和SE通道注意力模块，在自建数据集上达到了**69.5%的mAP@0.5**，相较于YOLOv10n提升了7.7个百分点。

### （二）多目标跟踪

在课堂密集场景中多目标跟踪中解决遮挡、身份切换等问题仍待解决。Zhang等提出的**ByteTrack**算法通过同时利用高置信度和低置信度检测结果进行关联，能够有效维持遮挡目标的轨迹连续性。陈云芳等以ByteTrack为基线算法改进关联策略，在MOT17和MOT20测试集上HOTA指标分别达到**64.5%**和**63.2%**。

### （三）人脸识别

Deng等提出的**ArcFace**（Additive Angular Margin Loss）通过引入加法角度边际损失增强了特征的判别能力，其开源实现InsightFace已成为学术界和工业界广泛使用的人脸分析工具箱。

### （四）头部姿态与视线估计

OpenFace是首个集人脸关键点检测、头部姿态估计和视线估计于一体的开源工具包。最新版**OpenFace 3.0**采用轻量级多任务统一模型架构，可在多种头姿、光照条件和视频分辨率上实时运行。Google的**MediaPipe**同样提供了轻量级的人脸和姿态估计解决方案，适用于边缘端部署。

### （五）视觉基础模型

SAM 2是Meta推出的视频分割基础模型，DINOv2提供了自监督训练的通用视觉特征。将SAM与DINOv2结合可在零样本条件下实现高精度分类，有望提升学生区域分割的鲁棒性。

### （六）课堂行为分析

戚译丹等人提出了融合行为分析和情感分析的智能化评估方法，构建了大规模数据集，采用VGGNet16和ResNet50分别进行行为识别和表情识别。然而，现有研究多通过单一视觉信息识别学习投入的某一维度，**对多维度融合的探究尚不深入**。

</details>

---

##  快速开始
### 1. 环境准备
### 2. 准备模型权重
### 3. 构建人脸特征库
### 4. 运行完整分析
### 5. 启动 Demo 界面
---
##  评价指标

| 评价维度 | 具体指标 | 说明 |
|:---------|:---------|:-----|
| 目标检测 | Precision / Recall / mAP@0.5 | 检测模型的精度评价 |
| 多目标跟踪 | MOTA / ID Switch / 跟踪召回率 | 衡量跟踪性能与稳定性 |
| 人脸识别 | 识别准确率 / 误签率 / 漏签率 | 辅助评价课堂出勤率 |
| 行为识别 | 抬头/低头分类准确率 / 离座检测准确率 | 模型识别动作的精度 |

---

## 👥 团队成员

| 成员 | 学号 | 专业 | 负责模块 |
|:-----|:-----|:-----|:---------|
|  雷璐 | 2024213717 | 智能科学与技术与数学与应用数学 | 身份匹配模块（ArcFace 特征提取与身份匹配） |
|  王佳怡 | 2024213714 | 智能科学与技术与数学与应用数学 | 目标检测与跟踪模块（YOLOv10 + ByteTrack） |
|  王燕 | 2024213884 | 信息与计算科学 | 行为状态分析模块（OpenFace / MediaPipe） |
|  田旭冉 | 2024213883 | 信息与计算科学 | 前端可视化与系统集成 |


---

## 预期产出

1.  **完整算法流程** — 从原始视频输入到学生状态输出的全流程视觉处理逻辑
2.  **优化算法体系** — 兼顾高精度身份匹配、行为感知与高实时推理性能
3.  **Demo 系统** — 可展示视频输入、学生检测框与ID、到课人数/抬头比例/离座统计等可视化图表
4.  **实验评估报告** — 记录各模块在准确率、实时性等方面的性能数据
5.  **文档与复现指南** — 算法流程说明、模型训练配置、数据集划分、超参数、Demo使用指南

---

## 参考文献

1. Liu Q, Jiang X, Jiang R. Classroom behavior recognition using computer vision: A systematic review[J]. *Sensors*, 2025, 25(2): 373.
2. Zikang W, et al. Improved YOLOv10: A real-time object detection approach in complex environments[J]. *Sensors*, 2025, 25(22): 6893.
3. Zhang Y, Sun P, Jiang Y, et al. ByteTrack: Multi-object tracking by associating every detection box[C]// *ECCV*, 2022.
4. Deng J, Guo J, Xue N, et al. ArcFace: Additive angular margin loss for deep face recognition[J]. *IEEE TPAMI*, 2019.
5. Baltrusaitis T, Zadeh A, Lim Y C, et al. OpenFace 2.0: Facial behavior analysis toolkit[C]// *IEEE FG 2018*.
6. Hu J, Mathur L, Liang P P, et al. OpenFace 3.0: A lightweight multitask system for comprehensive facial behavior analysis[J]. *arXiv:2506.02891*, 2025.
7. Barnatan F, et al. Zero-shot shape classification using vision foundation models[J]. *arXiv:2508.03235*, 2025.
8. 陈云芳, 方倩, 吕尊威, 等. 关联策略多特征增强的多目标跟踪[J]. *计算机科学*, 2026, 53(3): 231-239.
9. 戚译丹, 等. 融合行为与情感分析的课堂学习投入智能评估[J]. *武汉大学学报（理学版）*, 2025, 71(5): 621-633.
10. Yang R, et al. Research on improving students concentration by task-oriented method[J]. *Education and Information Technologies*, 2025.
11. Kirillov A, et al. Segment anything[J]. *arXiv:2304.02643*, 2023.
12. Oquab M, et al. DINOv2: Learning robust visual features without supervision[J]. *arXiv:2304.07193*, 2023.
---

<p align="center">
  Made with ❤️ by 110 Lab · Front-Tech Group 6 · © 2025
</p>
