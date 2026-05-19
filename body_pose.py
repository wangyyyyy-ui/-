"""
身体姿态估计模块

基于MediaPipe Pose实现身体关键点检测，
用于辅助判断学生课堂行为（如举手、起立、伏桌等）。

在课堂场景中，身体姿态是学习状态分析的重要补充:
    - 举手: 右手/左手高于肩膀
    - 起立: 臀部关键点高于一定阈值
    - 伏桌: 肩膀低于桌面高度
    - 正常坐姿: 肩膀和臀部在正常范围内
"""

import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class BodyAction(Enum):
    """身体动作枚举"""
    SITTING = "sitting"         # 正常坐姿
    STANDING = "standing"       # 起立
    HAND_RAISING = "hand_raising"  # 举手
    LEANING = "leaning"         # 伏桌/前倾
    UNKNOWN = "unknown"         # 无法判断


@dataclass
class BodyPoseResult:
    """身体姿态估计结果"""
    action: BodyAction          # 识别的动作
    confidence: float           # 置信度
    keypoints: Optional[np.ndarray] = None  # 身体关键点
    bbox: Optional[np.ndarray] = None       # 人体边界框


class BodyPoseEstimator:
    """
    身体姿态估计器
    
    使用MediaPipe Pose检测身体关键点，
    基于关键点空间关系判断学生课堂行为。
    
    Args:
        model_complexity: MediaPipe模型复杂度 (0/1/2)
        min_detection_confidence: 最小检测置信度
        min_tracking_confidence: 最小跟踪置信度
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._pose = None
        self._mp_pose = None

        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            print("[BodyPoseEstimator] MediaPipe Pose初始化成功")
        except ImportError:
            print("[BodyPoseEstimator] mediapipe未安装，身体姿态估计不可用")

    def estimate(self, image: np.ndarray, person_bbox: Optional[np.ndarray] = None) -> Optional[BodyPoseResult]:
        """
        估计图像中人体的身体姿态
        
        Args:
            image: 输入BGR图像
            person_bbox: 可选的人体边界框 [x1,y1,x2,y2]
            
        Returns:
            身体姿态结果，无人时返回None
        """
        if self._pose is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return None

        h, w = image.shape[:2]
        landmarks = results.pose_landmarks.landmark

        # 提取关键点坐标
        keypoints = np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in landmarks])

        # 分类动作
        action = self._classify_action(keypoints, h)

        return BodyPoseResult(
            action=action,
            confidence=0.8,
            keypoints=keypoints,
            bbox=person_bbox,
        )

    def _classify_action(self, keypoints: np.ndarray, image_height: int) -> BodyAction:
        """
        根据关键点分类身体动作
        
        MediaPipe Pose关键点索引:
            11: 左肩, 12: 右肩
            13: 左肘, 14: 右肘
            15: 左腕, 16: 右腕
            23: 左髋, 24: 右髋
            25: 左膝, 26: 右膝
        """
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_wrist = keypoints[15]
        right_wrist = keypoints[16]
        left_hip = keypoints[23]
        right_hip = keypoints[24]

        # 检查举手: 手腕高于肩膀
        if (left_wrist[2] > 0.5 and left_wrist[1] < left_shoulder[1] - 30) or \
           (right_wrist[2] > 0.5 and right_wrist[1] < right_shoulder[1] - 30):
            return BodyAction.HAND_RAISING

        # 检查起立: 髋部位置偏高
        hip_y = (left_hip[1] + right_hip[1]) / 2
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        if hip_y < image_height * 0.4:
            return BodyAction.STANDING

        # 检查伏桌: 肩膀位置偏低
        if shoulder_y > image_height * 0.7:
            return BodyAction.LEANING

        return BodyAction.SITTING
