"""
头部姿态与视线估计模块

基于MediaPipe Face Mesh实现头部姿态估计（pitch, yaw, roll）
和视线方向判断。

在课堂场景中，头部姿态是判断学生注意力状态的重要指标:
    - pitch < -15°: 低头（看桌面/手机）
    - pitch > 10°: 抬头（看天花板）
    - |yaw| > 30°: 侧转（看旁边同学）
    - |pitch| < 15° 且 |yaw| < 30°: 正视前方（专注听课）

参考:
    - MediaPipe Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh
    - OpenFace 2.0/3.0: Baltrusaitis et al.
"""

import numpy as np
import cv2
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class HeadDirection(Enum):
    """头部朝向枚举"""
    FORWARD = "forward"       # 正视前方
    DOWN = "down"             # 低头
    UP = "up"                 # 抬头
    LEFT = "left"             # 左转
    RIGHT = "right"           # 右转
    UNKNOWN = "unknown"       # 无法判断


class GazeDirection(Enum):
    """视线方向枚举"""
    ON_SCREEN = "on_screen"       # 看屏幕/黑板
    ON_DESK = "on_desk"           # 看桌面
    AWAY = "away"                 # 视线偏离
    UNKNOWN = "unknown"           # 无法判断


@dataclass
class HeadPoseResult:
    """头部姿态估计结果"""
    pitch: float               # 俯仰角（度），正值向上
    yaw: float                 # 偏航角（度），正值向右
    roll: float                # 翻滚角（度），正值向右倾
    head_direction: HeadDirection  # 头部朝向
    gaze_direction: GazeDirection  # 视线方向
    confidence: float          # 估计置信度
    landmarks: Optional[np.ndarray] = None  # 面部关键点


class HeadPoseEstimator:
    """
    头部姿态与视线估计器
    
    使用MediaPipe Face Mesh提取468个面部关键点，
    通过PnP算法求解头部姿态角(pitch, yaw, roll)，
    并根据角度阈值判断头部朝向和视线方向。
    
    Args:
        method: 估计方法 ('mediapipe' 或 'opencv')
        pitch_down_thresh: 低头判定阈值（度）
        pitch_up_thresh: 抬头判定阈值（度）
        yaw_thresh: 侧转判定阈值（度）
        model_complexity: MediaPipe模型复杂度 (0/1)
        min_detection_confidence: 最小检测置信度
        min_tracking_confidence: 最小跟踪置信度
    """

    def __init__(
        self,
        method: str = "mediapipe",
        pitch_down_thresh: float = 15.0,
        pitch_up_thresh: float = 10.0,
        yaw_thresh: float = 30.0,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.method = method
        self.pitch_down_thresh = pitch_down_thresh
        self.pitch_up_thresh = pitch_up_thresh
        self.yaw_thresh = yaw_thresh

        self._face_mesh = None
        self._mp_face_mesh = None

        if method == "mediapipe":
            self._init_mediapipe(
                model_complexity, min_detection_confidence, min_tracking_confidence
            )

    def _init_mediapipe(self, model_complexity, det_conf, track_conf):
        """初始化MediaPipe Face Mesh"""
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
                static_image_mode=False,
            )
            print("[HeadPoseEstimator] MediaPipe Face Mesh初始化成功")
        except ImportError:
            print("[HeadPoseEstimator] mediapipe未安装，使用OpenCV DNN后备方案")
            self.method = "opencv"

    def estimate(self, image: np.ndarray, face_bbox: Optional[np.ndarray] = None) -> List[HeadPoseResult]:
        """
        估计图像中所有可见人脸的头部姿态
        
        Args:
            image: 输入BGR图像
            face_bbox: 可选的人脸边界框 [x1,y1,x2,y2]，用于限定检测区域
            
        Returns:
            头部姿态结果列表
        """
        if self.method == "mediapipe" and self._face_mesh is not None:
            return self._estimate_with_mediapipe(image)
        else:
            return self._estimate_with_opencv(image, face_bbox)

    def _estimate_with_mediapipe(self, image: np.ndarray) -> List[HeadPoseResult]:
        """使用MediaPipe估计头部姿态"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        pose_results = []
        if results.multi_face_landmarks:
            h, w = image.shape[:2]

            for face_landmarks in results.multi_face_landmarks:
                # 提取关键点
                landmarks_2d = []
                landmarks_3d = []

                # 使用面部关键点子集进行PnP求解
                # 鼻尖、下巴、左眼外角、右眼外角、左嘴角、右嘴角
                key_indices = [1, 152, 33, 263, 61, 291]

                for idx in key_indices:
                    lm = face_landmarks.landmark[idx]
                    landmarks_2d.append([lm.x * w, lm.y * h])
                    # 3D模型坐标（归一化）
                    landmarks_3d.append([lm.x, lm.y, lm.z])

                landmarks_2d = np.array(landmarks_2d, dtype=np.float64)
                landmarks_3d = np.array(landmarks_3d, dtype=np.float64)

                # PnP求解姿态
                pitch, yaw, roll = self._solve_pose(landmarks_2d, landmarks_3d, (w, h))

                # 判断头部朝向
                head_direction = self._classify_head_direction(pitch, yaw)

                # 判断视线方向
                gaze_direction = self._classify_gaze_direction(pitch, yaw)

                pose_results.append(HeadPoseResult(
                    pitch=pitch,
                    yaw=yaw,
                    roll=roll,
                    head_direction=head_direction,
                    gaze_direction=gaze_direction,
                    confidence=0.9,
                    landmarks=np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark]),
                ))

        return pose_results

    def _estimate_with_opencv(self, image: np.ndarray, face_bbox: Optional[np.ndarray]) -> List[HeadPoseResult]:
        """使用OpenCV DNN后备方案估计头部姿态"""
        # 简化实现：基于人脸位置在画面中的相对位置粗略估计
        if face_bbox is not None:
            h, w = image.shape[:2]
            cx = (face_bbox[0] + face_bbox[2]) / 2
            cy = (face_bbox[1] + face_bbox[3]) / 2

            # 粗略估计偏航角
            yaw = (cx / w - 0.5) * 60  # 映射到-30~30度
            # 粗略估计俯仰角
            pitch = (0.5 - cy / h) * 40  # 映射到-20~20度

            head_direction = self._classify_head_direction(pitch, yaw)
            gaze_direction = self._classify_gaze_direction(pitch, yaw)

            return [HeadPoseResult(
                pitch=pitch,
                yaw=yaw,
                roll=0.0,
                head_direction=head_direction,
                gaze_direction=gaze_direction,
                confidence=0.5,
            )]

        return []

    def _solve_pose(
        self, landmarks_2d: np.ndarray, landmarks_3d: np.ndarray, image_size: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        使用PnP算法求解头部姿态角
        
        Args:
            landmarks_2d: 2D关键点坐标
            landmarks_3d: 3D模型坐标
            image_size: 图像尺寸 (W, H)
            
        Returns:
            (pitch, yaw, roll) 欧拉角（度）
        """
        w, h = image_size

        # 相机内参矩阵（近似）
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 求解PnP
        success, rvec, tvec = cv2.solvePnP(
            landmarks_3d, landmarks_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # 旋转向量转欧拉角
        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rmat)

        return pitch, yaw, roll

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """旋转矩阵转欧拉角"""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            roll = 0

        return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

    def _classify_head_direction(self, pitch: float, yaw: float) -> HeadDirection:
        """
        根据pitch和yaw角分类头部朝向
        
        判定规则:
            - pitch < -pitch_down_thresh: 低头
            - pitch > pitch_up_thresh: 抬头
            - yaw < -yaw_thresh: 左转
            - yaw > yaw_thresh: 右转
            - 其他: 正视前方
        """
        if pitch < -self.pitch_down_thresh:
            return HeadDirection.DOWN
        elif pitch > self.pitch_up_thresh:
            return HeadDirection.UP
        elif yaw < -self.yaw_thresh:
            return HeadDirection.LEFT
        elif yaw > self.yaw_thresh:
            return HeadDirection.RIGHT
        else:
            return HeadDirection.FORWARD

    def _classify_gaze_direction(self, pitch: float, yaw: float) -> GazeDirection:
        """
        根据头部姿态推断视线方向
        
        判定规则:
            - 低头 + 正视: 看桌面
            - 正视前方: 看屏幕/黑板
            - 侧转或抬头: 视线偏离
        """
        if pitch < -self.pitch_down_thresh and abs(yaw) < self.yaw_thresh:
            return GazeDirection.ON_DESK
        elif abs(pitch) < self.pitch_down_thresh and abs(yaw) < self.yaw_thresh:
            return GazeDirection.ON_SCREEN
        else:
            return GazeDirection.AWAY

    def estimate_for_tracks(
        self, image: np.ndarray, tracks: list
    ) -> Dict[int, HeadPoseResult]:
        """
        为跟踪目标估计头部姿态
        
        Args:
            image: 当前帧BGR图像
            tracks: 跟踪目标列表
            
        Returns:
            {track_id: HeadPoseResult} 映射
        """
        all_poses = self.estimate(image)

        if len(all_poses) == 0:
            return {}

        # 将姿态估计结果与跟踪目标关联
        result_map = {}
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            track_cx = (x1 + x2) / 2
            track_cy = (y1 + y2) / 2

            best_pose = None
            best_dist = float('inf')

            for pose in all_poses:
                if pose.landmarks is not None and len(pose.landmarks) > 1:
                    # 使用鼻尖关键点（索引1）作为人脸中心
                    nose = pose.landmarks[1]
                    dist = np.sqrt((nose[0] - track_cx) ** 2 + (nose[1] - track_cy) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_pose = pose

            if best_pose is not None and best_dist < max(x2 - x1, y2 - y1):
                result_map[track.track_id] = best_pose

        return result_map
