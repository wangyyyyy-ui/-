"""
可视化绘制模块

在视频帧上绘制:
    - 检测框与跟踪ID
    - 身份标签
    - 头部姿态方向指示
    - 课堂状态标签（颜色编码）
    - 课堂统计信息面板
    - 注意力热力图
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..state_analysis.state_classifier import StudentState
from ..state_analysis.temporal_fusion import SmoothedState, ClassroomStatistics
from ..tracking.byte_tracker import Track


# 状态颜色映射 (BGR格式)
STATE_COLORS = {
    StudentState.FOCUSED: (0, 200, 0),       # 绿色 - 专注
    StudentState.DISTRACTED: (0, 165, 255),   # 橙色 - 分心
    StudentState.HEAD_DOWN: (0, 0, 255),      # 红色 - 低头
    StudentState.LEFT_SEAT: (128, 128, 128),  # 灰色 - 离座
    StudentState.HAND_RAISING: (255, 255, 0), # 青色 - 举手
    StudentState.UNKNOWN: (255, 255, 255),     # 白色 - 未知
}

# 状态中文名称
STATE_NAMES_CN = {
    StudentState.FOCUSED: "专注",
    StudentState.DISTRACTED: "分心",
    StudentState.HEAD_DOWN: "低头",
    StudentState.LEFT_SEAT: "离座",
    StudentState.HAND_RAISING: "举手",
    StudentState.UNKNOWN: "未知",
}


class VisualizationDrawer:
    """
    可视化绘制器
    
    在视频帧上绘制检测结果、跟踪信息、状态标签和统计面板。
    
    Args:
        show_bbox: 是否显示检测框
        show_id: 是否显示跟踪ID
        show_identity: 是否显示身份标签
        show_state: 是否显示状态标签
        show_pose: 是否显示头部姿态
        show_stats: 是否显示统计面板
        show_heatmap: 是否显示注意力热力图
        font_scale: 字体大小
        line_thickness: 线条粗细
    """

    def __init__(
        self,
        show_bbox: bool = True,
        show_id: bool = True,
        show_identity: bool = True,
        show_state: bool = True,
        show_pose: bool = False,
        show_stats: bool = True,
        show_heatmap: bool = False,
        font_scale: float = 0.6,
        line_thickness: int = 2,
    ):
        self.show_bbox = show_bbox
        self.show_id = show_id
        self.show_identity = show_identity
        self.show_state = show_state
        self.show_pose = show_pose
        self.show_stats = show_stats
        self.show_heatmap = show_heatmap
        self.font_scale = font_scale
        self.line_thickness = line_thickness

        # 热力图累积
        self._heatmap_accum = None

    def draw_frame(
        self,
        image: np.ndarray,
        tracks: List[Track],
        smoothed_states: List[SmoothedState],
        stats: Optional[ClassroomStatistics] = None,
        head_poses: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        在单帧图像上绘制所有可视化信息
        
        Args:
            image: 原始BGR图像
            tracks: 跟踪目标列表
            smoothed_states: 平滑状态列表
            stats: 课堂统计信息
            head_poses: 头部姿态结果
            
        Returns:
            绘制后的图像
        """
        canvas = image.copy()

        # 构建状态映射
        state_map = {s.track_id: s for s in smoothed_states}

        # 绘制检测框和标签
        for track in tracks:
            tid = track.track_id
            state = state_map.get(tid)

            # 获取颜色
            if state is not None:
                color = STATE_COLORS.get(state.state, (255, 255, 255))
            else:
                color = (255, 255, 255)

            x1, y1, x2, y2 = map(int, track.bbox)

            # 绘制边界框
            if self.show_bbox:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.line_thickness)

            # 构建标签文本
            label_parts = []
            if self.show_id:
                label_parts.append(f"ID:{tid}")
            if self.show_identity and state and state.identity:
                label_parts.append(state.identity)
            if self.show_state and state:
                label_parts.append(STATE_NAMES_CN.get(state.state, "?"))

            if label_parts:
                label = " | ".join(label_parts)
                # 绘制标签背景
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                )
                cv2.rectangle(
                    canvas,
                    (x1, y1 - th - 8),
                    (x1 + tw + 4, y1),
                    color, -1
                )
                cv2.putText(
                    canvas, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255) if sum(color) < 400 else (0, 0, 0),
                    1, cv2.LINE_AA,
                )

        # 绘制统计面板
        if self.show_stats and stats is not None:
            canvas = self._draw_stats_panel(canvas, stats)

        # 绘制热力图
        if self.show_heatmap:
            canvas = self._draw_heatmap(canvas, tracks, state_map)

        return canvas

    def _draw_stats_panel(
        self, image: np.ndarray, stats: ClassroomStatistics
    ) -> np.ndarray:
        """绘制课堂统计信息面板"""
        h, w = image.shape[:2]
        panel_w = 280
        panel_h = 200

        # 半透明面板
        overlay = image.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, 10), (w - 10, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # 绘制文字
        x = w - panel_w
        y = 35
        dy = 28

        texts = [
            f"到课人数: {stats.total_students - stats.left_seat_count}",
            f"专注: {stats.focused_count}  分心: {stats.distracted_count}",
            f"低头: {stats.head_down_count}  举手: {stats.hand_raising_count}",
            f"离座: {stats.left_seat_count}",
            f"抬头率: {stats.focus_rate:.1%}",
            f"到课率: {stats.attendance_rate:.1%}",
        ]

        for text in texts:
            cv2.putText(
                image, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            y += dy

        return image

    def _draw_heatmap(
        self, image: np.ndarray, tracks: List[Track], state_map: Dict
    ) -> np.ndarray:
        """绘制注意力热力图"""
        h, w = image.shape[:2]

        if self._heatmap_accum is None:
            self._heatmap_accum = np.zeros((h, w), dtype=np.float32)

        # 在专注目标位置累积
        for track in tracks:
            state = state_map.get(track.track_id)
            if state and state.state == StudentState.FOCUSED:
                x1, y1, x2, y2 = map(int, track.bbox)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(self._heatmap_accum, (cx, cy), 30, 1.0, -1)

        # 生成热力图
        heatmap = cv2.GaussianBlur(self._heatmap_accum, (51, 51), 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # 叠加热力图
        overlay = cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)
        return overlay

    def reset_heatmap(self):
        """重置热力图"""
        self._heatmap_accum = None
