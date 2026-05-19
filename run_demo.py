"""
Demo演示脚本

使用合成数据或示例视频快速演示系统功能
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.yolo_detector import YOLODetector, Detection, DetectionResult
from src.tracking.byte_tracker import ByteTrackWrapper, Track, TrackingResult
from src.state_analysis.state_classifier import StateClassifier, StudentState
from src.state_analysis.temporal_fusion import TemporalFusion, SmoothedState, ClassroomStatistics
from src.visualization.drawer import VisualizationDrawer


def generate_demo_frame(frame_id: int, width: int = 1280, height: int = 720) -> np.ndarray:
    """生成模拟课堂场景的演示帧"""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 40

    # 绘制教室背景
    cv2.rectangle(frame, (0, 0), (width, height), (60, 60, 80), -1)
    cv2.rectangle(frame, (50, 50), (width - 50, 200), (80, 80, 100), -1)  # 黑板
    cv2.putText(frame, "Classroom Demo", (width // 2 - 120, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)

    # 模拟学生位置（3行4列）
    for row in range(3):
        for col in range(4):
            x = 150 + col * 250
            y = 280 + row * 140
            # 绘制简笔人形
            cv2.circle(frame, (x, y - 30), 20, (180, 180, 200), -1)  # 头
            cv2.rectangle(frame, (x - 25, y - 10), (x + 25, y + 40), (100, 100, 150), -1)  # 身体

    # 添加帧号
    cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def run_demo():
    """运行Demo"""
    print("=" * 60)
    print("课堂多目标视觉感知与学习状态分析 - Demo演示")
    print("=" * 60)

    # 初始化模块
    print("\n初始化可视化模块...")
    drawer = VisualizationDrawer(
        show_bbox=True, show_id=True, show_state=True, show_stats=True
    )
    classifier = StateClassifier()
    fusion = TemporalFusion(fps=30.0)

    print("开始Demo演示 (按Q退出)...\n")

    # 模拟12个学生
    num_students = 12
    track_ids = list(range(1, num_students + 1))

    for frame_id in range(300):
        frame = generate_demo_frame(frame_id)

        # 模拟跟踪结果
        tracks = []
        for i, tid in enumerate(track_ids):
            row = i // 4
            col = i % 4
            x = 150 + col * 250
            y = 280 + row * 140

            # 模拟一些学生离开
            if frame_id > 100 and tid == 3:
                continue
            if frame_id > 200 and tid == 7:
                continue

            tracks.append(Track(
                track_id=tid,
                bbox=np.array([x - 25, y - 50, x + 25, y + 40], dtype=np.float32),
                confidence=0.9,
                class_id=0,
                class_name="person",
                frame_id=frame_id,
            ))

        # 模拟状态分类
        state_results = []
        for track in tracks:
            tid = track.track_id
            # 模拟不同状态
            if tid in [2, 5, 8, 11]:
                state = StudentState.FOCUSED
            elif tid in [4, 9]:
                state = StudentState.HEAD_DOWN
            elif tid in [6]:
                state = StudentState.DISTRACTED
            elif tid in [10]:
                state = StudentState.HAND_RAISING
            else:
                state = StudentState.FOCUSED

            from src.state_analysis.state_classifier import StateResult
            state_results.append(StateResult(
                track_id=tid,
                state=state,
                confidence=0.8,
                identity=f"Student_{tid:02d}",
            ))

        # 时序融合
        smoothed_states, stats = fusion.update(
            state_results, frame_id, frame_id / 30.0,
            registered_students=num_students,
        )

        # 可视化
        vis_frame = drawer.draw_frame(frame, tracks, smoothed_states, stats)

        cv2.imshow("Classroom Analysis Demo", vis_frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        if frame_id % 50 == 0:
            print(f"  帧 {frame_id}: 到课{stats.total_students - stats.left_seat_count}人, "
                  f"抬头率{stats.focus_rate:.1%}")

    cv2.destroyAllWindows()
    print("\nDemo演示结束!")


if __name__ == "__main__":
    run_demo()
