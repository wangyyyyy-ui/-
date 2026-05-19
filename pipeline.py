"""
主流水线模块

将所有子模块串联为完整的课堂分析流水线:
    视频输入 -> 预处理 -> 目标检测 -> 多目标跟踪 -> 
    人脸识别 -> 头部姿态估计 -> 状态分类 -> 时序融合 -> 
    可视化输出 -> 统计导出

流水线支持两种运行模式:
    1. 实时模式: 处理摄像头或RTSP流，实时显示结果
    2. 离线模式: 处理视频文件，输出分析结果和标注视频
"""

import cv2
import time
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

from .preprocessing.video_loader import VideoLoader
from .preprocessing.image_enhance import ImageEnhancer
from .detection.yolo_detector import YOLODetector, DetectionResult
from .tracking.byte_tracker import ByteTrackWrapper, TrackingResult, Track
from .face_recognition.arcface import ArcFaceRecognizer, MatchResult
from .face_recognition.face_db import FaceDatabase
from .pose_estimation.head_pose import HeadPoseEstimator, HeadPoseResult
from .pose_estimation.body_pose import BodyPoseEstimator, BodyPoseResult
from .state_analysis.state_classifier import StateClassifier, StateResult, StudentState
from .state_analysis.temporal_fusion import TemporalFusion, SmoothedState, ClassroomStatistics
from .visualization.drawer import VisualizationDrawer
from .visualization.statistics import StatisticsExporter


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 视频配置
    source: str = "0"                    # 视频源
    sample_interval: int = 1             # 帧采样间隔
    target_size: Optional[tuple] = None  # 目标分辨率

    # 检测配置
    detector_model: str = "yolov10s"     # 检测模型
    det_conf_threshold: float = 0.25     # 检测置信度阈值
    det_iou_threshold: float = 0.45      # NMS IoU阈值
    det_device: str = "cuda"             # 检测设备

    # 跟踪配置
    track_thresh: float = 0.5            # 高置信度阈值
    track_buffer: int = 30               # 轨迹缓冲帧数
    match_thresh: float = 0.8            # 匹配IoU阈值

    # 人脸识别配置
    face_model: str = "buffalo_l"        # InsightFace模型
    face_threshold: float = 0.4          # 匹配阈值
    face_db_path: str = "data/face_db"   # 特征库路径

    # 姿态估计配置
    pose_method: str = "mediapipe"       # 姿态估计方法

    # 状态分析配置
    head_down_pitch: float = 15.0        # 低头阈值
    distracted_yaw: float = 30.0         # 分心阈值

    # 时序融合配置
    fusion_window: int = 15              # 滑动窗口大小
    leave_seat_frames: int = 30          # 离座判定帧数

    # 输出配置
    output_dir: str = "outputs"          # 输出目录
    save_video: bool = True              # 是否保存标注视频
    save_stats: bool = True              # 是否保存统计数据
    display: bool = True                 # 是否实时显示
    display_fps: bool = True             # 是否显示FPS


class ClassroomPipeline:
    """
    课堂分析主流水线
    
    串联所有子模块，实现从视频输入到分析输出的完整流程。
    
    Args:
        config: 流水线配置
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._init_modules()
        self._frame_count = 0
        self._start_time = None

    def _init_modules(self):
        """初始化所有子模块"""
        cfg = self.config

        print("=" * 60)
        print("课堂多目标视觉感知与学习状态分析系统")
        print("=" * 60)
        print("\n[1/7] 初始化视频加载器...")
        self.video_loader = VideoLoader(
            source=cfg.source,
            sample_interval=cfg.sample_interval,
            target_size=cfg.target_size,
        )

        print("[2/7] 初始化图像增强器...")
        self.enhancer = ImageEnhancer()

        print("[3/7] 初始化目标检测器...")
        self.detector = YOLODetector(
            model_name=cfg.detector_model,
            conf_threshold=cfg.det_conf_threshold,
            iou_threshold=cfg.det_iou_threshold,
            device=cfg.det_device,
        )

        print("[4/7] 初始化多目标跟踪器...")
        self.tracker = ByteTrackWrapper(
            track_thresh=cfg.track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            frame_rate=self.video_loader.fps,
        )

        print("[5/7] 初始化人脸识别器...")
        self.face_recognizer = ArcFaceRecognizer(
            model_name=cfg.face_model,
            threshold=cfg.face_threshold,
            device=cfg.det_device,
        )
        self.face_db = FaceDatabase(
            db_path=cfg.face_db_path,
            recognizer=self.face_recognizer,
        )

        print("[6/7] 初始化姿态估计器...")
        self.head_pose_estimator = HeadPoseEstimator(
            method=cfg.pose_method,
            pitch_down_thresh=cfg.head_down_pitch,
            yaw_thresh=cfg.distracted_yaw,
        )
        self.body_pose_estimator = BodyPoseEstimator()

        print("[7/7] 初始化状态分析器...")
        self.state_classifier = StateClassifier(
            head_down_pitch_thresh=cfg.head_down_pitch,
            distracted_yaw_thresh=cfg.distracted_yaw,
        )
        self.temporal_fusion = TemporalFusion(
            window_size=cfg.fusion_window,
            fps=self.video_loader.fps,
            leave_seat_threshold=cfg.leave_seat_frames,
        )

        self.drawer = VisualizationDrawer()
        self.exporter = StatisticsExporter(output_dir=cfg.output_dir)

        print("\n所有模块初始化完成!")
        print("=" * 60)

    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> Dict[str, Any]:
        """
        处理单帧图像
        
        Args:
            frame: BGR图像
            frame_id: 帧ID
            timestamp: 时间戳
            
        Returns:
            处理结果字典
        """
        # Step 1: 图像增强
        enhanced = self.enhancer.enhance(frame)

        # Step 2: 目标检测
        det_result = self.detector.detect(enhanced)

        # Step 3: 多目标跟踪
        track_result = self.tracker.update(det_result, frame_id)

        # Step 4: 人脸识别与身份匹配
        match_results = self.face_recognizer.match_tracks(
            enhanced, track_result.tracks, self.face_db
        )

        # Step 5: 头部姿态估计
        head_poses = self.head_pose_estimator.estimate_for_tracks(
            enhanced, track_result.tracks
        )

        # Step 6: 身体姿态估计（可选，较耗时）
        body_poses = {}
        for track in track_result.tracks:
            body_pose = self.body_pose_estimator.estimate(enhanced, track.bbox)
            if body_pose is not None:
                body_poses[track.track_id] = body_pose

        # Step 7: 状态分类
        identities = {r.track_id: r.identity for r in match_results if r.matched}
        state_results = self.state_classifier.classify_batch(
            tracks=track_result.tracks,
            head_poses=head_poses,
            body_poses=body_poses,
            lost_track_ids=[],
            identities=identities,
        )

        # Step 8: 时序融合
        smoothed_states, stats = self.temporal_fusion.update(
            state_results, frame_id, timestamp,
            registered_students=len(self.face_db),
        )

        return {
            "detection": det_result,
            "tracking": track_result,
            "face_matching": match_results,
            "head_poses": head_poses,
            "body_poses": body_poses,
            "state_results": state_results,
            "smoothed_states": smoothed_states,
            "statistics": stats,
        }

    def run(self) -> Dict[str, Any]:
        """
        运行完整流水线
        
        Returns:
            运行结果摘要
        """
        cfg = self.config
        self._start_time = time.time()

        # 视频写入器
        video_writer = None
        if cfg.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = Path(cfg.output_dir) / "output_annotated.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(output_path), fourcc, self.video_loader.fps,
                (self.video_loader.width, self.video_loader.height),
            )

        print(f"\n开始处理视频: {cfg.source}")
        print(f"视频信息: {self.video_loader.get_info()}")
        print("-" * 60)

        for frame_info in self.video_loader:
            self._frame_count += 1

            # 处理帧
            result = self.process_frame(
                frame_info.image, frame_info.frame_id, frame_info.timestamp
            )

            # 可视化
            vis_image = self.drawer.draw_frame(
                frame_info.image,
                result["tracking"].tracks,
                result["smoothed_states"],
                result["statistics"],
                result["head_poses"],
            )

            # 显示FPS
            if cfg.display_fps and self._start_time is not None:
                elapsed = time.time() - self._start_time
                fps = self._frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(
                    vis_image, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA,
                )

            # 保存视频
            if video_writer is not None:
                video_writer.write(vis_image)

            # 实时显示
            if cfg.display:
                cv2.imshow("Classroom Analysis", vis_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n用户中断处理")
                    break

            # 进度输出
            if self._frame_count % 100 == 0:
                stats = result["statistics"]
                print(
                    f"  帧 {self._frame_count}: "
                    f"到课{stats.total_students - stats.left_seat_count}人, "
                    f"抬头率{stats.focus_rate:.1%}, "
                    f"到课率{stats.attendance_rate:.1%}"
                )

        # 释放资源
        if video_writer is not None:
            video_writer.release()
        if cfg.display:
            cv2.destroyAllWindows()

        # 导出统计
        output_files = {}
        if cfg.save_stats:
            stats_history = self.temporal_fusion.get_statistics_history()
            leave_events = self.temporal_fusion.get_leave_seat_events()

            output_files["csv"] = self.exporter.export_csv(stats_history)
            output_files["json"] = self.exporter.export_json(stats_history, leave_events)
            output_files["plots"] = self.exporter.export_plots(stats_history)

        # 汇总
        elapsed = time.time() - self._start_time
        summary = {
            "total_frames": self._frame_count,
            "elapsed_time": round(elapsed, 2),
            "avg_fps": round(self._frame_count / elapsed, 2) if elapsed > 0 else 0,
            "output_files": output_files,
        }

        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"  总帧数: {summary['total_frames']}")
        print(f"  耗时: {summary['elapsed_time']}秒")
        print(f"  平均FPS: {summary['avg_fps']}")
        for key, path in output_files.items():
            if isinstance(path, list):
                for p in path:
                    print(f"  输出[{key}]: {p}")
            else:
                print(f"  输出[{key}]: {path}")
        print("=" * 60)

        return summary
