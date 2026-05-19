"""
多目标跟踪模块单元测试
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.byte_tracker import ByteTrackWrapper, Track, TrackingResult
from src.detection.yolo_detector import Detection, DetectionResult


class TestTracking:
    """多目标跟踪测试类"""

    @pytest.fixture
    def tracker(self):
        """创建跟踪器实例"""
        return ByteTrackWrapper(track_thresh=0.5, track_buffer=30)

    @pytest.fixture
    def sample_detections(self):
        """创建模拟检测结果"""
        dets = [
            Detection(
                bbox=np.array([100, 200, 150, 350], dtype=np.float32),
                confidence=0.9, class_id=0, class_name="person",
            ),
            Detection(
                bbox=np.array([300, 200, 350, 350], dtype=np.float32),
                confidence=0.85, class_id=0, class_name="person",
            ),
            Detection(
                bbox=np.array([500, 200, 550, 350], dtype=np.float32),
                confidence=0.7, class_id=0, class_name="person",
            ),
        ]
        return DetectionResult(
            detections=dets,
            inference_time=10.0,
            image_shape=(720, 1280),
        )

    def test_tracker_creation(self, tracker):
        """测试跟踪器创建"""
        assert tracker is not None
        assert tracker.track_thresh == 0.5

    def test_track_dataclass(self):
        """测试Track数据类"""
        track = Track(
            track_id=1,
            bbox=np.array([100, 200, 150, 350], dtype=np.float32),
            confidence=0.9,
            class_id=0,
            class_name="person",
            frame_id=0,
        )
        assert track.track_id == 1
        assert track.confidence == 0.9

    def test_tracker_reset(self, tracker):
        """测试跟踪器重置"""
        tracker.reset()
        stats = tracker.get_statistics()
        assert stats["total_tracks_created"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
