"""
流水线集成测试
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import PipelineConfig, ClassroomPipeline
from src.state_analysis.state_classifier import StateClassifier, StudentState
from src.state_analysis.temporal_fusion import TemporalFusion
from src.state_analysis.state_classifier import StateResult
from src.visualization.drawer import VisualizationDrawer


class TestPipelineConfig:
    """流水线配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = PipelineConfig()
        assert config.detector_model == "yolov10s"
        assert config.sample_interval == 1
        assert config.det_conf_threshold == 0.25

    def test_custom_config(self):
        """测试自定义配置"""
        config = PipelineConfig(
            source="test.mp4",
            detector_model="yolov10m",
            det_device="cpu",
        )
        assert config.source == "test.mp4"
        assert config.detector_model == "yolov10m"


class TestStateClassifier:
    """状态分类器测试"""

    @pytest.fixture
    def classifier(self):
        return StateClassifier()

    def test_classify_tracked_no_pose(self, classifier):
        """测试有跟踪无姿态的情况"""
        result = classifier.classify(
            track_id=1,
            is_tracked=True,
        )
        assert result.state == StudentState.UNKNOWN

    def test_classify_lost_track(self, classifier):
        """测试丢失跟踪的情况"""
        result = classifier.classify(
            track_id=1,
            is_tracked=False,
        )
        assert result.state == StudentState.LEFT_SEAT


class TestTemporalFusion:
    """时序融合测试"""

    @pytest.fixture
    def fusion(self):
        return TemporalFusion(fps=30.0, window_size=5)

    def test_fusion_creation(self, fusion):
        """测试融合器创建"""
        assert fusion is not None
        assert fusion.window_size == 5

    def test_fusion_update(self, fusion):
        """测试融合更新"""
        state_results = [
            StateResult(track_id=1, state=StudentState.FOCUSED, confidence=0.9),
            StateResult(track_id=2, state=StudentState.HEAD_DOWN, confidence=0.8),
        ]
        smoothed, stats = fusion.update(state_results, frame_id=0, timestamp=0.0)
        assert len(smoothed) == 2
        assert stats is not None

    def test_fusion_reset(self, fusion):
        """测试融合器重置"""
        fusion.reset()
        history = fusion.get_statistics_history()
        assert len(history) == 0


class TestVisualization:
    """可视化测试"""

    @pytest.fixture
    def drawer(self):
        return VisualizationDrawer()

    def test_drawer_creation(self, drawer):
        """测试绘制器创建"""
        assert drawer is not None

    def test_draw_empty_frame(self, drawer):
        """测试空帧绘制"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = drawer.draw_frame(frame, [], {}, None)
        assert result is not None
        assert result.shape == (720, 1280, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
