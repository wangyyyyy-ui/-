"""
目标检测模块单元测试
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.yolo_detector import YOLODetector, Detection, DetectionResult


class TestDetection:
    """目标检测测试类"""

    @pytest.fixture
    def detector(self):
        """创建检测器实例"""
        try:
            return YOLODetector(model_name="yolov10n", device="cpu")
        except Exception:
            pytest.skip("YOLO模型加载失败，可能未安装ultralytics")

    @pytest.fixture
    def sample_image(self):
        """创建测试图像"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_detector_creation(self, detector):
        """测试检测器创建"""
        assert detector is not None
        assert detector.model_name == "yolov10n"

    def test_detect_returns_result(self, detector, sample_image):
        """测试检测返回结果格式"""
        result = detector.detect(sample_image)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.detections, list)
        assert isinstance(result.inference_time, float)
        assert result.image_shape == (480, 640)

    def test_detection_fields(self, detector, sample_image):
        """测试检测结果字段"""
        result = detector.detect(sample_image)
        for det in result.detections:
            assert isinstance(det, Detection)
            assert det.bbox.shape == (4,)
            assert 0 <= det.confidence <= 1
            assert isinstance(det.class_id, int)
            assert isinstance(det.class_name, str)

    def test_model_info(self, detector):
        """测试模型信息获取"""
        info = detector.get_model_info()
        assert "model_name" in info
        assert "device" in info
        assert "conf_threshold" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
