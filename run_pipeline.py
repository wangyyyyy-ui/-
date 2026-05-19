"""
主流水线运行脚本

使用方法:
    python scripts/run_pipeline.py --source data/videos/classroom.mp4
    python scripts/run_pipeline.py --source 0  # 摄像头
    python scripts/run_pipeline.py --config configs/default.yaml
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ClassroomPipeline, PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="课堂多目标视觉感知与学习状态分析系统"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="视频源路径或摄像头索引"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径 (YAML)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov10s",
        help="检测模型名称"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="推理设备 (cpu/cuda/0)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="不显示实时画面"
    )
    parser.add_argument(
        "--no-save-video", action="store_true",
        help="不保存标注视频"
    )
    parser.add_argument(
        "--sample-interval", type=int, default=1,
        help="帧采样间隔"
    )
    parser.add_argument(
        "--face-db", type=str, default="data/face_db",
        help="人脸特征库路径"
    )
    return parser.parse_args()


def load_config_from_yaml(yaml_path: str) -> PipelineConfig:
    """从YAML文件加载配置"""
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    config = PipelineConfig()

    # 视频配置
    if "video" in cfg_dict:
        v = cfg_dict["video"]
        config.source = v.get("source", config.source)
        config.sample_interval = v.get("sample_interval", config.sample_interval)
        if v.get("target_size"):
            config.target_size = tuple(v["target_size"])

    # 检测配置
    if "detection" in cfg_dict:
        d = cfg_dict["detection"]
        config.detector_model = d.get("model", config.detector_model)
        config.det_conf_threshold = d.get("conf_threshold", config.det_conf_threshold)
        config.det_iou_threshold = d.get("iou_threshold", config.det_iou_threshold)
        config.det_device = d.get("device", config.det_device)

    # 跟踪配置
    if "tracking" in cfg_dict:
        t = cfg_dict["tracking"]
        config.track_thresh = t.get("track_thresh", config.track_thresh)
        config.track_buffer = t.get("track_buffer", config.track_buffer)
        config.match_thresh = t.get("match_thresh", config.match_thresh)

    # 人脸识别配置
    if "face_recognition" in cfg_dict:
        fr = cfg_dict["face_recognition"]
        config.face_model = fr.get("model", config.face_model)
        config.face_threshold = fr.get("threshold", config.face_threshold)
        config.face_db_path = fr.get("db_path", config.face_db_path)

    # 状态分析配置
    if "state_analysis" in cfg_dict:
        sa = cfg_dict["state_analysis"]
        config.head_down_pitch = sa.get("head_down_pitch", config.head_down_pitch)
        config.distracted_yaw = sa.get("distracted_yaw", config.distracted_yaw)

    # 输出配置
    if "output" in cfg_dict:
        o = cfg_dict["output"]
        config.output_dir = o.get("dir", config.output_dir)
        config.save_video = o.get("save_video", config.save_video)
        config.save_stats = o.get("save_stats", config.save_stats)
        config.display = o.get("display", config.display)

    return config


def main():
    args = parse_args()

    # 加载配置
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = PipelineConfig()

    # 命令行参数覆盖
    config.source = args.source
    config.detector_model = args.model
    config.det_device = args.device
    config.output_dir = args.output
    config.sample_interval = args.sample_interval
    config.face_db_path = args.face_db

    if args.no_display:
        config.display = False
    if args.no_save_video:
        config.save_video = False

    # 创建并运行流水线
    pipeline = ClassroomPipeline(config)
    summary = pipeline.run()

    print(f"\n运行摘要: {summary}")


if __name__ == "__main__":
    main()
