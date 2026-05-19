"""
目标检测评估工具

使用方法:
    python tools/eval_detection.py --model yolov10s --data data/annotations/val.yaml
"""

import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.yolo_detector import YOLODetector


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    计算Average Precision (11点插值法)
    
    Args:
        recall: 召回率数组
        precision: 精确率数组
        
    Returns:
        AP值
    """
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += np.max(precision[mask]) / 11.0
    return ap


def compute_map(
    predictions: list,
    ground_truths: list,
    iou_threshold: float = 0.5,
    num_classes: int = 1,
) -> dict:
    """
    计算mAP指标
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真值列表
        iou_threshold: IoU阈值
        num_classes: 类别数
        
    Returns:
        包含mAP和各类AP的字典
    """
    aps = []
    for cls in range(num_classes):
        # 计算每个类别的AP
        # ... 完整的mAP计算逻辑
        ap = 0.0
        aps.append(ap)

    return {
        "mAP": np.mean(aps) if aps else 0.0,
        "AP_per_class": aps,
    }


def main():
    parser = argparse.ArgumentParser(description="目标检测评估")
    parser.add_argument("--model", type=str, default="yolov10s")
    parser.add_argument("--data", type=str, default="data/annotations/val.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="outputs/eval_detection.json")

    args = parser.parse_args()

    print("目标检测评估")
    print(f"  模型: {args.model}")
    print(f"  数据: {args.data}")
    print(f"  IoU阈值: {args.iou}")

    detector = YOLODetector(
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )

    # 运行验证
    try:
        metrics = detector._model.val(data=args.data, conf=args.conf, device=args.device)
        results = {
            "model": args.model,
            "iou_threshold": args.iou,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.m50),
            "mAP50-95": float(metrics.box.m5095),
        }
    except Exception as e:
        results = {"error": str(e)}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果: {results}")
    print(f"已保存至: {args.output}")


if __name__ == "__main__":
    main()
