"""
多目标跟踪评估工具

评估指标:
    - MOTA (Multi-Object Tracking Accuracy)
    - IDF1 (ID F1 Score)
    - ID Switch (身份切换次数)
    - MT/ML (Mostly Tracked/Lost)
    - FP/FN (误检/漏检)

使用方法:
    python tools/eval_tracking.py --gt data/annotations/gt_mot.txt --result outputs/tracking_result.txt
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="多目标跟踪评估")
    parser.add_argument("--gt", type=str, required=True, help="真值标注文件(MOT格式)")
    parser.add_argument("--result", type=str, required=True, help="跟踪结果文件")
    parser.add_argument("--output", type=str, default="outputs/eval_tracking.json")

    args = parser.parse_args()

    print("多目标跟踪评估")

    try:
        import motmetrics as mm

        gt = mm.io.loadtxt(args.gt, fmt="mot15-2D")
        ts = mm.io.loadtxt(args.result, fmt="mot15-2D")

        mh = mm.metrics.create()
        acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)

        metrics = mh.compute(acc, metrics=[
            "mota", "idf1", "num_switches", "mostly_tracked",
            "mostly_lost", "num_false_positives", "num_misses"
        ])

        results = {
            "MOTA": float(metrics["mota"]),
            "IDF1": float(metrics["idf1"]),
            "ID_Switch": int(metrics["num_switches"]),
            "MT": int(metrics["mostly_tracked"]),
            "ML": int(metrics["mostly_lost"]),
            "FP": int(metrics["num_false_positives"]),
            "FN": int(metrics["num_misses"]),
        }

    except ImportError:
        print("请安装motmetrics: pip install motmetrics")
        results = {"error": "motmetrics not installed"}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果: {results}")


if __name__ == "__main__":
    main()
