"""
课堂状态分析评估工具

评估指标:
    - Overall Accuracy (总体分类准确率)
    - Per-class Precision/Recall/F1 (各类别指标)
    - Cohen's Kappa (一致性系数)
    - Confusion Matrix (混淆矩阵)

使用方法:
    python tools/eval_state.py --gt data/annotations/gt_states.json --pred outputs/pred_states.json
"""

import argparse
import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE_NAMES = ["focused", "distracted", "head_down", "left_seat", "hand_raising"]


def compute_confusion_matrix(
    y_true: list, y_pred: list, num_classes: int = 5
) -> np.ndarray:
    """计算混淆矩阵"""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def compute_per_class_metrics(cm: np.ndarray) -> dict:
    """
    从混淆矩阵计算各类别指标
    
    Args:
        cm: 混淆矩阵
        
    Returns:
        各类别Precision/Recall/F1
    """
    metrics = {}
    for i, name in enumerate(STATE_NAMES):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }

    return metrics


def compute_cohens_kappa(cm: np.ndarray) -> float:
    """
    计算Cohen's Kappa系数
    
    Kappa衡量分类结果与随机分类的一致性程度:
        < 0: 无一致性
        0-0.2: 轻微一致
        0.2-0.4: 一般一致
        0.4-0.6: 中等一致
        0.6-0.8: 高度一致
        0.8-1.0: 几乎完全一致
    """
    n = cm.sum()
    if n == 0:
        return 0.0

    po = np.trace(cm) / n  # 观察一致率
    pe = sum(cm[i, :].sum() * cm[:, i].sum() for i in range(len(cm))) / (n * n)  # 期望一致率

    if pe == 1.0:
        return 0.0

    return float((po - pe) / (1 - pe))


def main():
    parser = argparse.ArgumentParser(description="课堂状态分析评估")
    parser.add_argument("--gt", type=str, required=True, help="真值标注文件")
    parser.add_argument("--pred", type=str, required=True, help="预测结果文件")
    parser.add_argument("--output", type=str, default="outputs/eval_state.json")

    args = parser.parse_args()

    print("课堂状态分析评估")

    # 加载数据
    with open(args.gt, "r") as f:
        gt_data = json.load(f)
    with open(args.pred, "r") as f:
        pred_data = json.load(f)

    # 提取标签
    y_true = [STATE_NAMES.index(d["state"]) for d in gt_data if d["state"] in STATE_NAMES]
    y_pred = [STATE_NAMES.index(d["state"]) for d in pred_data if d["state"] in STATE_NAMES]

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # 计算指标
    cm = compute_confusion_matrix(y_true, y_pred)
    per_class = compute_per_class_metrics(cm)
    kappa = compute_cohens_kappa(cm)
    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

    results = {
        "overall_accuracy": float(accuracy),
        "cohens_kappa": kappa,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "num_samples": min_len,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n总体准确率: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"\n各类别指标:")
    for name, m in per_class.items():
        print(f"  {name}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")
    print(f"\n结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
