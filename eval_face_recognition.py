"""
人脸识别评估工具

评估指标:
    - Accuracy (识别准确率)
    - FAR (False Accept Rate, 误签率)
    - FRR (False Reject Rate, 漏签率)
    - EER (Equal Error Rate, 等错误率)
    - Rank-1 (首位命中率)

使用方法:
    python tools/eval_face_recognition.py --db data/face_db --test data/face_db/test
"""

import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """
    计算Equal Error Rate (EER)
    
    EER是FAR=FRR时的错误率，是生物特征识别系统的
    标准评估指标。EER越低，系统性能越好。
    
    Args:
        genuine_scores: 真实匹配的相似度分数
        impostor_scores: 冒充匹配的相似度分数
        
    Returns:
        EER值
    """
    from sklearn.metrics import roc_curve

    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    return float(eer)


def main():
    parser = argparse.ArgumentParser(description="人脸识别评估")
    parser.add_argument("--db", type=str, default="data/face_db")
    parser.add_argument("--test", type=str, default="data/face_db/test")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--output", type=str, default="outputs/eval_face.json")

    args = parser.parse_args()

    print("人脸识别评估")

    from src.face_recognition.arcface import ArcFaceRecognizer
    from src.face_recognition.face_db import FaceDatabase

    recognizer = ArcFaceRecognizer(threshold=args.threshold)
    face_db = FaceDatabase(db_path=args.db, recognizer=recognizer)

    if len(face_db) == 0:
        print("特征库为空，请先构建特征库")
        return

    # 评估逻辑（需要测试集）
    results = {
        "accuracy": 0.0,
        "FAR": 0.0,
        "FRR": 0.0,
        "EER": 0.0,
        "rank1": 0.0,
        "db_size": len(face_db),
        "threshold": args.threshold,
        "note": "需要测试集进行完整评估，请准备测试数据后重新运行",
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果: {results}")


if __name__ == "__main__":
    main()
