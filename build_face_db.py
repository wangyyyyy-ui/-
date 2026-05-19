"""
人脸特征库构建工具

从学生登记照片目录批量构建人脸特征库

使用方法:
    python tools/build_face_db.py --image_dir data/face_db/photos
    python tools/build_face_db.py --image_dir data/face_db/photos --clear

目录结构要求:
    image_dir/
    ├── student_001/
    │   ├── photo1.jpg
    │   └── photo2.jpg
    ├── student_002/
    │   └── photo1.jpg
    └── ...
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_recognition.arcface import ArcFaceRecognizer
from src.face_recognition.face_db import FaceDatabase


def main():
    parser = argparse.ArgumentParser(description="构建人脸特征库")
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="学生照片目录路径"
    )
    parser.add_argument(
        "--db_path", type=str, default="data/face_db",
        help="特征库保存路径"
    )
    parser.add_argument(
        "--model", type=str, default="buffalo_l",
        help="InsightFace模型名称"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="匹配阈值"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="清空已有特征库后重建"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  人脸特征库构建工具")
    print("=" * 60)

    # 初始化识别器
    print(f"\n初始化ArcFace识别器 (模型: {args.model})...")
    recognizer = ArcFaceRecognizer(
        model_name=args.model,
        threshold=args.threshold,
    )

    # 初始化特征库
    print(f"特征库路径: {args.db_path}")
    face_db = FaceDatabase(db_path=args.db_path, recognizer=recognizer)

    if args.clear:
        print("清空已有特征库...")
        face_db.clear()

    # 从目录构建
    print(f"\n从目录构建特征库: {args.image_dir}")
    result = face_db.build_from_directory(args.image_dir)

    print(f"\n构建结果:")
    print(f"  成功: {result.get('success', 0)}人")
    print(f"  失败: {result.get('failed', 0)}人")

    # 显示特征库统计
    stats = face_db.get_statistics()
    print(f"\n特征库统计:")
    print(f"  总人数: {stats['total_students']}")
    print(f"  学生列表: {stats['students']}")

    print("\n特征库构建完成!")


if __name__ == "__main__":
    main()
