#!/bin/bash
# ============================================================
# 课堂多目标视觉感知与学习状态分析系统 - 环境配置脚本
# ============================================================
# 使用方法:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e

echo "============================================================"
echo "  课堂多目标视觉感知与学习状态分析系统 - 环境配置"
echo "============================================================"

# 检查Python版本
echo ""
echo "[1/6] 检查Python环境..."
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | awk '{print $2}')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
            PYTHON_CMD=$cmd
            echo "  找到Python: $cmd ($version)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "  错误: 需要Python 3.8+，请先安装Python"
    exit 1
fi

# 检查CUDA
echo ""
echo "[2/6] 检查CUDA环境..."
if command -v nvidia-smi &> /dev/null; then
    echo "  检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "    - $line"
    done
    CUDA_AVAILABLE=1
else
    echo "  未检测到NVIDIA GPU，将使用CPU模式"
    CUDA_AVAILABLE=0
fi

# 创建虚拟环境（可选）
echo ""
echo "[3/6] 配置Python虚拟环境..."
if [ ! -d "venv" ]; then
    read -p "  是否创建Python虚拟环境? (y/n): " create_venv
    if [ "$create_venv" = "y" ]; then
        $PYTHON_CMD -m venv venv
        source venv/bin/activate
        echo "  虚拟环境已创建并激活"
        PYTHON_CMD=python
    fi
else
    echo "  虚拟环境已存在"
    source venv/bin/activate 2>/dev/null || true
    PYTHON_CMD=python
fi

# 安装PyTorch
echo ""
echo "[4/6] 安装PyTorch..."
if $PYTHON_CMD -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
    echo "  PyTorch已安装: $TORCH_VERSION"
else
    if [ "$CUDA_AVAILABLE" -eq 1 ]; then
        echo "  安装PyTorch (CUDA版本)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  安装PyTorch (CPU版本)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# 安装项目依赖
echo ""
echo "[5/6] 安装项目依赖..."
pip install -r requirements.txt

# 下载模型权重
echo ""
echo "[6/6] 下载预训练模型权重..."
mkdir -p weights

# YOLOv10权重（首次运行时自动下载）
echo "  YOLOv10权重将在首次运行时自动下载"

# InsightFace模型
echo "  InsightFace模型将在首次运行时自动下载"

# 创建必要目录
mkdir -p data/face_db
mkdir -p data/videos
mkdir -p data/annotations
mkdir -p outputs
mkdir -p logs

echo ""
echo "============================================================"
echo "  环境配置完成!"
echo "============================================================"
echo ""
echo "  快速开始:"
echo "    python scripts/run_pipeline.py --source data/videos/classroom.mp4"
echo "    python scripts/run_demo.py"
echo "    python scripts/run_evaluation.py"
echo ""
echo "  构建人脸特征库:"
echo "    python tools/build_face_db.py --image_dir data/face_db/photos"
echo ""
