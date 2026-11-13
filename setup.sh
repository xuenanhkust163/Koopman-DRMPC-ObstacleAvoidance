#!/bin/bash
# Experiment project quick setup script
# Usage: ./setup.sh

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo " K-DRMPC Experiment 项目设置"
echo "=================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    echo "请先安装Python 3.10或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ 找到Python: $PYTHON_VERSION"

# Check if in the correct directory
if [ ! -f "src/config.py" ]; then
    echo "❌ 错误: 未找到项目文件 src/config.py"
    exit 1
fi

echo "✓ 项目目录: $PROJECT_ROOT"
echo ""

# Select environment type
echo "请选择虚拟环境类型:"
echo "1) conda (推荐)"
echo "2) venv"
read -p "输入选项 (1/2): " choice

if [ "$choice" = "1" ]; then
    # Using conda
    if ! command -v conda &> /dev/null; then
        echo "❌ 错误: 未找到conda"
        echo "请先安装Anaconda或Miniconda"
        exit 1
    fi
    
    echo ""
    echo "使用conda创建环境..."
    conda create -n koopman_exp python=3.11 -y
    echo ""
    echo "激活环境..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate koopman_exp
    
    echo ""
    echo "安装依赖..."
    pip install -r requirements.txt
    
    ENV_NAME="koopman_exp (conda)"
    
elif [ "$choice" = "2" ]; then
    # Using venv
    echo ""
    echo "创建venv虚拟环境..."
    python3 -m venv venv
    
    echo ""
    echo "激活环境..."
    source venv/bin/activate
    
    echo ""
    echo "安装依赖..."
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -r requirements.txt
    
    ENV_NAME="venv"
    
else
    echo "❌ 无效选项"
    exit 1
fi

echo ""
echo "=================================================="
echo " 验证安装"
echo "=================================================="

# Verify core libraries
python3 -c "
import torch
import numpy
import casadi
import scipy
import matplotlib

print('✓ PyTorch:', torch.__version__)
print('✓ NumPy:', numpy.__version__)
print('✓ CasADi:', casadi.__version__)
print('✓ SciPy:', scipy.__version__)
print('✓ Matplotlib:', matplotlib.__version__)
print()
print('✓ 所有核心库安装成功!')
" || {
    echo "❌ 安装验证失败"
    exit 1
}

echo ""
echo "=================================================="
echo " 设置完成!"
echo "=================================================="
echo ""
echo "环境: $ENV_NAME"
echo ""
echo "下一步:"
echo "  激活环境: source venv/bin/activate  (或 conda activate koopman_exp)"
echo "  运行项目: 见 README.md说明"
echo ""
echo "文档:"
echo "  README.md  - 项目说明"
echo ""
