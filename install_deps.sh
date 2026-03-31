#!/bin/bash
echo "========================================="
echo "安装树莓派5追踪系统依赖"
echo "========================================="

# 激活虚拟环境
source yolo_env/bin/activate

# 安装 Python 包
echo "📦 安装 Python 包..."
pip install gpiozero pigpio

# 安装系统包
echo "📦 安装系统包..."
sudo apt update
sudo apt install -y pigpio python3-picamera2

# 启动 pigpio 服务
echo "🚀 启动 pigpio 服务..."
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

# 检查 pigpio 状态
sudo systemctl status pigpiod --no-pager

echo ""
echo "✅ 依赖安装完成！"
