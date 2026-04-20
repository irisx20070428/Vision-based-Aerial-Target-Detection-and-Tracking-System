#!/usr/bin/env python3
# servo_debug.py - 独立舵机调试工具（水平CH1，垂直CH3）
import time
import sys
import board
import busio
from adafruit_pca9685 import PCA9685

def angle_to_duty(angle):
    """角度转占空比 (0°->0.5ms, 180°->2.5ms)"""
    pulse = 0.5 + (angle / 180.0) * 2.0   # ms
    return int(pulse / 20.0 * 65535)

def set_angle(pca, channel, angle):
    """设置指定通道的角度"""
    duty = angle_to_duty(angle)
    pca.channels[channel].duty_cycle = duty

def main():
    print("=" * 50)
    print("舵机调试工具 (PCA9685)")
    print("水平: CH1  垂直: CH3")
    print("=" * 50)

    # 初始化 I2C 和 PCA9685
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50
        print("✅ PCA9685 初始化成功")
    except Exception as e:
        print(f"❌ PCA9685 初始化失败: {e}")
        return

    # 通道配置
    pan_ch = 1    # 水平
    tilt_ch = 3   # 垂直

    # 当前角度
    pan_angle = 90
    tilt_angle = 90
    step = 1

    # 设置初始位置
    set_angle(pca, pan_ch, pan_angle)
    set_angle(pca, tilt_ch, tilt_angle)
    time.sleep(0.5)

    print("\n操作说明：")
    print("  w/s  : 水平舵机增加/减小角度")
    print("  i/k  : 垂直舵机增加/减小角度")
    print("  r    : 重置两个舵机到 90°")
    print("  +/-  : 增加/减小步进值")
    print("  q    : 退出")
    print(f"\n当前角度: 水平={pan_angle}°, 垂直={tilt_angle}°  步进={step}°")

    import tty, termios

    def get_key():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    try:
        while True:
            key = get_key()

            if key == 'w':
                pan_angle = min(180, pan_angle + step)
                set_angle(pca, pan_ch, pan_angle)
                print(f"\r水平: {pan_angle:3d}° | 垂直: {tilt_angle:3d}° | 步进: {step}°   ", end="")
            elif key == 's':
                pan_angle = max(0, pan_angle - step)
                set_angle(pca, pan_ch, pan_angle)
                print(f"\r水平: {pan_angle:3d}° | 垂直: {tilt_angle:3d}° | 步进: {step}°   ", end="")
            elif key == 'i':
                tilt_angle = min(180, tilt_angle + step)
                set_angle(pca, tilt_ch, tilt_angle)
                print(f"\r水平: {pan_angle:3d}° | 垂直: {tilt_angle:3d}° | 步进: {step}°   ", end="")
            elif key == 'k':
                tilt_angle = max(0, tilt_angle - step)
                set_angle(pca, tilt_ch, tilt_angle)
                print(f"\r水平: {pan_angle:3d}° | 垂直: {tilt_angle:3d}° | 步进: {step}°   ", end="")
            elif key == 'r':
                pan_angle = 90
                tilt_angle = 90
                set_angle(pca, pan_ch, pan_angle)
                set_angle(pca, tilt_ch, tilt_angle)
                print(f"\r水平: {pan_angle:3d}° | 垂直: {tilt_angle:3d}° | 步进: {step}°   ", end="")
            elif key == '+':
                step = min(10, step + 1)
                print(f"\r水平: {pan_angle:3d}° | 垂直: {tilt_angle:3d}° | 步进: {step}°   ", end="")
            elif key == '-':
                step = max(1, step - 1)
                print(f"\r水平: {pan_angle:3d}° | 垂直: {tilt_angle:3d}° | 步进: {step}°   ", end="")
            elif key == 'q':
                print("\n\n退出调试")
                break
    finally:
        # 可选：退出前释放舵机（停止PWM信号）
        pca.channels[pan_ch].duty_cycle = 0
        pca.channels[tilt_ch].duty_cycle = 0
        pca.deinit()
        print("✅ 舵机已释放")

if __name__ == "__main__":
    main()