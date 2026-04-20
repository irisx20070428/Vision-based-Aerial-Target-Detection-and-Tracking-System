#!/usr/bin/env python3
# servo_calibrate_pan.py - 仅校准水平舵机中点 (PCA9685 CH1)

import time
import sys
import board
import busio
from adafruit_pca9685 import PCA9685

def angle_to_duty(angle):
    """角度转占空比 (0-180度 -> 0.5-2.5ms脉冲)"""
    pulse = 0.5 + (angle / 180.0) * 2.0
    return int(pulse / 20.0 * 65535)

def save_pan_offset(offset):
    """将水平偏移量写入 config.py"""
    config_path = "config.py"
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        found = False
        for line in lines:
            if line.strip().startswith('SERVO_PAN_OFFSET'):
                new_lines.append(f"SERVO_PAN_OFFSET = {offset}\n")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"SERVO_PAN_OFFSET = {offset}\n")
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        print(f"✅ 已保存 SERVO_PAN_OFFSET = {offset} 到 config.py")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def main():
    print("=" * 50)
    print("水平舵机中点校准 (PCA9685 CH1)")
    print("=" * 50)
    print("说明：将云台/舵机臂调整到您认为的\"正前方\"")
    print("使用键盘 w/s 微调角度，按 q 保存并退出\n")

    # 初始化 I2C 和 PCA9685
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50
        print("✅ PCA9685 初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    pan_ch = 1          # 水平舵机通道 (CH1)
    angle = 90          # 起始角度 90°
    offset = 0
    step = 1

    # 设置初始角度
    pca.channels[pan_ch].duty_cycle = angle_to_duty(angle)
    time.sleep(0.5)

    print(f"当前角度: {angle}° (偏移: {offset:+d}°)  步进: {step}°")
    print("操作: w/s 增加/减小角度 | r 重置90° | +/- 步进 | q 保存退出 | x 不保存退出")

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
            print(f"\r角度: {angle:3d}° (偏移 {offset:+3d}°) | 步进: {step}°   ", end="")
            key = get_key()

            if key == 'w':
                angle = min(180, angle + step)
                offset = angle - 90
                pca.channels[pan_ch].duty_cycle = angle_to_duty(angle)
            elif key == 's':
                angle = max(0, angle - step)
                offset = angle - 90
                pca.channels[pan_ch].duty_cycle = angle_to_duty(angle)
            elif key == 'r':
                angle = 90
                offset = 0
                pca.channels[pan_ch].duty_cycle = angle_to_duty(angle)
                print("\n🔄 重置到 90°")
            elif key == '+':
                step = min(10, step + 1)
                print(f"\n步进值: {step}°", end="")
            elif key == '-':
                step = max(1, step - 1)
                print(f"\n步进值: {step}°", end="")
            elif key == 'q':
                print(f"\n\n✅ 保存偏移量 {offset} 到 config.py")
                save_pan_offset(offset)
                break
            elif key == 'x':
                print("\n\n❌ 未保存，退出")
                break
    finally:
        pca.deinit()
        print("舵机已释放")

if __name__ == "__main__":
    main()