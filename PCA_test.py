#!/usr/bin/env python3
import time
import board
import busio
from adafruit_pca9685 import PCA9685

# 角度转占空比 (0°->0.5ms, 180°->2.5ms)
def angle_to_duty(angle):
    pulse = 0.5 + (angle / 180.0) * 2.0   # ms
    return int(pulse / 20.0 * 65535)

# 初始化 I2C 和 PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

print("测试水平舵机 (CH1)")
for angle in [0, 45, 90, 135, 180, 135, 90, 45, 0]:
    print(f"  转到 {angle} 度")
    pca.channels[1].duty_cycle = angle_to_duty(angle)
    time.sleep(0.5)

print("测试垂直舵机 (CH3)")
for angle in [0, 45, 90, 135, 180, 135, 90, 45, 0]:
    print(f"  转到 {angle} 度")
    pca.channels[3].duty_cycle = angle_to_duty(angle)
    time.sleep(0.5)

# 归位
pca.channels[1].duty_cycle = angle_to_duty(90)
pca.channels[3].duty_cycle = angle_to_duty(90)
print("归位完成")

pca.deinit()