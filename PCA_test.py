import time
import board
import busio
from adafruit_servokit import ServoKit

i2c = busio.I2C(board.SCL, board.SDA)
kit = ServoKit(i2c=i2c, channels=16)
kit.frequency = 50

print("测试水平舵机 (CH1)...")
for angle in [0, 45, 90, 135, 180, 135, 90, 45, 0]:
    print(f"  CH1: {angle}°")
    kit.servo[1].angle = angle
    time.sleep(0.5)

print("测试垂直舵机 (CH3)...")
for angle in [0, 45, 90, 135, 180, 135, 90, 45, 0]:
    print(f"  CH3: {angle}°")
    kit.servo[3].angle = angle
    time.sleep(0.5)

# 归位
kit.servo[1].angle = 90
kit.servo[3].angle = 90
print("舵机已归位到 90 度")

kit.servo[1].angle = None
kit.servo[3].angle = None
print("测试完成")