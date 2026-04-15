# servo_calibrate.py
import time
from gpiozero import Servo
from config import Config

# 使用你 config.py 中的引脚定义
PAN_PIN = Config.SERVO_PAN_PIN
TILT_PIN = Config.SERVO_TILT_PIN

print("开始舵机校准程序...")

# 初始化舵机，使用标准的SG90脉冲范围
pan_servo = Servo(PAN_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
tilt_servo = Servo(TILT_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# 发送90度（中位）对应的信号，gpiozero中值为0
mid_value = 0

print(f"正在将舵机移动到中心位置 (90°)...")
pan_servo.value = mid_value
tilt_servo.value = mid_value

# 等待舵机物理转动到位
time.sleep(2)

print("校准完成！舵机应停在90°中心位置。")
print("现在可以安全地安装舵机臂或进行机械连接。")

# 保持舵机在中心位置，不要立即退出
# input("按 Enter 键退出程序并释放舵机...")

# 退出前detach舵机，释放引脚
pan_servo.detach()
tilt_servo.detach()
print("舵机已释放。")