#!/usr/bin/env python3
"""
舵机控制器 - 树莓派5兼容版本
使用 gpiozero 和 pigpio 实现精确舵机控制
"""
import time
import threading
from config import Config

# 尝试导入 gpiozero，如果失败则提示安装
try:
    from gpiozero import Servo
    from gpiozero.pins.pigpio import PiGPIOPin
    from gpiozero import Device

    # 使用 pigpio 引脚工厂
    Device.pin_factory = PiGPIOPin()
    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False
    print("警告: gpiozero 未安装，请在树莓派上运行: pip install gpiozero")


class ServoController:
    """舵机控制器 - 控制云台 (树莓派5兼容版)"""

    def __init__(self):
        self.pan_pin = Config.SERVO_PAN_PIN
        self.tilt_pin = Config.SERVO_TILT_PIN

        # 舵机参数：脉冲宽度范围 (ms)
        self.min_pulse = 0.5  # 毫秒
        self.max_pulse = 2.5  # 毫秒
        self.frame_width = 20.0 / 1000  # 20ms周期 = 50Hz

        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE
        self.target_pan = Config.SERVO_CENTER_ANGLE
        self.target_tilt = Config.SERVO_CENTER_ANGLE

        self.pan_servo = None
        self.tilt_servo = None

        self.move_lock = threading.Lock()

        self._init_gpio()

    def _angle_to_value(self, angle):
        """将角度（0-180）转换为 gpiozero 的 value（-1到1）"""
        angle = max(0, min(180, angle))
        value = (angle / 90.0) - 1
        return value

    def _value_to_angle(self, value):
        """将 gpiozero 的 value 转换为角度"""
        angle = (value + 1) * 90
        return max(0, min(180, angle))

    def _init_gpio(self):
        """初始化GPIO和舵机"""
        if not GPIOZERO_AVAILABLE:
            print("❌ gpiozero 不可用，请安装: pip install gpiozero")
            raise ImportError("gpiozero not installed")

        try:
            # 创建舵机对象
            self.pan_servo = Servo(
                self.pan_pin,
                min_pulse_width=self.min_pulse / 1000,
                max_pulse_width=self.max_pulse / 1000,
                frame_width=self.frame_width
            )

            self.tilt_servo = Servo(
                self.tilt_pin,
                min_pulse_width=self.min_pulse / 1000,
                max_pulse_width=self.max_pulse / 1000,
                frame_width=self.frame_width
            )

            # 移动到中心位置
            self.pan_servo.mid()
            self.tilt_servo.mid()
            self.current_pan = Config.SERVO_CENTER_ANGLE
            self.current_tilt = Config.SERVO_CENTER_ANGLE
            self.target_pan = Config.SERVO_CENTER_ANGLE
            self.target_tilt = Config.SERVO_CENTER_ANGLE

            print(f"✅ 舵机控制器初始化成功 (gpiozero + pigpio)")
            print(f"   - 水平舵机: GPIO{self.pan_pin}")
            print(f"   - 垂直舵机: GPIO{self.tilt_pin}")

        except Exception as e:
            print(f"❌ 舵机初始化失败: {e}")
            print("   请确保 pigpiod 服务正在运行:")
            print("   sudo systemctl start pigpiod")
            raise

    def _set_angle_immediate(self, pan_angle, tilt_angle):
        """立即设置角度"""
        try:
            pan_value = self._angle_to_value(pan_angle)
            tilt_value = self._angle_to_value(tilt_angle)

            self.pan_servo.value = pan_value
            self.tilt_servo.value = tilt_value

            self.current_pan = pan_angle
            self.current_tilt = tilt_angle
        except Exception as e:
            print(f"设置角度失败: {e}")

    def set_target(self, pan_angle, tilt_angle):
        """设置目标角度"""
        pan_angle = max(Config.SERVO_ANGLE_MIN, min(Config.SERVO_ANGLE_MAX, pan_angle))
        tilt_angle = max(Config.SERVO_ANGLE_MIN, min(Config.SERVO_ANGLE_MAX, tilt_angle))

        self.target_pan = pan_angle
        self.target_tilt = tilt_angle

    def update(self, dt=0.02):
        """更新舵机位置（带加速度和速度限制）"""
        with self.move_lock:
            pan_diff = self.target_pan - self.current_pan
            tilt_diff = self.target_tilt - self.current_tilt

            max_move = Config.MAX_ANGLE_SPEED * dt

            pan_move = max(-max_move, min(max_move, pan_diff))
            tilt_move = max(-max_move, min(max_move, tilt_diff))

            if abs(pan_diff) > max_move * 2:
                pan_move *= (1 + Config.ACCELERATION)
            elif abs(pan_diff) < max_move * 0.5:
                pan_move *= (1 - Config.ACCELERATION * 0.5)

            pan_move = max(-max_move, min(max_move, pan_move))
            tilt_move = max(-max_move, min(max_move, tilt_move))

            new_pan = self.current_pan + pan_move
            new_tilt = self.current_tilt + tilt_move

            self._set_angle_immediate(new_pan, new_tilt)

    def get_current_angles(self):
        """获取当前角度"""
        return self.current_pan, self.current_tilt

    def reset_to_center(self):
        """重置到中心"""
        self.set_target(Config.SERVO_CENTER_ANGLE, Config.SERVO_CENTER_ANGLE)
        print("🔄 云台重置到中心位置")

    def cleanup(self):
        """清理资源"""
        try:
            if self.pan_servo:
                self.pan_servo.detach()
            if self.tilt_servo:
                self.tilt_servo.detach()
            print("✅ 舵机已关闭")
        except:
            pass