# servo_controller.py - 使用 gpiozero 版本
import time
import threading
from gpiozero import Servo
from config import Config


class ServoController:
    """舵机控制器 - 使用 gpiozero"""

    def __init__(self):
        self.pan_pin = Config.SERVO_PAN_PIN
        self.tilt_pin = Config.SERVO_TILT_PIN

        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE
        self.target_pan = Config.SERVO_CENTER_ANGLE
        self.target_tilt = Config.SERVO_CENTER_ANGLE

        self.move_lock = threading.Lock()

        self.pan_servo = None
        self.tilt_servo = None

        self._init_servos()

    def _init_servos(self):
        """初始化舵机"""
        print("初始化舵机...")

        try:
            # 初始化水平舵机 (GPIO 18)
            # 参数说明:
            # min_pulse_width: 最小脉冲宽度 (0度)
            # max_pulse_width: 最大脉冲宽度 (180度)
            # 对于 SG90 舵机: 0.5ms = 0度, 2.5ms = 180度
            self.pan_servo = Servo(
                self.pan_pin,
                min_pulse_width=0.5 / 1000,  # 0.5ms
                max_pulse_width=2.5 / 1000  # 2.5ms
            )

            # 初始化垂直舵机 (GPIO 27)
            self.tilt_servo = Servo(
                self.tilt_pin,
                min_pulse_width=0.5 / 1000,
                max_pulse_width=2.5 / 1000
            )

            # 移动到中心位置
            self._set_angle_immediate(self.current_pan, self.current_tilt)

            print(f"✅ 舵机控制器初始化成功")
            print(f"   - 水平舵机: GPIO{self.pan_pin}")
            print(f"   - 垂直舵机: GPIO{self.tilt_pin}")
            print(f"   - 脉冲范围: 0.5ms - 2.5ms")

        except Exception as e:
            print(f"❌ 舵机初始化失败: {e}")
            raise e

    def _angle_to_value(self, angle):
        """
        将角度转换为 gpiozero 的 value (-1 到 1)

        参数:
            angle: 0-180 度

        返回:
            value: -1 到 1
            -1 = 0度
            0 = 90度
            1 = 180度
        """
        # 将角度映射到 -1 到 1
        value = (angle - 90) / 90
        return max(-1, min(1, value))

    def _value_to_angle(self, value):
        """
        将 gpiozero 的 value 转换为角度

        参数:
            value: -1 到 1

        返回:
            angle: 0-180 度
        """
        angle = (value * 90) + 90
        return max(0, min(180, angle))

    def _set_angle_immediate(self, pan_angle, tilt_angle):
        """立即设置角度"""
        pan_value = self._angle_to_value(pan_angle)
        tilt_value = self._angle_to_value(tilt_angle)

        self.pan_servo.value = pan_value
        self.tilt_servo.value = tilt_value

        self.current_pan = pan_angle
        self.current_tilt = tilt_angle

    def set_target(self, pan_angle, tilt_angle):
        """设置目标角度"""
        # 限制角度范围
        pan_angle = max(Config.SERVO_ANGLE_MIN, min(Config.SERVO_ANGLE_MAX, pan_angle))
        tilt_angle = max(Config.SERVO_ANGLE_MIN, min(Config.SERVO_ANGLE_MAX, tilt_angle))

        self.target_pan = pan_angle
        self.target_tilt = tilt_angle

    def update(self, dt=0.02):
        """
        更新舵机位置（平滑移动）
        """
        with self.move_lock:
            # 计算角度差
            pan_diff = self.target_pan - self.current_pan
            tilt_diff = self.target_tilt - self.current_tilt

            # === 1. 添加死区 ===
            deadband = 0.3  # 死区范围（度）
            if abs(pan_diff) < deadband:
                pan_diff = 0
            if abs(tilt_diff) < deadband:
                tilt_diff = 0

            # === 2. 计算最大移动步长 ===
            max_move = Config.MAX_ANGLE_SPEED * dt

            # === 3. 修复加速度逻辑 ===
            # 直接限制移动步长，不使用错误的加速度
            pan_move = max(-max_move, min(max_move, pan_diff))
            tilt_move = max(-max_move, min(max_move, tilt_diff))

            # 可选：简单的缓动效果（加速/减速）
            # 距离目标较远时加速，较近时减速
            if abs(pan_diff) > 10:  # 大于10度，全速移动
                pan_move = max(-max_move, min(max_move, pan_diff))
            elif abs(pan_diff) > 2:  # 2-10度，线性减速
                speed_factor = abs(pan_diff) / 10
                pan_move = pan_diff * speed_factor
                pan_move = max(-max_move, min(max_move, pan_move))
            else:  # 小于2度，不移动（由死区处理）
                pan_move = 0

            if abs(tilt_diff) > 10:
                tilt_move = max(-max_move, min(max_move, tilt_diff))
            elif abs(tilt_diff) > 2:
                speed_factor = abs(tilt_diff) / 10
                tilt_move = tilt_diff * speed_factor
                tilt_move = max(-max_move, min(max_move, tilt_move))
            else:
                tilt_move = 0

            # 更新角度
            if pan_move != 0 or tilt_move != 0:
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
        # 停止舵机
        if self.pan_servo:
            self.pan_servo.detach()
        if self.tilt_servo:
            self.tilt_servo.detach()
        print("✅ 舵机已关闭")