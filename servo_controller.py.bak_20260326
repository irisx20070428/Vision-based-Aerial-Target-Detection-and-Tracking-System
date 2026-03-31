# servo_controller.py
import RPi.GPIO as GPIO
import time
import threading
import math
from config import Config


class ServoController:
    """舵机控制器 - 控制云台"""

    def __init__(self):
        self.pan_pin = Config.SERVO_PAN_PIN
        self.tilt_pin = Config.SERVO_TILT_PIN
        self.frequency = Config.SERVO_FREQUENCY

        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE
        self.target_pan = Config.SERVO_CENTER_ANGLE
        self.target_tilt = Config.SERVO_CENTER_ANGLE

        self.pan_pwm = None
        self.tilt_pwm = None

        self.moving = False
        self.move_lock = threading.Lock()

        self._init_gpio()

    def _init_gpio(self):
        """初始化GPIO"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # 设置引脚
        GPIO.setup(self.pan_pin, GPIO.OUT)
        GPIO.setup(self.tilt_pin, GPIO.OUT)

        # 创建PWM
        self.pan_pwm = GPIO.PWM(self.pan_pin, self.frequency)
        self.tilt_pwm = GPIO.PWM(self.tilt_pin, self.frequency)

        # 启动PWM
        self.pan_pwm.start(0)
        self.tilt_pwm.start(0)

        # 移动到中心
        self._set_angle_immediate(self.current_pan, self.current_tilt)

        print(f"✅ 舵机控制器初始化成功")
        print(f"   - 水平舵机: GPIO{self.pan_pin}")
        print(f"   - 垂直舵机: GPIO{self.tilt_pin}")

    def _angle_to_duty(self, angle):
        """角度转占空比"""
        # 公式: duty = angle / 18 + 2.5
        duty = angle / 18 + 2.5
        return max(2.5, min(12.5, duty))

    def _set_angle_immediate(self, pan_angle, tilt_angle):
        """立即设置角度"""
        pan_duty = self._angle_to_duty(pan_angle)
        tilt_duty = self._angle_to_duty(tilt_angle)

        self.pan_pwm.ChangeDutyCycle(pan_duty)
        self.tilt_pwm.ChangeDutyCycle(tilt_duty)

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
        更新舵机位置（带加速度和速度限制）

        参数:
            dt: 时间间隔（秒）
        """
        with self.move_lock:
            # 计算角度差
            pan_diff = self.target_pan - self.current_pan
            tilt_diff = self.target_tilt - self.current_tilt

            # 计算移动步长（带加速度）
            max_move = Config.MAX_ANGLE_SPEED * dt

            # 限制移动速度
            pan_move = max(-max_move, min(max_move, pan_diff))
            tilt_move = max(-max_move, min(max_move, tilt_diff))

            # 应用加速度（平滑加速减速）
            if abs(pan_diff) > max_move * 2:
                pan_move *= (1 + Config.ACCELERATION)
            elif abs(pan_diff) < max_move * 0.5:
                pan_move *= (1 - Config.ACCELERATION * 0.5)

            pan_move = max(-max_move, min(max_move, pan_move))
            tilt_move = max(-max_move, min(max_move, tilt_move))

            # 更新角度
            new_pan = self.current_pan + pan_move
            new_tilt = self.current_tilt + tilt_move

            # 设置角度
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
        if self.pan_pwm:
            self.pan_pwm.stop()
        if self.tilt_pwm:
            self.tilt_pwm.stop()
        GPIO.cleanup()
        print("✅ 舵机已关闭")