# servo_controller.py - 添加 target 角度属性访问
import time
import threading
from gpiozero import Servo
from config import Config


class ServoController:
    """舵机控制器 - 使用 gpiozero - 防抽搐优化版"""

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

        # 速度限制
        self.max_angle_change = 1.0
        self.last_update_time = time.time()

        # 追踪使能标志
        self.tracking_enabled = False
        self.initialized = False

        self._init_servos()

    def _init_servos(self):
        """初始化舵机并回正"""
        print("初始化舵机...")

        try:
            self.pan_servo = Servo(
                self.pan_pin,
                min_pulse_width=0.6 / 1000,
                max_pulse_width=2.4 / 1000
            )

            self.tilt_servo = Servo(
                self.tilt_pin,
                min_pulse_width=0.6 / 1000,
                max_pulse_width=2.4 / 1000
            )

            # 先禁用舵机输出
            self.pan_servo.detach()
            self.tilt_servo.detach()
            time.sleep(0.1)

            # 设置到中心位置
            pan_value = self._angle_to_value(Config.SERVO_CENTER_ANGLE)
            tilt_value = self._angle_to_value(Config.SERVO_CENTER_ANGLE)

            self.pan_servo.value = pan_value
            self.tilt_servo.value = tilt_value

            time.sleep(0.5)

            self.current_pan = Config.SERVO_CENTER_ANGLE
            self.current_tilt = Config.SERVO_CENTER_ANGLE
            self.target_pan = Config.SERVO_CENTER_ANGLE
            self.target_tilt = Config.SERVO_CENTER_ANGLE

            self.initialized = True

            print(f"✅ 舵机控制器初始化成功")
            print(f"   - 水平舵机: GPIO{self.pan_pin}")
            print(f"   - 垂直舵机: GPIO{self.tilt_pin}")
            print(f"   - 最大转动速度: {Config.MAX_ANGLE_SPEED}°/秒")
            print(f"   - 初始位置: 中心 ({Config.SERVO_CENTER_ANGLE}°)")

        except Exception as e:
            print(f"❌ 舵机初始化失败: {e}")
            raise e

    def enable_tracking(self, enabled):
        """启用/禁用舵机追踪"""
        with self.move_lock:
            self.tracking_enabled = enabled
            if not enabled:
                self.target_pan = self.current_pan
                self.target_tilt = self.current_tilt
            print(f"🎯 舵机追踪: {'已启用' if enabled else '已禁用'}")

    def set_target(self, pan_angle, tilt_angle):
        """设置目标角度"""
        if not self.tracking_enabled:
            return

        pan_angle = max(Config.SERVO_ANGLE_MIN, min(Config.SERVO_ANGLE_MAX, pan_angle))
        tilt_angle = max(Config.SERVO_ANGLE_MIN, min(Config.SERVO_ANGLE_MAX, tilt_angle))

        with self.move_lock:
            self.target_pan = pan_angle
            self.target_tilt = tilt_angle

    def update(self, dt=None):
        """更新舵机位置"""
        if not self.initialized or not self.tracking_enabled:
            return

        # 计算时间间隔
        current_time = time.time()
        if dt is None:
            dt = min(0.1, current_time - self.last_update_time)
            dt = max(0.05, dt)

        self.last_update_time = current_time

        with self.move_lock:
            # 计算角度差
            pan_diff = self.target_pan - self.current_pan
            tilt_diff = self.target_tilt - self.current_tilt

            # 死区处理
            deadband = 1.0
            if abs(pan_diff) < deadband:
                pan_diff = 0
            if abs(tilt_diff) < deadband:
                tilt_diff = 0

            if pan_diff == 0 and tilt_diff == 0:
                return

            # 计算最大移动步长
            max_speed = Config.MAX_ANGLE_SPEED
            max_step = max_speed * dt
            max_step = min(max_step, self.max_angle_change)

            # 移动
            pan_move = 0
            tilt_move = 0

            if pan_diff != 0:
                if abs(pan_diff) > 10:
                    speed_factor = 0.8
                elif abs(pan_diff) > 5:
                    speed_factor = 0.5
                else:
                    speed_factor = 0.3
                pan_move = pan_diff * speed_factor * dt
                pan_move = max(-max_step, min(max_step, pan_move))

            if tilt_diff != 0:
                if abs(tilt_diff) > 10:
                    speed_factor = 0.8
                elif abs(tilt_diff) > 5:
                    speed_factor = 0.5
                else:
                    speed_factor = 0.3
                tilt_move = tilt_diff * speed_factor * dt
                tilt_move = max(-max_step, min(max_step, tilt_move))

            if abs(pan_move) > 0.1 or abs(tilt_move) > 0.1:
                new_pan = self.current_pan + pan_move
                new_tilt = self.current_tilt + tilt_move
                self._set_angle_immediate(new_pan, new_tilt)

    def _angle_to_value(self, angle):
        """将角度转换为 gpiozero 的 value (-1 到 1)"""
        value = (angle - 90) / 90
        return max(-1, min(1, value))

    def _set_angle_immediate(self, pan_angle, tilt_angle):
        """立即设置角度"""
        pan_value = self._angle_to_value(pan_angle)
        tilt_value = self._angle_to_value(tilt_angle)

        self.pan_servo.value = pan_value
        self.tilt_servo.value = tilt_value

        self.current_pan = pan_angle
        self.current_tilt = tilt_angle

    def get_current_angles(self):
        """获取当前角度"""
        return self.current_pan, self.current_tilt

    def reset_to_center(self):
        """重置到中心"""
        if self.tracking_enabled:
            self.set_target(Config.SERVO_CENTER_ANGLE, Config.SERVO_CENTER_ANGLE)
            print("🔄 云台重置到中心位置")

    def cleanup(self):
        """清理资源"""
        print("🔄 正在关闭舵机...")
        if self.pan_servo:
            self.pan_servo.detach()
        if self.tilt_servo:
            self.tilt_servo.detach()
        print("✅ 舵机已关闭")