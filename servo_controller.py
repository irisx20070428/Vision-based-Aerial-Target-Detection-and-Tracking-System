# servo_controller.py - 完整修复版
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
        self.max_angle_change = 1.0  # 降低到1度，更平滑
        self.last_update_time = time.time()

        # 追踪使能标志（默认禁用）
        self.tracking_enabled = False

        # 添加初始化完成标志
        self.initialized = False

        self._init_servos()

    def _init_servos(self):
        """初始化舵机并回正"""
        print("初始化舵机...")

        try:
            # 放宽脉冲范围，减少抖动
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

            # 重要：先禁用舵机输出（detach）
            self.pan_servo.detach()
            self.tilt_servo.detach()
            time.sleep(0.1)

            # 设置到中心位置
            pan_value = self._angle_to_value(Config.SERVO_CENTER_ANGLE)
            tilt_value = self._angle_to_value(Config.SERVO_CENTER_ANGLE)

            # 重新连接并设置位置
            self.pan_servo.value = pan_value
            self.tilt_servo.value = tilt_value

            # 等待舵机稳定
            time.sleep(0.5)

            # 记录当前位置为目标位置
            self.current_pan = Config.SERVO_CENTER_ANGLE
            self.current_tilt = Config.SERVO_CENTER_ANGLE
            self.target_pan = Config.SERVO_CENTER_ANGLE
            self.target_tilt = Config.SERVO_CENTER_ANGLE

            self.initialized = True

            print(f"✅ 舵机控制器初始化成功")
            print(f"   - 水平舵机: GPIO{self.pan_pin}")
            print(f"   - 垂直舵机: GPIO{self.tilt_pin}")
            print(f"   - 初始位置: 中心 ({Config.SERVO_CENTER_ANGLE}°)")
            print(f"   - 追踪模式: 已禁用（等待点击选择）")

        except Exception as e:
            print(f"❌ 舵机初始化失败: {e}")
            raise e

    def enable_tracking(self, enabled):
        """启用/禁用舵机追踪"""
        with self.move_lock:
            self.tracking_enabled = enabled
            if not enabled:
                # 禁用时，目标位置设为当前位置，避免突然移动
                self.target_pan = self.current_pan
                self.target_tilt = self.current_tilt
            print(f"🎯 舵机追踪: {'已启用' if enabled else '已禁用'}")

    def set_target(self, pan_angle, tilt_angle):
        """设置目标角度（仅在追踪启用时生效）"""
        if not self.tracking_enabled:
            return  # 追踪禁用时完全忽略目标角度

        pan_angle = max(Config.SERVO_ANGLE_MIN, min(Config.SERVO_ANGLE_MAX, pan_angle))
        tilt_angle = max(Config.SERVO_ANGLE_MIN, min(Config.SERVO_ANGLE_MAX, tilt_angle))

        with self.move_lock:
            self.target_pan = pan_angle
            self.target_tilt = tilt_angle

    def update(self, dt=None):
        """
        更新舵机位置（平滑移动）
        仅在追踪启用时才会移动
        """
        # 如果未初始化或追踪禁用，直接返回
        if not self.initialized or not self.tracking_enabled:
            return

        # 自动计算时间间隔
        current_time = time.time()
        if dt is None:
            dt = min(0.05, current_time - self.last_update_time)
            dt = max(0.01, dt)

        self.last_update_time = current_time

        with self.move_lock:
            # 计算角度差
            pan_diff = self.target_pan - self.current_pan
            tilt_diff = self.target_tilt - self.current_tilt

            # 扩大死区 - 1度以内不移动
            deadband = 1.0
            if abs(pan_diff) < deadband:
                pan_diff = 0
            if abs(tilt_diff) < deadband:
                tilt_diff = 0

            # 如果没有需要移动的，直接返回
            if pan_diff == 0 and tilt_diff == 0:
                return

            # 计算最大移动步长（非常缓慢）
            max_speed = Config.MAX_ANGLE_SPEED  # 度/秒
            max_step = max_speed * dt
            max_step = min(max_step, self.max_angle_change)

            # 非常缓慢的线性移动
            pan_move = 0
            tilt_move = 0

            if pan_diff != 0:
                # 使用更慢的速度因子
                speed_factor = 0.5  # 降低速度到50%
                pan_move = pan_diff * speed_factor * dt
                pan_move = max(-max_step, min(max_step, pan_move))

            if tilt_diff != 0:
                speed_factor = 0.5
                tilt_move = tilt_diff * speed_factor * dt
                tilt_move = max(-max_step, min(max_step, tilt_move))

            # 只有当移动量足够大时才更新
            if abs(pan_move) > 0.2 or abs(tilt_move) > 0.2:
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
        """重置到中心（仅当追踪启用时）"""
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