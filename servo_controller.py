# servo_controller.py - PCA9685 底层驱动版本
import time
import board
import busio
from adafruit_pca9685 import PCA9685
from config import Config

class ServoController:
    def __init__(self):
        # 初始化 I2C 和 PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50   # 标准舵机频率

        # 通道配置（根据你的接线）
        self.pan_ch = 1   # 水平舵机接 CH1
        self.tilt_ch = 3  # 垂直舵机接 CH3

        # 角度 -> 占空比映射 (0°->0.5ms, 180°->2.5ms)
        self.min_duty = int(0.5 / 20 * 65535)   # 约 1638
        self.max_duty = int(2.5 / 20 * 65535)   # 约 8192

        # 初始角度（从 config 读取）
        self.current_pan = Config.SERVO_PAN_INIT_ANGLE
        self.current_tilt = Config.SERVO_TILT_INIT_ANGLE
        self.tracking_enabled = False

        # 设置初始位置
        self._set_angle(self.pan_ch, self.current_pan)
        self._set_angle(self.tilt_ch, self.current_tilt)

        print(f"✅ 舵机控制器 (PCA9685) 初始化成功")
        print(f"   - 水平舵机: CH{self.pan_ch}")
        print(f"   - 垂直舵机: CH{self.tilt_ch}")

    def _angle_to_duty(self, angle):
        """角度转占空比 (0-65535)"""
        ratio = angle / 180.0
        return int(self.min_duty + ratio * (self.max_duty - self.min_duty))

    def _set_angle(self, channel, angle):
        """立即设置指定通道的角度"""
        duty = self._angle_to_duty(angle)
        self.pca.channels[channel].duty_cycle = duty

    def set_target(self, pan_angle, tilt_angle):
        """设置目标角度（如果追踪启用）"""
        if not self.tracking_enabled:
            return
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))
        self._set_angle(self.pan_ch, pan_angle)
        self._set_angle(self.tilt_ch, tilt_angle)
        self.current_pan = pan_angle
        self.current_tilt = tilt_angle

        print(f"[SERVO] set_target({pan_angle}, {tilt_angle}), tracking_enabled={self.tracking_enabled}")

    def get_current_angles(self):
        """获取当前角度"""
        return self.current_pan, self.current_tilt

    def enable_tracking(self, enabled):
        """启用/禁用舵机追踪"""
        self.tracking_enabled = enabled
        print(f"🎯 舵机追踪: {'已启用' if enabled else '已禁用'}")

    def reset_to_center(self):
        """重置到初始角度（如果追踪启用）"""
        if self.tracking_enabled:
            self.set_target(Config.SERVO_PAN_INIT_ANGLE, Config.SERVO_TILT_INIT_ANGLE)

    def set_angle_immediate(self, pan_angle, tilt_angle):
        """立即设置角度（无视 tracking_enabled，用于退出归位）"""
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))
        self._set_angle(self.pan_ch, pan_angle)
        self._set_angle(self.tilt_ch, tilt_angle)
        self.current_pan = pan_angle
        self.current_tilt = tilt_angle

    def update(self, dt=None):
        """
        为了兼容主程序中的调用，保留 update 方法。
        由于 PCA9685 直接设置角度，不需要平滑移动，此方法可空实现。
        如果后续需要平滑移动，可以在这里实现插值逻辑。
        """
        pass

    def cleanup(self):
        """释放资源，停止 PWM 信号"""
        self.pca.channels[self.pan_ch].duty_cycle = 0
        self.pca.channels[self.tilt_ch].duty_cycle = 0
        self.pca.deinit()
        print("✅ 舵机已关闭")