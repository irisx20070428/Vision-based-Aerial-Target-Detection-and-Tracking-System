import time
import board
import busio
from adafruit_pca9685 import PCA9685
from config import Config


class ServoController:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50

        # 按你现在项目的接线
        self.pan_ch = 1
        self.tilt_ch = 3

        # SG90 常用脉宽范围：0.5ms ~ 2.5ms，对应 0° ~ 180°
        self.min_duty = int(0.5 / 20 * 65535)
        self.max_duty = int(2.5 / 20 * 65535)

        self.current_pan = float(Config.SERVO_PAN_INIT_ANGLE)
        self.current_tilt = float(Config.SERVO_TILT_INIT_ANGLE)

        self.target_pan = self.current_pan
        self.target_tilt = self.current_tilt

        self.tracking_enabled = False

        # 平滑移动速度（度/秒）
        self.max_speed = getattr(Config, "MAX_ANGLE_SPEED", 180)

        self.last_update_time = time.time()

        # 上电先到初始位置
        self._set_angle(self.pan_ch, self.current_pan)
        self._set_angle(self.tilt_ch, self.current_tilt)

        print("Servo controller ready (PCA9685)")
        print(f"pan channel: CH{self.pan_ch}")
        print(f"tilt channel: CH{self.tilt_ch}")

    def _clamp(self, value, low, high):
        return max(low, min(high, value))

    def _angle_to_duty(self, angle):
        angle = self._clamp(angle, 0, 180)
        ratio = angle / 180.0
        return int(self.min_duty + ratio * (self.max_duty - self.min_duty))

    def _set_angle(self, channel, angle):
        duty = self._angle_to_duty(angle)
        self.pca.channels[channel].duty_cycle = duty

    def _move_towards(self, current, target, max_step):
        diff = target - current
        if abs(diff) <= max_step:
            return target
        if diff > 0:
            return current + max_step
        return current - max_step

    def set_target(self, pan_angle, tilt_angle):
        if not self.tracking_enabled:
            return

        self.target_pan = self._clamp(pan_angle, Config.SERVO_ANGLE_MIN, Config.SERVO_ANGLE_MAX)
        self.target_tilt = self._clamp(tilt_angle, Config.SERVO_ANGLE_MIN, Config.SERVO_ANGLE_MAX)

    def update(self, dt=None):
        now = time.time()

        if dt is None:
            dt = now - self.last_update_time

        self.last_update_time = now

        if dt <= 0:
            dt = 0.02

        max_step = self.max_speed * dt

        new_pan = self._move_towards(self.current_pan, self.target_pan, max_step)
        new_tilt = self._move_towards(self.current_tilt, self.target_tilt, max_step)

        changed = False

        if abs(new_pan - self.current_pan) > 0.01:
            self.current_pan = new_pan
            self._set_angle(self.pan_ch, self.current_pan)
            changed = True

        if abs(new_tilt - self.current_tilt) > 0.01:
            self.current_tilt = new_tilt
            self._set_angle(self.tilt_ch, self.current_tilt)
            changed = True

        return changed

    def get_current_angles(self):
        return self.current_pan, self.current_tilt

    def enable_tracking(self, enabled):
        self.tracking_enabled = enabled
        print(f"Servo tracking {'enabled' if enabled else 'disabled'}")

    def reset_to_center(self):
        self.target_pan = float(Config.SERVO_PAN_INIT_ANGLE)
        self.target_tilt = float(Config.SERVO_TILT_INIT_ANGLE)

    def set_angle_immediate(self, pan_angle, tilt_angle):
        pan_angle = self._clamp(pan_angle, Config.SERVO_ANGLE_MIN, Config.SERVO_ANGLE_MAX)
        tilt_angle = self._clamp(tilt_angle, Config.SERVO_ANGLE_MIN, Config.SERVO_ANGLE_MAX)

        self.current_pan = pan_angle
        self.current_tilt = tilt_angle
        self.target_pan = pan_angle
        self.target_tilt = tilt_angle

        self._set_angle(self.pan_ch, self.current_pan)
        self._set_angle(self.tilt_ch, self.current_tilt)

    def cleanup(self):
        try:
            self.pca.channels[self.pan_ch].duty_cycle = 0
            self.pca.channels[self.tilt_ch].duty_cycle = 0
        finally:
            self.pca.deinit()
            print("Servo controller closed")
