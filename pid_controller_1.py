import time
from config import Config


class PIDController:
    def __init__(self, kp, ki, kd, max_output=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.last_d = 0.0

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.last_d = 0.0

    def update(self, error):
        now = time.time()

        if self.last_time is None:
            dt = 0.05
        else:
            dt = now - self.last_time

        if dt < 0.03:
            dt = 0.03
        if dt > 0.2:
            dt = 0.2

        # 穿过中心时清积分，避免冲过头
        if error * self.last_error < 0:
            self.integral = 0.0

        # 小误差才积分，大误差时不让积分越滚越大
        if abs(error) < 12:
            self.integral += error * dt
            if self.integral > 20:
                self.integral = 20
            elif self.integral < -20:
                self.integral = -20
        else:
            self.integral *= 0.9

        d_raw = (error - self.last_error) / dt
        d = 0.3 * d_raw + 0.7 * self.last_d

        output = self.kp * error + self.ki * self.integral + self.kd * d

        if output > self.max_output:
            output = self.max_output
        elif output < -self.max_output:
            output = -self.max_output

        self.last_error = error
        self.last_time = now
        self.last_d = d

        return output


class PanTiltController:
    def __init__(self):
        self.pid_pan = PIDController(
            Config.PID_PAN_Kp,
            Config.PID_PAN_Ki,
            Config.PID_PAN_Kd,
            max_output=10
        )

        self.pid_tilt = PIDController(
            Config.PID_TILT_Kp,
            Config.PID_TILT_Ki,
            Config.PID_TILT_Kd,
            max_output=6
        )

        self.image_width = Config.CAMERA_RESOLUTION[0]
        self.image_height = Config.CAMERA_RESOLUTION[1]
        self.image_center_x = Config.IMAGE_CENTER_X
        self.image_center_y = Config.IMAGE_CENTER_Y

        self.horizontal_fov = Config.CAMERA_HORIZONTAL_FOV
        self.vertical_fov = Config.CAMERA_VERTICAL_FOV

        self.degrees_per_pixel_x = self.horizontal_fov / self.image_width
        self.degrees_per_pixel_y = self.vertical_fov / self.image_height

        self.current_pan = float(Config.SERVO_PAN_INIT_ANGLE)
        self.current_tilt = float(Config.SERVO_TILT_INIT_ANGLE)

        self.servo_min = Config.SERVO_ANGLE_MIN
        self.servo_max = Config.SERVO_ANGLE_MAX

        self.dead_zone_pixels = Config.DEAD_ZONE
        self.tracking_enabled = False

        self.max_angle_change_pan = 8
        self.max_angle_change_tilt = 4

        self.recovery_frames = 0
        self.recovery_limit = 5

        self.debug_counter = 0

        # 这个方向是按你现在日志的行为保留的
        # 如果发现目标越追越偏，把 pan_sign 改成 1
        self.pan_sign = -1
        self.tilt_sign = -1

        print("PID controller ready")

    def pixels_to_angle(self, error_pixels_x, error_pixels_y):
        angle_error_x = error_pixels_x * self.degrees_per_pixel_x
        angle_error_y = error_pixels_y * self.degrees_per_pixel_y
        return angle_error_x, angle_error_y

    def reset(self):
        self.pid_pan.reset()
        self.pid_tilt.reset()
        self.current_pan = float(Config.SERVO_PAN_INIT_ANGLE)
        self.current_tilt = float(Config.SERVO_TILT_INIT_ANGLE)
        self.recovery_frames = 0
        print("PID reset")

    def set_tracking_enabled(self, enabled, initial_angles=None):
        self.tracking_enabled = enabled
        self.pid_pan.reset()
        self.pid_tilt.reset()

        if enabled:
            if initial_angles is not None:
                self.current_pan = float(initial_angles[0])
                self.current_tilt = float(initial_angles[1])
            self.recovery_frames = 2
            print(f"PID tracking on: pan={self.current_pan:.1f}, tilt={self.current_tilt:.1f}")
        else:
            self.recovery_frames = 0
            print("PID tracking off")

    def update_image_center(self, width, height):
        self.image_width = width
        self.image_height = height
        self.image_center_x = width // 2
        self.image_center_y = height // 2
        self.degrees_per_pixel_x = self.horizontal_fov / width
        self.degrees_per_pixel_y = self.vertical_fov / height

    def _clamp(self, value, low, high):
        return max(low, min(high, value))

    def compute_angles(self, target_x, target_y):
        if not self.tracking_enabled or target_x is None or target_y is None:
            return self.current_pan, self.current_tilt

        error_pixels_x = int(target_x - self.image_center_x)
        error_pixels_y = int(target_y - self.image_center_y)

        if abs(error_pixels_x) < self.dead_zone_pixels:
            error_pixels_x = 0
        if abs(error_pixels_y) < self.dead_zone_pixels:
            error_pixels_y = 0

        if error_pixels_x == 0 and error_pixels_y == 0:
            self.pid_pan.reset()
            self.pid_tilt.reset()
            return self.current_pan, self.current_tilt

        angle_error_x, angle_error_y = self.pixels_to_angle(error_pixels_x, error_pixels_y)

        if abs(angle_error_x) < 0.6:
            angle_error_x = 0
            self.pid_pan.reset()

        if abs(angle_error_y) < 0.6:
            angle_error_y = 0
            self.pid_tilt.reset()

        control_x = self.pid_pan.update(angle_error_x) if angle_error_x != 0 else 0
        control_y = self.pid_tilt.update(angle_error_y) if angle_error_y != 0 else 0

        # 目标恢复后的前几帧慢一点
        if self.recovery_frames > 0:
            pan_limit = 3
            tilt_limit = 2
            self.recovery_frames -= 1
        else:
            pan_limit = self.max_angle_change_pan
            tilt_limit = self.max_angle_change_tilt

        # 接近中心时减小每次步长，防止过冲
        if abs(angle_error_x) < 3:
            pan_limit = min(pan_limit, 1.5)
        elif abs(angle_error_x) < 8:
            pan_limit = min(pan_limit, 3)

        if abs(angle_error_y) < 3:
            tilt_limit = min(tilt_limit, 1.2)
        elif abs(angle_error_y) < 8:
            tilt_limit = min(tilt_limit, 2)

        control_x = self._clamp(control_x, -pan_limit, pan_limit)
        control_y = self._clamp(control_y, -tilt_limit, tilt_limit)

        if abs(control_x) < 0.25:
            control_x = 0
        if abs(control_y) < 0.25:
            control_y = 0

        self.current_pan += self.pan_sign * control_x
        self.current_tilt += self.tilt_sign * control_y

        self.current_pan = self._clamp(self.current_pan, self.servo_min, self.servo_max)
        self.current_tilt = self._clamp(self.current_tilt, self.servo_min, self.servo_max)

        # 到边界就清积分，防止卡边界
        if self.current_pan == self.servo_min or self.current_pan == self.servo_max:
            self.pid_pan.integral = 0

        if self.current_tilt == self.servo_min or self.current_tilt == self.servo_max:
            self.pid_tilt.integral = 0

        self.debug_counter += 1
        if self.debug_counter % 20 == 0:
            print(
                f"[PID] err_px=({error_pixels_x},{error_pixels_y}) "
                f"err_deg=({angle_error_x:.2f},{angle_error_y:.2f}) "
                f"ctrl=({control_x:.2f},{control_y:.2f}) "
                f"angle=({self.current_pan:.2f},{self.current_tilt:.2f})"
            )

        return self.current_pan, self.current_tilt

    def get_status(self):
        return {
            "tracking_enabled": self.tracking_enabled,
            "current_pan": self.current_pan,
            "current_tilt": self.current_tilt,
            "recovery_frames": self.recovery_frames,
            "degrees_per_pixel": (self.degrees_per_pixel_x, self.degrees_per_pixel_y),
        }
