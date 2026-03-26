# pid_controller.py
import time
from config import Config


class PIDController:
    """PID控制器 - 平滑控制云台"""

    def __init__(self, Kp, Ki, Kd, max_output=30):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output

        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

        self.output_min = -max_output
        self.output_max = max_output

    def reset(self):
        """重置控制器"""
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        """
        更新PID输出

        参数:
            error: 当前误差（目标位置 - 当前位置）

        返回:
            output: 控制输出
        """
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0:
            dt = 0.02

        # 比例项
        P = self.Kp * error

        # 积分项（抗积分饱和）
        self.integral += error * dt
        self.integral = max(-100, min(100, self.integral))
        I = self.Ki * self.integral

        # 微分项
        derivative = (error - self.last_error) / dt
        D = self.Kd * derivative

        # 计算输出
        output = P + I + D

        # 限制输出
        output = max(self.output_min, min(self.output_max, output))

        # 更新状态
        self.last_error = error
        self.last_time = current_time

        return output


class PanTiltController:
    """云台控制器 - 根据目标位置计算角度"""

    def __init__(self):
        # 水平PID
        self.pid_pan = PIDController(
            Config.PID_PAN_Kp,
            Config.PID_PAN_Ki,
            Config.PID_PAN_Kd
        )

        # 垂直PID
        self.pid_tilt = PIDController(
            Config.PID_TILT_Kp,
            Config.PID_TILT_Ki,
            Config.PID_TILT_Kd
        )

        self.image_center_x = Config.IMAGE_CENTER_X
        self.image_center_y = Config.IMAGE_CENTER_Y

        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE

        self.dead_zone = Config.DEAD_ZONE
        self.tracking_enabled = True

        print(f"✅ 云台PID控制器初始化成功")
        print(f"   - 图像中心: ({self.image_center_x}, {self.image_center_y})")
        print(f"   - 死区范围: {self.dead_zone}像素")

    def compute_angles(self, target_x, target_y):
        """
        根据目标位置计算云台角度

        参数:
            target_x, target_y: 目标在图像中的位置

        返回:
            pan_angle, tilt_angle: 水平和垂直角度
        """
        if not self.tracking_enabled or target_x is None or target_y is None:
            return self.current_pan, self.current_tilt

        # 计算误差
        error_x = self.image_center_x - target_x
        error_y = self.image_center_y - target_y

        # 死区处理 - 小误差不移动
        if abs(error_x) < self.dead_zone:
            error_x = 0
        if abs(error_y) < self.dead_zone:
            error_y = 0

        # PID计算
        control_x = self.pid_pan.update(error_x)
        control_y = self.pid_tilt.update(error_y)

        # 更新角度
        self.current_pan += control_x
        self.current_tilt += control_y

        # 限制角度范围
        self.current_pan = max(Config.SERVO_ANGLE_MIN,
                               min(Config.SERVO_ANGLE_MAX, self.current_pan))
        self.current_tilt = max(Config.SERVO_ANGLE_MIN,
                                min(Config.SERVO_ANGLE_MAX, self.current_tilt))

        return self.current_pan, self.current_tilt

    def reset(self):
        """重置控制器"""
        self.pid_pan.reset()
        self.pid_tilt.reset()
        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE

    def set_tracking_enabled(self, enabled):
        """设置跟踪启用状态"""
        self.tracking_enabled = enabled

    def update_image_center(self, width, height):
        """更新图像中心"""
        self.image_center_x = width // 2
        self.image_center_y = height // 2
        print(f"📐 图像中心更新: ({self.image_center_x}, {self.image_center_y})")

    def get_status(self):
        """获取控制器状态"""
        return {
            'pan_angle': self.current_pan,
            'tilt_angle': self.current_tilt,
            'tracking_enabled': self.tracking_enabled,
            'error': (self.pid_pan.last_error, self.pid_tilt.last_error)
        }