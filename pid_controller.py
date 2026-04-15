# pid_controller.py - 修改 compute_angles 方法
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

        # 添加输出平滑
        self.last_output = 0
        self.output_smoothing = 0.3  # 平滑因子

        self.output_min = -max_output
        self.output_max = max_output

    def reset(self):
        """重置控制器"""
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.last_output = 0

    def update(self, error):
        """更新PID输出"""
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

        # 计算原始输出
        raw_output = P + I + D

        # 平滑输出（低通滤波）
        output = self.last_output * (1 - self.output_smoothing) + raw_output * self.output_smoothing

        # 限制输出
        output = max(self.output_min, min(self.output_max, output))

        # 更新状态
        self.last_error = error
        self.last_time = current_time
        self.last_output = output

        return output


class PanTiltController:
    def __init__(self):
        # ... 原有代码 ...

        # 添加初始输出保护
        self.first_update = True  # 首次更新标志
        self.initial_pan = Config.SERVO_CENTER_ANGLE
        self.initial_tilt = Config.SERVO_CENTER_ANGLE

        print(f"✅ 云台PID控制器初始化成功")
        print(f"   - 图像中心: ({self.image_center_x}, {self.image_center_y})")
        print(f"   - 死区范围: {self.dead_zone}像素")
        print(f"   - 初始输出: 禁用状态")

    def compute_angles(self, target_x, target_y):
        """
        根据目标位置计算云台角度
        """
        if not self.tracking_enabled or target_x is None or target_y is None:
            return self.current_pan, self.current_tilt

        # 首次更新时，不产生任何输出
        if self.first_update:
            self.first_update = False
            return self.current_pan, self.current_tilt

        # 计算误差
        error_x = self.image_center_x - target_x
        error_y = self.image_center_y - target_y

        # 扩大死区 - 50像素以内不移动
        if abs(error_x) < self.dead_zone:
            error_x = 0
        if abs(error_y) < self.dead_zone:
            error_y = 0

        # 如果误差很小，不移动
        if error_x == 0 and error_y == 0:
            return self.current_pan, self.current_tilt

        # PID计算
        control_x = self.pid_pan.update(error_x)
        control_y = self.pid_tilt.update(error_y)

        # 限制单次变化量（更小）
        self.max_angle_change = 3  # 降低到3度
        control_x = max(-self.max_angle_change, min(self.max_angle_change, control_x))
        control_y = max(-self.max_angle_change, min(self.max_angle_change, control_y))

        # 更新角度
        self.current_pan += control_x
        self.current_tilt += control_y

        # 限制角度范围
        self.current_pan = max(Config.SERVO_ANGLE_MIN,
                               min(Config.SERVO_ANGLE_MAX, self.current_pan))
        self.current_tilt = max(Config.SERVO_ANGLE_MIN,
                                min(Config.SERVO_ANGLE_MAX, self.current_tilt))

        return self.current_pan, self.current_tilt