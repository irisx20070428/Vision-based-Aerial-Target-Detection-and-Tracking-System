# pid_controller.py - 完整修复版
import time
from config import Config


class PanTiltController:
    def __init__(self):
        # PID增益已经很低，进一步降低
        self.pid_pan = PIDController(
            Kp=Config.PID_PAN_Kp * 0.3,  # 再降低到30%
            Ki=Config.PID_PAN_Ki * 0.2,  # 再降低到20%
            Kd=Config.PID_PAN_Kd * 0.5,  # 再降低到50%
            max_output=8  # 减小最大输出到8度
        )

        self.pid_tilt = PIDController(
            Kp=Config.PID_TILT_Kp * 0.3,
            Ki=Config.PID_TILT_Ki * 0.2,
            Kd=Config.PID_TILT_Kd * 0.5,
            max_output=8
        )

        self.image_center_x = Config.IMAGE_CENTER_X
        self.image_center_y = Config.IMAGE_CENTER_Y

        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE

        self.dead_zone = 80  # 更大的死区（80像素）
        self.tracking_enabled = False

        # 角度变化限制 - 非常小
        self.max_angle_change = 1.5  # 单次最大1.5度（原3度）

        self.first_update = True

        print(f"✅ 云台PID控制器初始化成功")
        print(f"   - 图像中心: ({self.image_center_x}, {self.image_center_y})")
        print(f"   - 死区范围: {self.dead_zone}像素")
        print(f"   - 最大输出: {self.max_angle_change}度/次")

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
    """云台控制器 - 根据目标位置计算角度"""

    def __init__(self):
        # 降低PID增益，减少抖动
        self.pid_pan = PIDController(
            Kp=Config.PID_PAN_Kp * 0.5,  # 降低比例增益
            Ki=Config.PID_PAN_Ki * 0.3,  # 降低积分增益
            Kd=Config.PID_PAN_Kd * 0.8,  # 降低微分增益
            max_output=15  # 减小最大输出
        )

        self.pid_tilt = PIDController(
            Kp=Config.PID_TILT_Kp * 0.5,
            Ki=Config.PID_TILT_Ki * 0.3,
            Kd=Config.PID_TILT_Kd * 0.8,
            max_output=15
        )

        # 重要：明确初始化图像中心坐标
        self.image_center_x = Config.IMAGE_CENTER_X
        self.image_center_y = Config.IMAGE_CENTER_Y

        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE

        self.dead_zone = Config.DEAD_ZONE
        self.tracking_enabled = False  # 默认禁用追踪

        # 增加死区，减少微调
        self.dead_zone = 50  # 像素，覆盖Config中的值

        # 角度变化限制
        self.max_angle_change = 3  # 单次最大角度变化

        # 添加初始输出保护
        self.first_update = True  # 首次更新标志

        print(f"✅ 云台PID控制器初始化成功")
        print(f"   - 图像中心: ({self.image_center_x}, {self.image_center_y})")
        print(f"   - 死区范围: {self.dead_zone}像素")
        print(f"   - 最大角度变化: {self.max_angle_change}度/次")
        print(f"   - 初始输出: 禁用状态")

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

        # 首次更新时，不产生任何输出
        if self.first_update:
            self.first_update = False
            return self.current_pan, self.current_tilt

        # 计算误差
        error_x = self.image_center_x - target_x
        error_y = self.image_center_y - target_y

        # 死区处理 - 小误差不移动
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

        # 限制单次变化量
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

    def reset(self):
        """重置控制器"""
        self.pid_pan.reset()
        self.pid_tilt.reset()
        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE
        self.first_update = True  # 重置首次更新标志
        print("🔄 PID控制器已重置")

    def set_tracking_enabled(self, enabled):
        """设置跟踪启用状态"""
        self.tracking_enabled = enabled
        if enabled:
            self.first_update = True  # 重新启用时重置首次更新标志
        print(f"🎯 PID追踪: {'已启用' if enabled else '已禁用'}")

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