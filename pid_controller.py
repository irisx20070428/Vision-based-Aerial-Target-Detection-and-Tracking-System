# pid_controller.py - 完整修复版（慢速平滑追踪）
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
        # 限制积分项范围
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
    """云台控制器 - 根据目标位置计算角度（慢速平滑版）"""

    def __init__(self):
        # 从配置读取PID参数，并进一步降低增益以实现慢速追踪
        # 使用更低的增益，让舵机转动更慢、更平滑
        self.pid_pan = PIDController(
            Kp=Config.PID_PAN_Kp * 0.3,  # 降低到30%
            Ki=Config.PID_PAN_Ki * 0.2,  # 降低到20%
            Kd=Config.PID_PAN_Kd * 0.5,  # 降低到50%
            max_output=8  # 减小最大输出到8度
        )

        self.pid_tilt = PIDController(
            Kp=Config.PID_TILT_Kp * 0.3,
            Ki=Config.PID_TILT_Ki * 0.2,
            Kd=Config.PID_TILT_Kd * 0.5,
            max_output=8
        )

        # 图像中心坐标（从配置读取）
        self.image_center_x = Config.IMAGE_CENTER_X
        self.image_center_y = Config.IMAGE_CENTER_Y

        # 当前舵机角度
        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE

        # 死区范围（像素）- 扩大死区，减少微小移动
        self.dead_zone = 80  # 80像素以内不响应

        # 追踪启用标志
        self.tracking_enabled = False

        # 角度变化限制（度）- 单次最大变化量
        self.max_angle_change = 1.5  # 每次最多变化1.5度

        # 首次更新标志（避免初始突变）
        self.first_update = True

        print(f"✅ 云台PID控制器初始化成功")
        print(f"   - 图像中心: ({self.image_center_x}, {self.image_center_y})")
        print(f"   - 死区范围: {self.dead_zone}像素")
        print(f"   - 最大角度变化: {self.max_angle_change}度/次")
        print(f"   - PID最大输出: 8度")
        print(f"   - 追踪状态: 已禁用")

    def compute_angles(self, target_x, target_y):
        """
        根据目标位置计算云台角度

        参数:
            target_x, target_y: 目标在图像中的位置（像素坐标）

        返回:
            pan_angle, tilt_angle: 水平和垂直角度（0-180度）
        """
        # 如果追踪未启用，返回当前位置
        if not self.tracking_enabled or target_x is None or target_y is None:
            return self.current_pan, self.current_tilt

        # 首次更新时，不产生任何输出，避免初始化时的突变
        if self.first_update:
            self.first_update = False
            print(f"🎯 PID控制器首次激活，保持当前位置")
            return self.current_pan, self.current_tilt

        # 计算误差（图像中心 - 目标位置）
        # 正误差表示目标在中心左侧/上方，需要向右/下转动
        error_x = self.image_center_x - target_x
        error_y = self.image_center_y - target_y

        # 死区处理 - 误差小于死区时不移动
        if abs(error_x) < self.dead_zone:
            error_x = 0
        if abs(error_y) < self.dead_zone:
            error_y = 0

        # 如果误差为零，不移动
        if error_x == 0 and error_y == 0:
            return self.current_pan, self.current_tilt

        # PID计算控制量
        control_x = self.pid_pan.update(error_x)
        control_y = self.pid_tilt.update(error_y)

        # 限制单次变化量（防止突变）
        control_x = max(-self.max_angle_change, min(self.max_angle_change, control_x))
        control_y = max(-self.max_angle_change, min(self.max_angle_change, control_y))

        # 更新角度
        self.current_pan += control_x
        self.current_tilt += control_y

        # 限制角度范围（0-180度）
        self.current_pan = max(Config.SERVO_ANGLE_MIN,
                               min(Config.SERVO_ANGLE_MAX, self.current_pan))
        self.current_tilt = max(Config.SERVO_ANGLE_MIN,
                                min(Config.SERVO_ANGLE_MAX, self.current_tilt))

        return self.current_pan, self.current_tilt

    def reset(self):
        """重置PID控制器和角度"""
        self.pid_pan.reset()
        self.pid_tilt.reset()
        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE
        self.first_update = True  # 重置首次更新标志
        print("🔄 PID控制器已重置，角度已回中")

    def set_tracking_enabled(self, enabled):
        """设置追踪启用状态"""
        self.tracking_enabled = enabled
        if enabled:
            self.first_update = True  # 重新启用时重置首次更新标志
            print(f"🎯 PID追踪已启用（死区:{self.dead_zone}px, 最大变化:{self.max_angle_change}°）")
        else:
            print(f"🎯 PID追踪已禁用")

    def update_image_center(self, width, height):
        """更新图像中心坐标（当分辨率改变时使用）"""
        self.image_center_x = width // 2
        self.image_center_y = height // 2
        print(f"📐 图像中心已更新: ({self.image_center_x}, {self.image_center_y})")

    def set_dead_zone(self, dead_zone_pixels):
        """动态设置死区大小"""
        self.dead_zone = dead_zone_pixels
        print(f"📐 死区已更新: {self.dead_zone}像素")

    def set_max_angle_change(self, max_change_degrees):
        """动态设置最大角度变化量"""
        self.max_angle_change = max_change_degrees
        print(f"📐 最大角度变化已更新: {self.max_angle_change}度/次")

    def get_status(self):
        """获取控制器状态（用于调试）"""
        return {
            'pan_angle': self.current_pan,
            'tilt_angle': self.current_tilt,
            'tracking_enabled': self.tracking_enabled,
            'dead_zone': self.dead_zone,
            'max_angle_change': self.max_angle_change,
            'error': (self.pid_pan.last_error, self.pid_tilt.last_error)
        }

    def get_debug_info(self):
        """获取调试信息"""
        return {
            'image_center': (self.image_center_x, self.image_center_y),
            'current_angles': (self.current_pan, self.current_tilt),
            'pid_outputs': (self.pid_pan.last_output, self.pid_tilt.last_output),
            'first_update': self.first_update
        }