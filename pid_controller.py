# pid_controller.py - 修正像素到角度的转换
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
        self.output_smoothing = 0.3

        self.output_min = -max_output
        self.output_max = max_output

    def reset(self):
        """重置控制器"""
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.last_output = 0

    def update(self, error_degrees):
        """
        更新PID输出（输入已经是角度单位）

        参数:
            error_degrees: 角度误差（度）
        """
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0:
            dt = 0.02

        # 比例项
        P = self.Kp * error_degrees

        # 积分项
        self.integral += error_degrees * dt
        self.integral = max(-100, min(100, self.integral))
        I = self.Ki * self.integral

        # 微分项
        derivative = (error_degrees - self.last_error) / dt
        D = self.Kd * derivative

        # 计算原始输出
        raw_output = P + I + D

        # 平滑输出
        output = self.last_output * (1 - self.output_smoothing) + raw_output * self.output_smoothing

        # 限制输出
        output = max(self.output_min, min(self.output_max, output))

        # 更新状态
        self.last_error = error_degrees
        self.last_time = current_time
        self.last_output = output

        return output


class PanTiltController:
    """云台控制器 - 将像素误差转换为角度误差"""

    def __init__(self):
        # PID控制器（输入已经是角度误差）
        self.pid_pan = PIDController(
            Kp=Config.PID_PAN_Kp,
            Ki=Config.PID_PAN_Ki,
            Kd=Config.PID_PAN_Kd,
            max_output=15
        )

        self.pid_tilt = PIDController(
            Kp=Config.PID_TILT_Kp,
            Ki=Config.PID_TILT_Ki,
            Kd=Config.PID_TILT_Kd,
            max_output=15
        )

        # 图像尺寸和中心
        self.image_width = Config.CAMERA_RESOLUTION[0]
        self.image_height = Config.CAMERA_RESOLUTION[1]
        self.image_center_x = Config.IMAGE_CENTER_X
        self.image_center_y = Config.IMAGE_CENTER_Y

        # 摄像头视野角度
        self.horizontal_fov = Config.CAMERA_HORIZONTAL_FOV  # 水平视野（度）
        self.vertical_fov = Config.CAMERA_VERTICAL_FOV  # 垂直视野（度）

        # 计算每像素对应的角度
        # 水平：总视野角度 / 图像宽度 = 每像素角度
        self.degrees_per_pixel_x = self.horizontal_fov / self.image_width
        self.degrees_per_pixel_y = self.vertical_fov / self.image_height

        # 当前舵机角度
        self.current_pan = Config.SERVO_CENTER_ANGLE
        self.current_tilt = Config.SERVO_CENTER_ANGLE

        # 舵机角度范围
        self.servo_min = Config.SERVO_ANGLE_MIN
        self.servo_max = Config.SERVO_ANGLE_MAX
        self.servo_center = Config.SERVO_CENTER_ANGLE

        # 死区（像素）
        self.dead_zone_pixels = Config.DEAD_ZONE

        # 追踪启用标志
        self.tracking_enabled = False

        # 角度变化限制
        self.max_angle_change = 10  # 单次最大变化10度

        # 首次更新标志
        self.first_update = True

        # 调试计数器
        self.debug_counter = 0

        print(f"✅ 云台PID控制器初始化成功")
        print(f"   - 图像分辨率: {self.image_width}x{self.image_height}")
        print(f"   - 图像中心: ({self.image_center_x}, {self.image_center_y})")
        print(f"   - 水平视野: {self.horizontal_fov}° → {self.degrees_per_pixel_x:.3f}°/像素")
        print(f"   - 垂直视野: {self.vertical_fov}° → {self.degrees_per_pixel_y:.3f}°/像素")
        print(f"   - 像素死区: {self.dead_zone_pixels}px")
        print(f"   - 最大角度变化: {self.max_angle_change}°/次")

    def pixels_to_angle(self, error_pixels_x, error_pixels_y):
        """
        将像素误差转换为角度误差

        参数:
            error_pixels_x: 水平像素误差（目标x - 中心x）
            error_pixels_y: 垂直像素误差（目标y - 中心y）

        返回:
            angle_error_x, angle_error_y: 角度误差（度）
            - 正误差表示目标在右侧，需要向右转
            - 负误差表示目标在左侧，需要向左转
        """
        # 像素误差转换为角度误差
        # 注意：实际摄像头成像是倒像，需要根据实际测试调整符号
        angle_error_x = error_pixels_x * self.degrees_per_pixel_x
        angle_error_y = error_pixels_y * self.degrees_per_pixel_y

        return angle_error_x, angle_error_y

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

        # 首次更新时，不产生任何输出
        if self.first_update:
            self.first_update = False
            print(f"🎯 PID控制器首次激活")
            print(f"   - 目标位置: ({target_x}, {target_y})")
            print(f"   - 图像中心: ({self.image_center_x}, {self.image_center_y})")
            return self.current_pan, self.current_tilt

        # 计算像素误差
        error_pixels_x = target_x - self.image_center_x  # 正：目标在右侧
        error_pixels_y = target_y - self.image_center_y  # 正：目标在下侧

        # 死区处理（像素级）
        if abs(error_pixels_x) < self.dead_zone_pixels:
            error_pixels_x = 0
        if abs(error_pixels_y) < self.dead_zone_pixels:
            error_pixels_y = 0

        # 将像素误差转换为角度误差
        angle_error_x, angle_error_y = self.pixels_to_angle(error_pixels_x, error_pixels_y)

        # 调试输出（每50帧）
        self.debug_counter += 1
        if self.debug_counter % 50 == 0 and (abs(angle_error_x) > 0.1 or abs(angle_error_y) > 0.1):
            print(f"📊 角度计算 - 像素误差:({error_pixels_x:+d},{error_pixels_y:+d})px "
                  f"→ 角度误差:({angle_error_x:+.1f}°,{angle_error_y:+.1f}°) "
                  f"当前角度:({self.current_pan:.1f}°,{self.current_tilt:.1f}°)")

        # 如果角度误差很小，不移动
        if abs(angle_error_x) < 0.3 and abs(angle_error_y) < 0.3:
            return self.current_pan, self.current_tilt

        # PID计算（输入已经是角度误差）
        control_x = self.pid_pan.update(angle_error_x)  # 水平控制量
        control_y = self.pid_tilt.update(angle_error_y)  # 垂直控制量

        # 限制单次变化量
        control_x = max(-self.max_angle_change, min(self.max_angle_change, control_x))
        control_y = max(-self.max_angle_change, min(self.max_angle_change, control_y))

        # 更新角度（注意：舵机控制方向可能需要取反）
        # 如果目标在右侧（正误差），舵机应该向右转（增加角度）
        self.current_pan += control_x
        self.current_tilt += control_y

        # 限制角度范围
        self.current_pan = max(self.servo_min, min(self.servo_max, self.current_pan))
        self.current_tilt = max(self.servo_min, min(self.servo_max, self.current_tilt))

        return self.current_pan, self.current_tilt

    def reset(self):
        """重置PID控制器和角度"""
        self.pid_pan.reset()
        self.pid_tilt.reset()
        self.current_pan = self.servo_center
        self.current_tilt = self.servo_center
        self.first_update = True
        print("🔄 PID控制器已重置，角度已回中")

    def set_tracking_enabled(self, enabled):
        """设置追踪启用状态"""
        self.tracking_enabled = enabled
        if enabled:
            self.first_update = True
            print(f"🎯 PID追踪已启用")
        else:
            print(f"🎯 PID追踪已禁用")

    def update_image_center(self, width, height):
        """更新图像中心坐标"""
        self.image_width = width
        self.image_height = height
        self.image_center_x = width // 2
        self.image_center_y = height // 2
        # 重新计算每像素角度
        self.degrees_per_pixel_x = self.horizontal_fov / width
        self.degrees_per_pixel_y = self.vertical_fov / height
        print(f"📐 图像中心已更新: ({self.image_center_x}, {self.image_center_y})")
        print(f"   - 每像素角度: {self.degrees_per_pixel_x:.3f}°/px")

    def get_status(self):
        """获取控制器状态"""
        return {
            'pan_angle': self.current_pan,
            'tilt_angle': self.current_tilt,
            'tracking_enabled': self.tracking_enabled,
            'image_center': (self.image_center_x, self.image_center_y),
            'degrees_per_pixel': (self.degrees_per_pixel_x, self.degrees_per_pixel_y)
        }