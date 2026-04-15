# config.py - 确保包含以下配置
import os


class Config:
    """系统配置 - 慢速平滑追踪版"""

    # 摄像头配置
    CAMERA_TYPE = "usb"
    CAMERA_RESOLUTION = (640, 480)
    CAMERA_FPS = 15

    # YOLO配置
    YOLO_CONF_THRESHOLD = 0.5
    YOLO_MODEL = 'yolov5n'
    YOLO_TARGET_CLASSES = [0]
    YOLO_FRAME_SKIP = 3
    YOLO_IMAGE_SIZE = 320

    # 舵机配置 - 非常慢的速度
    SERVO_PAN_PIN = 18
    SERVO_TILT_PIN = 27
    SERVO_FREQUENCY = 50
    SERVO_ANGLE_MIN = 0
    SERVO_ANGLE_MAX = 180
    SERVO_CENTER_ANGLE = 90

    # 最大转动速度（度/秒）
    MAX_ANGLE_SPEED = 10  # 10度/秒，非常慢

    ACCELERATION = 0.03  # 加速度
    DEAD_ZONE = 50  # 死区（像素）

    # PID控制参数 - 低增益
    PID_PAN_Kp = 0.04  # 比例增益
    PID_PAN_Ki = 0.0005  # 积分增益
    PID_PAN_Kd = 0.002  # 微分增益
    PID_TILT_Kp = 0.04  # 比例增益
    PID_TILT_Ki = 0.0005  # 积分增益
    PID_TILT_Kd = 0.002  # 微分增益

    # 图像中心点（重要：必须定义）
    IMAGE_CENTER_X = CAMERA_RESOLUTION[0] // 2  # 320
    IMAGE_CENTER_Y = CAMERA_RESOLUTION[1] // 2  # 240

    # 特征追踪配置
    SIMILARITY_THRESHOLD = 0.45
    TRACKING_HISTORY_LEN = 1000

    # 日志配置
    LOG_LEVEL = "INFO"
    SAVE_DATASET = True
    DATASET_DIR = "person_dataset"