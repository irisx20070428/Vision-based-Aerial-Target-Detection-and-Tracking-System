# config.py - 添加摄像头视野配置
import os


class Config:
    """系统配置 - 修正版"""

    # 摄像头配置
    CAMERA_TYPE = "usb"
    CAMERA_RESOLUTION = (640, 480)
    CAMERA_FPS = 15

    # 摄像头视野角度（根据实际摄像头调整）
    # 常见USB摄像头水平视野约60-70度，垂直约40-50度
    CAMERA_HORIZONTAL_FOV = 65.0  # 水平视野角度
    CAMERA_VERTICAL_FOV = 48.0  # 垂直视野角度

    # YOLO配置
    YOLO_CONF_THRESHOLD = 0.5
    YOLO_MODEL = 'yolov5n'
    YOLO_TARGET_CLASSES = [0]
    YOLO_FRAME_SKIP = 4
    YOLO_IMAGE_SIZE = 320

    # 舵机配置
    SERVO_PAN_PIN = 18
    SERVO_TILT_PIN = 27
    SERVO_FREQUENCY = 50
    SERVO_ANGLE_MIN = 0
    SERVO_ANGLE_MAX = 180
    SERVO_PAN_INIT_ANGLE = 90  # 水平初始角度，例如 0 度（最左）
    SERVO_TILT_INIT_ANGLE = 40  # 垂直初始角度，例如 0 度（水平）

    # 最大转动速度（度/秒）
    MAX_ANGLE_SPEED = 180 # 60度/秒

    ACCELERATION = 0.05
    DEAD_ZONE = 10  # 像素死区

    # PID控制参数
    PID_PAN_Kp = 0.25  # 比例增益
    PID_PAN_Ki = 0.01  # 积分增益
    PID_PAN_Kd = 0.01  # 微分增益
    PID_TILT_Kp = 0.25
    PID_TILT_Ki = 0.01
    PID_TILT_Kd = 0.01

    # 图像中心点
    IMAGE_CENTER_X = CAMERA_RESOLUTION[0] // 2  # 320
    IMAGE_CENTER_Y = CAMERA_RESOLUTION[1] // 2  # 240

    # 特征追踪配置
    SIMILARITY_THRESHOLD = 0.45
    TRACKING_HISTORY_LEN = 1000

    # 日志配置
    LOG_LEVEL = "INFO"
    SAVE_DATASET = True
    DATASET_DIR = "person_dataset"
