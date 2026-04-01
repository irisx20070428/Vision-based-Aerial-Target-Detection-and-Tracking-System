# config.py
import os


class Config:
    """系统配置"""

    # 摄像头配置 - 改为 usb
    CAMERA_TYPE = "usb"  # 改为 usb，不使用 picamera2
    CAMERA_RESOLUTION = (640, 480)
    CAMERA_FPS = 30

    # YOLO配置
    YOLO_CONF_THRESHOLD = 0.5
    YOLO_MODEL = 'yolov5s'
    YOLO_TARGET_CLASSES = [0]
    YOLO_FRAME_SKIP = 2

    # 舵机配置
    SERVO_PAN_PIN = 18
    SERVO_TILT_PIN = 27
    SERVO_FREQUENCY = 50
    SERVO_ANGLE_MIN = 0
    SERVO_ANGLE_MAX = 180
    SERVO_CENTER_ANGLE = 90

    # PID控制参数
    PID_PAN_Kp = 0.25
    PID_PAN_Ki = 0.01
    PID_PAN_Kd = 0.05

    PID_TILT_Kp = 0.25
    PID_TILT_Ki = 0.01
    PID_TILT_Kd = 0.05

    # 控制参数
    MAX_ANGLE_SPEED = 30
    ACCELERATION = 0.3
    DEAD_ZONE = 15

    # 图像中心点
    IMAGE_CENTER_X = CAMERA_RESOLUTION[0] // 2
    IMAGE_CENTER_Y = CAMERA_RESOLUTION[1] // 2

    # 特征追踪配置
    SIMILARITY_THRESHOLD = 0.6
    TRACKING_HISTORY_LEN = 100
    USE_SIMPLIFIED_FEATURES = True

    # 日志配置
    LOG_LEVEL = "INFO"
    SAVE_DATASET = True
    DATASET_DIR = "person_dataset"