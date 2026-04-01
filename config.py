# config.py - 完整版
import os


class Config:
    """系统配置"""

    # 摄像头配置
    CAMERA_TYPE = "usb"  # 改为 usb
    CAMERA_RESOLUTION = (640, 480)
    CAMERA_FPS = 30

    # YOLO配置
    YOLO_CONF_THRESHOLD = 0.5
    YOLO_MODEL = 'yolov5s'  # 可选: yolov5n, yolov5s, yolov5m
    YOLO_TARGET_CLASSES = [0]  # 0 = person
    YOLO_FRAME_SKIP = 2  # 跳帧检测，提高性能（添加这个配置）

    # 舵机配置
    SERVO_PAN_PIN = 18  # 水平舵机 GPIO18
    SERVO_TILT_PIN = 27  # 垂直舵机 GPIO27
    SERVO_FREQUENCY = 50  # 50Hz
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
    MAX_ANGLE_SPEED = 30  # 最大角速度 (度/秒)
    ACCELERATION = 0.3  # 加速度系数
    DEAD_ZONE = 15  # 死区范围 (像素)

    # 图像中心点
    IMAGE_CENTER_X = CAMERA_RESOLUTION[0] // 2
    IMAGE_CENTER_Y = CAMERA_RESOLUTION[1] // 2

    # 特征追踪配置
    SIMILARITY_THRESHOLD = 0.6
    TRACKING_HISTORY_LEN = 100
    USE_SIMPLIFIED_FEATURES = True  # 树莓派使用简化特征（添加这个配置）

    # 日志配置
    LOG_LEVEL = "INFO"
    SAVE_DATASET = True
    DATASET_DIR = "person_dataset"