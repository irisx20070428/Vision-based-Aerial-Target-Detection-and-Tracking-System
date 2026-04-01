# config.py - 调整追踪参数
import os


class Config:
    """系统配置"""

    # 摄像头配置
    CAMERA_TYPE = "usb"
    CAMERA_RESOLUTION = (320, 240)  # 保持较小分辨率
    CAMERA_FPS = 15

    # YOLO配置
    YOLO_CONF_THRESHOLD = 0.5
    YOLO_MODEL = 'yolov5n'
    YOLO_TARGET_CLASSES = [0]
    YOLO_FRAME_SKIP = 2  # 降低跳帧，提高追踪连续性
    YOLO_IMAGE_SIZE = 320

    # 特征追踪配置 - 降低阈值，提高追踪成功率
    SIMILARITY_THRESHOLD = 0.65  # 初始阈值
    TRACKING_HISTORY_LEN = 100
    USE_SIMPLIFIED_FEATURES = False  # 使用完整特征
    FEATURE_EXTRACT_INTERVAL = 1  # 每帧都提取特征

    # 舵机配置
    SERVO_PAN_PIN = 18
    SERVO_TILT_PIN = 27
    SERVO_FREQUENCY = 50
    SERVO_ANGLE_MIN = 0
    SERVO_ANGLE_MAX = 180
    SERVO_CENTER_ANGLE = 90
    MAX_ANGLE_SPEED = 30

    # PID控制参数
    PID_PAN_Kp = 0.2
    PID_PAN_Ki = 0.005
    PID_PAN_Kd = 0.03
    PID_TILT_Kp = 0.2
    PID_TILT_Ki = 0.005
    PID_TILT_Kd = 0.03

    # 控制参数
    ACCELERATION = 0.2
    DEAD_ZONE = 20

    # 图像中心点
    IMAGE_CENTER_X = CAMERA_RESOLUTION[0] // 2
    IMAGE_CENTER_Y = CAMERA_RESOLUTION[1] // 2

    # 日志配置
    LOG_LEVEL = "INFO"
    SAVE_DATASET = True
    DATASET_DIR = "person_dataset"