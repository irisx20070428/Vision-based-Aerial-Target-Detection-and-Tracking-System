# config.py - 确保追踪参数正确
import os


class Config:
    """系统配置 - 增强追踪版"""

    # 摄像头配置
    CAMERA_TYPE = "usb"
    CAMERA_RESOLUTION = (320, 240)
    CAMERA_FPS = 15

    # YOLO配置
    YOLO_CONF_THRESHOLD = 0.5
    YOLO_MODEL = 'yolov5n'
    YOLO_TARGET_CLASSES = [0]
    YOLO_FRAME_SKIP = 2
    YOLO_IMAGE_SIZE = 320

    # 舵机配置
    SERVO_PAN_PIN = 18
    SERVO_TILT_PIN = 27
    SERVO_FREQUENCY = 50
    SERVO_ANGLE_MIN = 0
    SERVO_ANGLE_MAX = 180
    SERVO_CENTER_ANGLE = 90
    MAX_ANGLE_SPEED = 30
    ACCELERATION = 0.3
    DEAD_ZONE = 20

    # PID控制参数
    PID_PAN_Kp = 0.2
    PID_PAN_Ki = 0.005
    PID_PAN_Kd = 0.02
    PID_TILT_Kp = 0.2
    PID_TILT_Ki = 0.005
    PID_TILT_Kd = 0.02

    # 图像中心点
    IMAGE_CENTER_X = CAMERA_RESOLUTION[0] // 2
    IMAGE_CENTER_Y = CAMERA_RESOLUTION[1] // 2

    # 特征追踪配置
    SIMILARITY_THRESHOLD = 0.45
    TRACKING_HISTORY_LEN = 1000

    # 日志配置
    LOG_LEVEL = "INFO"
    SAVE_DATASET = True
    DATASET_DIR = "person_dataset"