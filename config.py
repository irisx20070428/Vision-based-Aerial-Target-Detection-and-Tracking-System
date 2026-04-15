# config.py - 确保追踪参数正确
import os


class Config:
    """系统配置 - 增强追踪版"""

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

    # 舵机配置 - 降低速度，增加稳定性
    SERVO_PAN_PIN = 18
    SERVO_TILT_PIN = 27
    SERVO_FREQUENCY = 50
    SERVO_ANGLE_MIN = 0
    SERVO_ANGLE_MAX = 180
    SERVO_CENTER_ANGLE = 90
    MAX_ANGLE_SPEED = 30  # 降低到30度/秒（原60）
    ACCELERATION = 0.05  # 降低加速度
    DEAD_ZONE = 50  # 增加死区到50像素

    # PID控制参数 - 大幅降低增益
    PID_PAN_Kp = 0.08  # 大幅降低（原0.15）
    PID_PAN_Ki = 0.001  # 大幅降低（原0.003）
    PID_PAN_Kd = 0.005  # 大幅降低（原0.01）
    PID_TILT_Kp = 0.08  # 大幅降低
    PID_TILT_Ki = 0.001  # 大幅降低
    PID_TILT_Kd = 0.005  # 大幅降低

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