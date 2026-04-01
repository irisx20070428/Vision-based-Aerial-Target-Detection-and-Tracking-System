# config.py - 性能优化版
import os


class Config:
    """系统配置 - 性能优化版"""

    # 摄像头配置 - 降低分辨率
    CAMERA_TYPE = "usb"
    CAMERA_RESOLUTION = (320, 240)  # 从640x480降低到320x240，速度提升4倍
    CAMERA_FPS = 15  # 降低帧率，从30降到15

    # YOLO配置 - 使用轻量级模型
    YOLO_CONF_THRESHOLD = 0.5
    YOLO_MODEL = 'yolov5n'  # 从yolov5s改为yolov5n，速度提升2-3倍
    YOLO_TARGET_CLASSES = [0]  # 0 = person
    YOLO_FRAME_SKIP = 3  # 从2增加到3，每3帧检测一次
    YOLO_IMAGE_SIZE = 320  # 检测时缩小图像

    # 特征追踪配置 - 简化特征
    SIMILARITY_THRESHOLD = 0.6
    TRACKING_HISTORY_LEN = 50  # 减少历史记录
    USE_SIMPLIFIED_FEATURES = True  # 使用简化特征
    FEATURE_EXTRACT_INTERVAL = 3  # 每3帧提取一次特征

    # 舵机配置
    SERVO_PAN_PIN = 18
    SERVO_TILT_PIN = 27
    SERVO_FREQUENCY = 50
    SERVO_ANGLE_MIN = 0
    SERVO_ANGLE_MAX = 180
    SERVO_CENTER_ANGLE = 90
    MAX_ANGLE_SPEED = 20  # 降低速度，减少计算

    # PID控制参数 - 适当调整
    PID_PAN_Kp = 0.2  # 降低Kp，减少抖动
    PID_PAN_Ki = 0.005
    PID_PAN_Kd = 0.03

    PID_TILT_Kp = 0.2
    PID_TILT_Ki = 0.005
    PID_TILT_Kd = 0.03

    # 控制参数
    ACCELERATION = 0.2  # 降低加速度
    DEAD_ZONE = 20  # 增加死区

    # 图像中心点
    IMAGE_CENTER_X = CAMERA_RESOLUTION[0] // 2
    IMAGE_CENTER_Y = CAMERA_RESOLUTION[1] // 2

    # 日志配置 - 减少输出
    LOG_LEVEL = "ERROR"
    SAVE_DATASET = False  # 关闭数据集保存，提高性能
    DATASET_DIR = "person_dataset"