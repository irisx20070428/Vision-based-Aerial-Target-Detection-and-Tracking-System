# camera.py - 保持你原来的逻辑
import cv2
import time
import numpy as np
from datetime import datetime
import threading
from config import Config


class CameraManager:
    """摄像头管理器 - 支持树莓派Picamera2和USB摄像头"""

    def __init__(self, camera_type=None):
        self.camera_type = camera_type or Config.CAMERA_TYPE
        self.width = Config.CAMERA_RESOLUTION[0]
        self.height = Config.CAMERA_RESOLUTION[1]
        self.fps_target = Config.CAMERA_FPS

        self.cap = None
        self.picam2 = None
        self.frame = None
        self.running = False
        self.frame_lock = threading.Lock()

        # FPS计算
        self.prev_frame_time = 0
        self.fps = 0

        self._init_camera()

    def _init_camera(self):
        """初始化摄像头"""
        print("=" * 50)
        print("📹 摄像头初始化")
        print("=" * 50)

        if self.camera_type == "picamera2":
            self._init_picamera2()
        else:
            self._init_usb_camera()

    def _init_picamera2(self):
        """初始化树莓派Picamera2摄像头"""
        try:
            from picamera2 import Picamera2
            import libcamera

            self.picam2 = Picamera2()

            # 配置摄像头
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.fps_target}
            )
            self.picam2.configure(config)

            # 启动摄像头
            self.picam2.start()
            time.sleep(2)  # 等待摄像头稳定

            print(f"✅ Picamera2初始化成功")
            print(f"   - 分辨率: {self.width} x {self.height}")
            print(f"   - 帧率: {self.fps_target} FPS")

            # 启动采集线程
            self.running = True
            self.capture_thread = threading.Thread(target=self._picamera2_capture_loop)
            self.capture_thread.start()

        except Exception as e:
            print(f"❌ Picamera2初始化失败: {e}")
            print("   尝试使用USB摄像头...")
            self.camera_type = "usb"
            self._init_usb_camera()

    def _init_usb_camera(self):
        """初始化USB摄像头 - 树莓派优化（指定 V4L2 后端）"""
        import platform
        if platform.system() == 'Linux':
            # 尝试多个设备索引
            for i in range(3):
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    self.cap = cap
                    print(f"✅ 使用摄像头 ID {i}")
                    break
            else:
                raise Exception("无法打开USB摄像头！请检查连接和权限。\n"
                                "尝试运行: sudo chmod 666 /dev/video*")
        else:
            # Windows 或其他系统使用默认后端
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开USB摄像头！")

        # 设置分辨率、帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"✅ USB摄像头初始化成功")
        print(f"   - 分辨率: {actual_width} x {actual_height}")

        self.running = True
        self.capture_thread = threading.Thread(target=self._usb_camera_capture_loop)
        self.capture_thread.start()

    def _picamera2_capture_loop(self):
        """Picamera2采集循环"""
        frame_time = 1.0 / self.fps_target
        while self.running:
            try:
                frame_rgb = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                with self.frame_lock:
                    self.frame = frame_bgr
                time.sleep(frame_time)
            except Exception as e:
                print(f"采集错误: {e}")

    def _usb_camera_capture_loop(self):
        """USB摄像头采集循环（带帧率控制）"""
        frame_time = 1.0 / self.fps_target if self.fps_target > 0 else 0.033
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame
            time.sleep(frame_time)  # 控制采集频率

    def get_frame(self):
        """获取最新帧"""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def calculate_fps(self):
        """计算FPS"""
        curr_time = time.time()
        if self.prev_frame_time != 0:
            time_diff = curr_time - self.prev_frame_time
            self.fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_frame_time = curr_time
        return self.fps

    def draw_info(self, frame, extra_info=None):
        """在画面上绘制信息"""
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # 计算FPS
        current_fps = self.calculate_fps()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 创建半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # 显示FPS
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示分辨率
        cv2.putText(frame, f"{w}x{h}", (120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示时间
        time_size = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.putText(frame, current_time, (w - time_size[0] - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 显示额外信息
        if extra_info:
            # 分行显示
            lines = extra_info.split('\n')
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, 55 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 图像中心十字线
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        # 底部提示
        cv2.putText(frame, "Hover: highlight | Click: track | c: clear | +/-: threshold | q: quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return frame

    def release(self):
        """释放资源"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)

        if self.picam2:
            self.picam2.stop()
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        print("✅ 摄像头已关闭")