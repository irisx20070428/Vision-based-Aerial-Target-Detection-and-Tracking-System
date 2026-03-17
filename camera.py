# camera.py (调整显示布局，为鼠标提示留出空间)
import cv2
import time
from datetime import datetime


class MacCameraPreview:
    """Mac 摄像头预览 - 显示FPS和时间"""

    def __init__(self, camera_id=0, width=640, height=480):
        """初始化摄像头"""
        print("=" * 50)
        print("📹 Mac 摄像头实时预览")
        print("=" * 50)

        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            for i in range(1, 5):
                print(f"尝试摄像头 ID {i}...")
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"✅ 找到摄像头 ID {i}")
                    break

            if not self.cap.isOpened():
                raise Exception(f"❌ 无法打开任何摄像头！")

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 获取实际分辨率
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n✅ 摄像头初始化成功")
        print(f"   - 分辨率: {self.actual_width} x {self.actual_height}")

        # FPS计算变量
        self.prev_frame_time = 0
        self.fps = 0

        print("\n▶️ 摄像头已就绪")
        print("=" * 50)

    def get_frame(self):
        """获取一帧画面"""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def calculate_fps(self):
        """计算实时FPS"""
        curr_time = time.time()
        if self.prev_frame_time != 0:
            time_diff = curr_time - self.prev_frame_time
            self.fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_frame_time = curr_time
        return self.fps

    def draw_info(self, frame, extra_info=None):
        """在画面上绘制FPS和时间"""
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # 计算FPS
        current_fps = self.calculate_fps()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 创建半透明黑色背景条 (顶部 - 增加高度为80)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 显示FPS
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示分辨率
        cv2.putText(frame, f"{w}x{h}", (100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 显示时间
        time_size = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.putText(frame, current_time, (w - time_size[0] - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 显示额外信息
        if extra_info:
            cv2.putText(frame, extra_info, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 图像中心十字线
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

        # 底部退出提示
        cv2.putText(frame, "Mouse: hover to highlight | click to select | 'c' clear | 'q' quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return frame

    def release(self):
        """释放摄像头资源"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 摄像头已关闭")