# main.py
import cv2
import time
import sys
import os
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from camera import CameraManager
from servo_controller import ServoController
from pid_controller import PanTiltController
from yolo_detector import YOLOPersonDetector
from analyze_dataset import analyze_dataset


class PersonTrackingSystem:
    """人物追踪系统 - 完整版"""

    def __init__(self, conf_threshold=None):
        print("=" * 70)
        print("🎯 树莓派人物追踪系统 - 云台控制版")
        print("=" * 70)

        # 设置阈值
        if conf_threshold:
            Config.YOLO_CONF_THRESHOLD = conf_threshold

        self.camera = None
        self.detector = None
        self.servo = None
        self.pid_controller = None

        self.frame = None
        self.detections = []
        self.running = True
        self.show_detections = True
        self.show_dataset_info = False

        self._init_modules()

        # 性能监控
        self.performance = {
            'detect_time': [],
            'track_time': [],
            'fps': []
        }
        self.last_perf_time = time.time()

    def _init_modules(self):
        """初始化所有模块"""
        try:
            # 1. 摄像头
            print("\n📹 初始化摄像头...")
            self.camera = CameraManager()

            # 2. 舵机
            print("\n🤖 初始化舵机...")
            self.servo = ServoController()

            # 3. PID控制器
            print("\n🎛️ 初始化PID控制器...")
            self.pid_controller = PanTiltController()

            # 4. YOLO检测器
            print("\n🎯 初始化YOLO检测器...")
            self.detector = YOLOPersonDetector(
                conf_threshold=Config.YOLO_CONF_THRESHOLD
            )

            # 5. 窗口
            self.window_name = "Person Tracking System - Raspberry Pi"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            self._print_instructions()

        except Exception as e:
            print(f"\n❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()
            sys.exit(1)

    def _print_instructions(self):
        """打印操作说明"""
        print("\n" + "=" * 70)
        print("▶️ 系统启动成功！")
        print("=" * 70)
        print("🖱️ 鼠标操作:")
        print("   - 悬停: 人物框高亮")
        print("   - 左键点击: 开始追踪该人物")
        print("")
        print("⌨️ 键盘命令:")
        print("   - 'q': 退出程序")
        print("   - 'c': 清除追踪")
        print("   - 'r': 云台重置到中心")
        print("   - 's': 保存画面")
        print("   - 'a': 分析数据集")
        print("   - '+/-': 调整置信度阈值")
        print("   - 'd': 切换检测框显示")
        print("=" * 70)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        if self.detector is None:
            return

        self.detector.update_mouse_position(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            hovered_index = self.detector.hovered_person_index
            if hovered_index >= 0 and self.frame is not None:
                self.detector.select_hovered_person(
                    self.detections, hovered_index, self.frame
                )
                # 启用PID跟踪
                self.pid_controller.set_tracking_enabled(True)

    def run(self):
        """主循环"""
        print("\n开始检测...")

        last_servo_update = time.time()
        last_analysis_time = 0


        try:
            while self.running:
                # 1. 获取画面
                self.frame = self.camera.get_frame()
                if self.frame is None:
                    time.sleep(0.01)
                    continue

                # 2. 检测人物
                self.detections, _ = self.detector.detect(self.frame)

                # 3. 追踪模式：计算目标位置
                target_center = None
                target_bbox = None

                if self.detector.is_selecting_mode:
                    # 使用智能追踪器
                    best_match, score = self.detector.smart_tracker.track_in_new_frame(
                        self.detections, self.frame
                    )

                    if best_match:
                        x1, y1, x2, y2 = best_match['bbox']
                        target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        target_bbox = (x1, y1, x2, y2)

                        # 在画面上绘制追踪信息
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.circle(self.frame, target_center, 8, (0, 0, 255), -1)

                        info = f"Tracking | Score: {score:.2f}"
                        cv2.putText(self.frame, info, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 4. PID控制 - 计算云台角度
                if self.detector.is_selecting_mode and target_center:
                    # 计算目标角度
                    pan_angle, tilt_angle = self.pid_controller.compute_angles(
                        target_center[0], target_center[1]
                    )

                    # 设置舵机目标
                    self.servo.set_target(pan_angle, tilt_angle)

                # 5. 更新舵机位置（平滑移动）
                current_time = time.time()
                dt = current_time - last_servo_update
                if dt > 0.02:  # 50Hz更新
                    self.servo.update(dt)
                    last_servo_update = current_time

                # 6. 绘制检测框
                if self.show_detections:
                    self.frame = self.detector.draw_detections(self.frame, self.detections)

                # 7. 获取系统状态
                servo_angles = self.servo.get_current_angles()
                pid_status = self.pid_controller.get_status()

                # 8. 构建信息文本
                mode = "TRACKING" if self.detector.is_selecting_mode else "DETECTION"

                info_lines = [
                    f"Mode: {mode}",
                    f"Pan: {servo_angles[0]:.1f}° | Tilt: {servo_angles[1]:.1f}°",
                    f"Threshold: {self.detector.get_confidence_threshold():.2f}"
                ]

                if self.detector.is_selecting_mode and target_center:
                    error_x = Config.IMAGE_CENTER_X - target_center[0]
                    error_y = Config.IMAGE_CENTER_Y - target_center[1]
                    info_lines.append(f"Error: ({error_x}, {error_y}) px")

                info_text = "\n".join(info_lines)

                # 9. 绘制信息
                display_frame = self.camera.draw_info(self.frame, info_text)

                # 10. 显示画面
                cv2.imshow(self.window_name, display_frame)

                # 11. 按键处理
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    self.running = False
                elif key == ord('c'):
                    self.detector.clear_selection()
                    self.pid_controller.reset()
                    print("\n🔄 已清除追踪")
                elif key == ord('r'):
                    self.servo.reset_to_center()
                    self.pid_controller.reset()
                elif key == ord('s'):
                    self._save_frame(display_frame)
                elif key == ord('a'):
                    analyze_dataset()
                elif key == ord('+'):
                    new_conf = self.detector.get_confidence_threshold() + 0.05
                    self.detector.set_confidence_threshold(min(new_conf, 0.9))
                elif key == ord('-'):
                    new_conf = self.detector.get_confidence_threshold() - 0.05
                    self.detector.set_confidence_threshold(max(new_conf, 0.1))
                elif key == ord('d'):
                    self.show_detections = not self.show_detections

        except KeyboardInterrupt:
            print("\n👋 用户中断")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def _save_frame(self, frame):
        """保存画面"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n📸 已保存: {filename}")
        except Exception as e:
            print(f"\n❌ 保存失败: {e}")

    def cleanup(self):
        """清理资源"""
        if self.camera:
            self.camera.release()
        if self.servo:
            self.servo.cleanup()
        cv2.destroyAllWindows()
        print("✅ 系统已关闭")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='树莓派人物追踪系统')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    args = parser.parse_args()

    app = PersonTrackingSystem(conf_threshold=args.conf)
    app.run()