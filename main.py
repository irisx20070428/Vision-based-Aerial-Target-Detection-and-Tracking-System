# main.py - 整合版（支持树莓派摄像头 + 可选舵机，保留你的检测追踪逻辑）
import cv2
import time
import sys
import os
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from camera import CameraManager          # 使用你朋友的摄像头管理器
from yolo_detector import YOLOPersonDetector
from analyze_dataset import analyze_dataset

# 舵机相关（可选，如果 --no-servo 则不会导入）
SERVO_AVAILABLE = False
ServoController = None
PanTiltController = None
try:
    from servo_controller import ServoController
    from pid_controller import PanTiltController
    SERVO_AVAILABLE = True
except ImportError:
    print("⚠️ 舵机模块未找到，将以 --no-servo 模式运行")


class PersonDetectionApp:
    """人物检测应用程序 - 整合版（支持树莓派和舵机）"""

    def __init__(self, conf_threshold=None, no_servo=False):
        print("=" * 70)
        print("🎯 实时人物检测系统 - 整合版（树莓派优化）")
        print("=" * 70)

        self.no_servo = no_servo or not SERVO_AVAILABLE
        self.camera = None
        self.detector = None
        self.servo = None
        self.pid_controller = None
        self.frame = None
        self.detections = []
        self.running = True
        self.show_detections = True
        self.show_dataset_info = False

        # 性能监控
        self.last_analysis_time = 0
        self.last_servo_update = 0

        # 降低 YOLO 检测频率的变量
        self.detect_skip = Config.YOLO_FRAME_SKIP   # 从 config 读取
        self.detect_counter = 0
        self.last_detections = []

        # 初始化所有模块
        self._init_modules(conf_threshold)

    def _init_modules(self, conf_threshold):
        """初始化摄像头、检测器、舵机等"""
        try:
            # 1. 摄像头
            print("\n📹 初始化摄像头...")
            self.camera = CameraManager()   # 使用你朋友的 CameraManager

            # 2. YOLO 检测器（你的版本）
            print("\n🤖 初始化YOLO检测器...")
            if conf_threshold is None:
                conf_threshold = Config.YOLO_CONF_THRESHOLD
            self.detector = YOLOPersonDetector(conf_threshold=conf_threshold)

            # 3. 舵机和PID（如果未禁用）
            if not self.no_servo and SERVO_AVAILABLE:
                print("\n🤖 初始化舵机...")
                self.servo = ServoController()
                print("\n🎛️ 初始化PID控制器...")
                self.pid_controller = PanTiltController()
            else:
                print("\n⚠️ 舵机控制已禁用（--no-servo 或模块缺失）")
                self.servo = None
                self.pid_controller = None

            # 4. 窗口设置
            self.window_name = "Person Tracking System - Hover & Click"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1000, 700)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            self._print_instructions()

        except Exception as e:
            print(f"\n❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()
            sys.exit(1)

    def _print_instructions(self):
        print("\n" + "=" * 70)
        print("▶️ 系统启动成功！")
        print("=" * 70)
        print("🖱️ 鼠标操作:")
        print("   - 悬停: 人物框高亮显示")
        print("   - 左键点击: 开始追踪该人物")
        print("")
        print("⌨️ 键盘命令:")
        print("   - 'q' 或 ESC: 退出程序")
        print("   - 'c': 清除追踪选择")
        if self.servo:
            print("   - 'r': 云台重置到中心")
        print("   - 's': 保存当前画面")
        print("   - 'a': 分析数据集统计信息")
        print("   - 'i': 显示/隐藏数据集信息")
        print("   - 'p': 打印追踪统计摘要")
        print("   - '+': 提高置信度阈值")
        print("   - '-': 降低置信度阈值")
        print("   - 'd': 切换检测框显示")
        print("=" * 70)

    def mouse_callback(self, event, x, y, flags, param):
        if self.detector is None:
            return
        self.detector.update_mouse_position(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            hovered_index = self.detector.hovered_person_index
            if hovered_index >= 0 and self.frame is not None:
                self.detector.select_hovered_person(self.detections, hovered_index, self.frame)
                print(f"📸 已保存人物图像到数据集，开始追踪")
                # 如果启用了PID，则启用追踪
                if self.pid_controller:
                    self.pid_controller.set_tracking_enabled(True)

    def draw_dataset_info(self, frame):
        if not self.show_dataset_info or self.detector is None:
            return frame
        h, w = frame.shape[:2]
        panel_x, panel_y = w - 300, 100
        panel_w, panel_h = 280, 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (255,255,255), 1)
        cv2.putText(frame, "📊 Dataset Info", (panel_x+10, panel_y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        dataset = self.detector.smart_tracker.dataset
        y_offset = panel_y + 55
        total_samples = sum(len(samples) for samples in dataset.dataset.values())
        cv2.putText(frame, f"Total samples: {total_samples}", (panel_x+10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y_offset += 25
        persons = len(dataset.dataset)
        cv2.putText(frame, f"Different persons: {persons}", (panel_x+10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame

    def run(self):
        if not self.camera or not self.detector:
            print("❌ 系统未正确初始化")
            return

        print("\n开始检测...")
        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                # 1. 获取摄像头画面
                self.frame = self.camera.get_frame()
                if self.frame is None:
                    time.sleep(0.01)
                    continue

                # 2. 降低 YOLO 检测频率
                self.detect_counter += 1
                if self.detect_counter % self.detect_skip == 0:
                    self.detections, _ = self.detector.detect(self.frame)
                    self.last_detections = self.detections
                else:
                    self.detections = self.last_detections

                # 3. 绘制检测框（你的绘制逻辑）
                display_frame = self.frame.copy()
                if self.show_detections:
                    display_frame = self.detector.draw_detections(display_frame, self.detections)

                # 4. 追踪模式下获取目标中心点（用于舵机控制）
                target_center = None
                if self.detector.is_selecting_mode and self.detector.selected_person is not None:
                    x1, y1, x2, y2, conf, _ = self.detector.selected_person
                    target_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # 5. PID 控制（如果启用舵机）
                if self.pid_controller and target_center:
                    pan_angle, tilt_angle = self.pid_controller.compute_angles(target_center[0], target_center[1])
                    if self.servo:
                        self.servo.set_target(pan_angle, tilt_angle)

                # 6. 更新舵机位置（平滑移动）
                if self.servo:
                    now = time.time()
                    dt = now - self.last_servo_update
                    if dt > 0.02:   # 50Hz
                        self.servo.update(dt)
                        self.last_servo_update = now

                # 7. 构建信息文本
                detection_summary = self.detector.get_detection_summary(self.detections)
                mode_text = "[TRACKING MODE]" if self.detector.is_selecting_mode else "[NORMAL MODE]"
                info_lines = [
                    f"{mode_text} {detection_summary}",
                    f"conf: {self.detector.get_confidence_threshold():.2f}"
                ]
                if self.servo:
                    pan, tilt = self.servo.get_current_angles()
                    info_lines.append(f"Pan: {pan:.1f}° Tilt: {tilt:.1f}°")
                if target_center:
                    h, w = self.frame.shape[:2]
                    err_x = target_center[0] - w//2
                    err_y = target_center[1] - h//2
                    info_lines.append(f"Error: ({err_x:+d}, {err_y:+d}) px")
                info_text = "\n".join(info_lines)

                # 8. 绘制鼠标位置提示
                if self.detector.mouse_x >= 0 and self.detector.mouse_y >= 0:
                    mouse_text = f"Mouse: ({self.detector.mouse_x}, {self.detector.mouse_y})"
                    if self.detector.hovered_person_index >= 0:
                        mouse_text += f" - Person #{self.detector.hovered_person_index + 1}"
                    h, w = display_frame.shape[:2]
                    text_size = cv2.getTextSize(mouse_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.putText(display_frame, mouse_text, (w - text_size[0] - 10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # 9. 摄像头信息绘制（使用 CameraManager 的 draw_info）
                display_frame = self.camera.draw_info(display_frame, info_text)
                display_frame = self.draw_dataset_info(display_frame)

                # 10. 显示画面
                cv2.imshow(self.window_name, display_frame)

                # 11. 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                elif key == ord('c'):
                    self.detector.clear_selection()
                    if self.pid_controller:
                        self.pid_controller.reset()
                    print("\n🔄 已清除追踪选择")
                elif key == ord('r') and self.servo:
                    self.servo.reset_to_center()
                    if self.pid_controller:
                        self.pid_controller.reset()
                    print("\n🔄 云台已重置到中心")
                elif key == ord('s'):
                    self._save_frame(display_frame)
                elif key == ord('a'):
                    print("\n📊 正在分析数据集...")
                    analyze_dataset()
                elif key == ord('i'):
                    self.show_dataset_info = not self.show_dataset_info
                    print(f"\n📊 数据集信息显示: {'开启' if self.show_dataset_info else '关闭'}")
                elif key == ord('p'):
                    if self.detector.is_selecting_mode:
                        summary = self.detector.get_tracking_summary()
                        print("\n" + summary)
                elif key == ord('+') or key == ord('='):
                    new_conf = self.detector.get_confidence_threshold() + 0.05
                    self.detector.set_confidence_threshold(min(new_conf, 0.9))
                elif key == ord('-') or key == ord('_'):
                    new_conf = self.detector.get_confidence_threshold() - 0.05
                    self.detector.set_confidence_threshold(max(new_conf, 0.1))
                elif key == ord('d'):
                    self.show_detections = not self.show_detections
                    print(f"\n👁️ 检测框显示: {'开启' if self.show_detections else '关闭'}")

                # 性能统计
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 性能: FPS={fps:.1f}, 检测人数={len(self.detections)}")

        except KeyboardInterrupt:
            print("\n\n👋 用户中断")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def _save_frame(self, frame):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "tracking" if self.detector.is_selecting_mode else "normal"
            filename = f"detection_{mode}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n📸 画面已保存: {filename}")
        except Exception as e:
            print(f"\n❌ 保存失败: {e}")

    def cleanup(self):
        if self.camera:
            self.camera.release()
        if self.servo:
            self.servo.cleanup()
        cv2.destroyAllWindows()
        if self.detector and hasattr(self.detector, 'smart_tracker'):
            dataset = self.detector.smart_tracker.dataset
            total_samples = sum(len(samples) for samples in dataset.dataset.values())
            print(f"\n📊 会话结束 - 数据集总样本数: {total_samples}")
        print("✅ 系统已关闭")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='树莓派人物追踪系统 - 整合版')
    parser.add_argument('--conf', type=float, default=None, help='置信度阈值 (0-1)')
    parser.add_argument('--no-servo', action='store_true', help='禁用舵机控制（调试模式）')
    args = parser.parse_args()

    app = PersonDetectionApp(conf_threshold=args.conf, no_servo=args.no_servo)
    app.run()