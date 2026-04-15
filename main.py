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

# main.py - 修复版（移除重复的异常处理）
import cv2
import time
import sys
import os
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from camera import CameraManager
from yolo_detector import YOLOPersonDetector
from analyze_dataset import analyze_dataset

# 舵机相关（可选）
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
    def _init_modules(self, conf_threshold):
        """初始化摄像头、检测器、舵机等"""
        try:
            # 1. 摄像头
            print("\n📹 初始化摄像头...")
            self.camera = CameraManager()

            # 2. YOLO 检测器
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

                # 重要：初始化时禁用舵机追踪
                self.servo.enable_tracking(False)  # 舵机禁用追踪
                self.pid_controller.set_tracking_enabled(False)  # PID禁用

                print("✅ 舵机已初始化并回正，等待点击选择人物开始追踪")
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

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数 - 点击开始追踪"""
        if self.detector is None:
            return

        self.detector.update_mouse_position(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            hovered_index = self.detector.hovered_person_index
            if hovered_index >= 0 and self.frame is not None:
                # 选择人物开始追踪
                self.detector.select_hovered_person(self.detections, hovered_index, self.frame)

                # 重要：启用舵机追踪（只有点击后才启用）
                if self.servo:
                    self.servo.enable_tracking(True)
                    print("🎯 舵机追踪已启用")

                if self.pid_controller:
                    self.pid_controller.set_tracking_enabled(True)

                # 设置追踪标志
                self.is_tracking = True

                print(f"📸 已保存人物图像到数据集，开始追踪")
                print(f"💡 提示：按 'c' 键可停止追踪")

    def run(self):
        """主循环"""
        frame_count = 0
        start_time = time.time()

        if not self.camera or not self.detector:
            print("❌ 系统未正确初始化")
            return

        print("\n开始检测...")
        print("💡 提示：鼠标悬停人物框，点击左键开始追踪")

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

                # 3. 绘制检测框
                display_frame = self.frame.copy()
                if self.show_detections:
                    display_frame = self.detector.draw_detections(display_frame, self.detections)

                # 4. 获取追踪目标（只有在追踪模式下才获取）
                target_center = None
                if self.detector.is_selecting_mode and self.detector.selected_person is not None:
                    # 获取当前追踪的人物位置
                    x1, y1, x2, y2, conf, _ = self.detector.selected_person
                    target_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # 确保追踪标志正确
                    if not self.is_tracking:
                        self.is_tracking = True
                else:
                    # 没有选中人物时，确保舵机不追踪
                    if self.is_tracking:
                        self.is_tracking = False
                        if self.servo:
                            self.servo.enable_tracking(False)
                        if self.pid_controller:
                            self.pid_controller.set_tracking_enabled(False)
                        print("⏸️ 舵机追踪已暂停（未选中人物）")

                # 5. PID 控制（只在追踪模式下且舵机可用时）
                if self.pid_controller and target_center and self.is_tracking:
                    # 计算目标角度
                    pan_angle, tilt_angle = self.pid_controller.compute_angles(target_center[0], target_center[1])

                    # 设置舵机目标（舵机内部会检查 tracking_enabled）
                    if self.servo:
                        self.servo.set_target(pan_angle, tilt_angle)

                # 6. 更新舵机位置（平滑移动）
                if self.servo:
                    now = time.time()
                    dt = now - self.last_servo_update
                    if dt > 0.02:  # 50Hz
                        self.servo.update(dt)
                        self.last_servo_update = now

                # 7. 构建信息文本
                detection_summary = self.detector.get_detection_summary(self.detections)

                # 显示追踪状态
                if self.is_tracking and self.servo and self.servo.tracking_enabled:
                    mode_text = "[TRACKING ACTIVE]"
                elif self.detector.is_selecting_mode:
                    mode_text = "[TRACKING SELECTED - WAITING]"
                else:
                    mode_text = "[NORMAL MODE]"

                info_lines = [
                    f"{mode_text} {detection_summary}",
                    f"conf: {self.detector.get_confidence_threshold():.2f}"
                ]

                if self.servo:
                    if self.servo.tracking_enabled:
                        pan, tilt = self.servo.get_current_angles()
                        info_lines.append(f"Pan: {pan:.1f}° Tilt: {tilt:.1f}° [TRACKING]")
                    else:
                        info_lines.append(f"Servo: [STAND BY]")

                if target_center and self.is_tracking:
                    h, w = self.frame.shape[:2]
                    err_x = target_center[0] - w // 2
                    err_y = target_center[1] - h // 2
                    info_lines.append(f"Error: ({err_x:+d}, {err_y:+d}) px")

                info_text = "\n".join(info_lines)

                # 8. 绘制鼠标位置提示
                if self.detector.mouse_x >= 0 and self.detector.mouse_y >= 0:
                    mouse_text = f"Mouse: ({self.detector.mouse_x}, {self.detector.mouse_y})"
                    if self.detector.hovered_person_index >= 0:
                        mouse_text += f" - Person #{self.detector.hovered_person_index + 1} (Click to track)"
                    h, w = display_frame.shape[:2]
                    text_size = cv2.getTextSize(mouse_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.putText(display_frame, mouse_text, (w - text_size[0] - 10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # 9. 添加启动提示
                if not self.is_tracking and self.servo:
                    cv2.putText(display_frame, "Click on a person to start tracking",
                                (10, display_frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 10. 摄像头信息绘制
                display_frame = self.camera.draw_info(display_frame, info_text, show_fps=False)
                display_frame = self.draw_dataset_info(display_frame)

                # 11. 显示画面
                cv2.imshow(self.window_name, display_frame)

                # 12. 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                elif key == ord('c'):
                    # 清除追踪选择
                    self.detector.clear_selection()
                    self.is_tracking = False
                    if self.servo:
                        self.servo.enable_tracking(False)
                    if self.pid_controller:
                        self.pid_controller.reset()
                        self.pid_controller.set_tracking_enabled(False)
                    print("\n🔄 已清除追踪选择，舵机追踪已停止")
                elif key == ord('r') and self.servo:
                    if self.servo.tracking_enabled:
                        self.servo.reset_to_center()
                        if self.pid_controller:
                            self.pid_controller.reset()
                        print("\n🔄 云台已重置到中心")
                    else:
                        print("\n⚠️ 舵机未启用追踪，无需重置")
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
                    print(f"📊 性能: FPS={fps:.1f}, 检测人数={len(self.detections)}, 追踪={self.is_tracking}")

        except KeyboardInterrupt:
            print("\n\n👋 用户中断")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("\n正在关闭系统...")

        if self.camera:
            self.camera.release()

        if self.servo:
            # 先禁用追踪
            try:
                self.servo.enable_tracking(False)
                print("🔄 舵机追踪已禁用")
                # 舵机回中
                print("🔄 舵机正在回中...")
                self.servo.reset_to_center()
                time.sleep(0.5)
                self.servo.cleanup()
            except Exception as e:
                print(f"⚠️ 舵机关闭时出错: {e}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='树莓派人物追踪系统 - 整合版')
    parser.add_argument('--conf', type=float, default=None, help='置信度阈值 (0-1)')
    parser.add_argument('--no-servo', action='store_true', help='禁用舵机控制（调试模式）')
    args = parser.parse_args()

    app = PersonDetectionApp(conf_threshold=args.conf, no_servo=args.no_servo)
    app.run()