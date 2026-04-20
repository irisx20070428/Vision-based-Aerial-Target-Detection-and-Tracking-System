# main.py - 完整修复版（添加舵机指令显示）
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

        # 添加明确的追踪状态
        self.is_tracking = False
        self.tracked_person_center = None
        self.has_clicked = False

        # 添加舵机指令显示变量
        self.current_servo_command = "等待点击"  # 当前舵机指令
        self.last_pan_target = 90  # 上次水平目标角度
        self.last_tilt_target = 90  # 上次垂直目标角度

        # 降低 YOLO 检测频率的变量
        self.detect_skip = Config.YOLO_FRAME_SKIP
        self.detect_counter = 0
        self.last_detections = []

        # 初始化所有模块
        self._init_modules(conf_threshold)

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

                if self.pid_controller and self.servo:
                    pan_actual, tilt_actual = self.servo.get_current_angles()
                    self.pid_controller.current_pan = pan_actual
                    self.pid_controller.current_tilt = tilt_actual

                # 重要：初始化时禁用舵机追踪
                if hasattr(self.servo, 'enable_tracking'):
                    self.servo.enable_tracking(False)
                if hasattr(self.pid_controller, 'set_tracking_enabled'):
                    self.pid_controller.set_tracking_enabled(False)

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

    def _print_instructions(self):
        """打印操作说明"""
        print("\n" + "=" * 70)
        print("▶️ 系统启动成功！")
        print("=" * 70)
        print("🖱️ 鼠标操作:")
        print("   - 悬停: 人物框高亮显示")
        print("   - 左键点击: 开始追踪该人物（同时启用舵机）")
        print("")
        print("⌨️ 键盘命令:")
        print("   - 'q' 或 ESC: 退出程序")
        print("   - 'c': 清除追踪选择（停止舵机）")
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
            # 强制检测当前帧，获取最新检测结果
            if self.frame is not None:
                detections, _ = self.detector.detect(self.frame)
                # 使用最新检测结果寻找悬停人物
                hovered_index = self.detector.find_hovered_person(detections)
                if hovered_index >= 0:
                    self.detector.select_hovered_person(detections, hovered_index, self.frame)
                    # 启用舵机等后续操作（保持原有代码）
                    if self.servo and hasattr(self.servo, 'enable_tracking'):
                        self.servo.enable_tracking(True)
                        print("🎯 舵机追踪已启用")
                        self.current_servo_command = "舵机已启用"
                    if self.pid_controller and hasattr(self.pid_controller, 'set_tracking_enabled'):
                        pan_actual, tilt_actual = self.servo.get_current_angles()
                        self.pid_controller.set_tracking_enabled(True, initial_angles=(pan_actual, tilt_actual))
                    self.is_tracking = True
                    self.has_clicked = True
                    print(f"📸 已保存人物图像到数据集，开始追踪")
                    print(f"💡 提示：按 'c' 键可停止追踪")

    def draw_dataset_info(self, frame):
        """绘制数据集信息"""
        if not self.show_dataset_info or self.detector is None:
            return frame

        h, w = frame.shape[:2]
        panel_x, panel_y = w - 300, 100
        panel_w, panel_h = 280, 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 1)
        cv2.putText(frame, "📊 Dataset Info", (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        dataset = self.detector.smart_tracker.dataset
        y_offset = panel_y + 55
        total_samples = sum(len(samples) for samples in dataset.dataset.values())
        cv2.putText(frame, f"Total samples: {total_samples}", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        persons = len(dataset.dataset)
        cv2.putText(frame, f"Different persons: {persons}", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def draw_servo_command_panel(self, frame):
        """在屏幕一角绘制舵机指令面板"""
        h, w = frame.shape[:2]

        # 面板位置（右上角，在原有信息下方）
        panel_x = w - 280
        panel_y = 100
        panel_w = 270
        panel_h = 150

        # 创建半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制边框
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 255), 2)

        # 标题
        cv2.putText(frame, "Servo Command", (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 获取当前舵机状态
        if self.servo and hasattr(self.servo, 'tracking_enabled'):
            if self.servo.tracking_enabled:
                # 获取当前角度
                pan, tilt = self.servo.get_current_angles()
                # 获取目标角度
                if hasattr(self.servo, 'target_pan'):
                    target_pan = self.servo.target_pan
                    target_tilt = self.servo.target_tilt
                else:
                    target_pan = pan
                    target_tilt = tilt

                # 显示当前角度
                cv2.putText(frame, f"Current Pan: {pan:.1f}°", (panel_x + 10, panel_y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Current Tilt: {tilt:.1f}°", (panel_x + 10, panel_y + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 显示目标角度
                cv2.putText(frame, f"Target Pan: {target_pan:.1f}°", (panel_x + 10, panel_y + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Target Tilt: {target_tilt:.1f}°", (panel_x + 10, panel_y + 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 显示误差
                error_pan = target_pan - pan
                error_tilt = target_tilt - tilt
                cv2.putText(frame, f"Error: ({error_pan:+.1f}, {error_tilt:+.1f})", (panel_x + 10, panel_y + 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            else:
                # 舵机未启用
                cv2.putText(frame, "STAND BY", (panel_x + 10, panel_y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(frame, "Click on person to start", (panel_x + 10, panel_y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # 无舵机
            cv2.putText(frame, "No Servo", (panel_x + 10, panel_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

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
                if hasattr(self.detector, 'is_selecting_mode') and self.detector.is_selecting_mode:
                    if hasattr(self.detector, 'selected_person') and self.detector.selected_person is not None:
                        x1, y1, x2, y2, conf, _ = self.detector.selected_person
                        target_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        # 确保追踪标志正确
                        if not self.is_tracking and self.has_clicked:
                            self.is_tracking = True
                else:
                    # 没有选中人物时，确保舵机不追踪
                    if self.is_tracking:
                        self.is_tracking = False
                        if self.servo and hasattr(self.servo, 'enable_tracking'):
                            self.servo.enable_tracking(False)
                        if self.pid_controller and hasattr(self.pid_controller, 'set_tracking_enabled'):
                            self.pid_controller.set_tracking_enabled(False)
                        self.current_servo_command = "追踪已停止"
                        print("⏸️ 舵机追踪已暂停（未选中人物）")

                # 5. PID 控制（只在追踪模式下且舵机可用时）
                if self.pid_controller and target_center and self.is_tracking:
                    # 计算目标角度
                    pan_angle, tilt_angle = self.pid_controller.compute_angles(target_center[0], target_center[1])

                    # 更新舵机指令显示
                    if abs(pan_angle - self.last_pan_target) > 0.5 or abs(tilt_angle - self.last_tilt_target) > 0.5:
                        self.last_pan_target = pan_angle
                        self.last_tilt_target = tilt_angle
                        self.current_servo_command = f"Pan:{pan_angle:.1f}° Tilt:{tilt_angle:.1f}°"

                    # 设置舵机目标（舵机内部会检查 tracking_enabled）
                    if self.servo and hasattr(self.servo, 'set_target'):
                        self.servo.set_target(pan_angle, tilt_angle)

                # 6. 更新舵机位置（平滑移动）
                if self.servo and hasattr(self.servo, 'update'):
                    now = time.time()
                    # 更新频率10Hz
                    if now - self.last_servo_update >= 0.1:
                        self.servo.update(0.1)
                        self.last_servo_update = now

                # 7. 构建信息文本
                detection_summary = self.detector.get_detection_summary(self.detections)

                # 显示追踪状态
                if self.is_tracking and self.servo and hasattr(self.servo,
                                                               'tracking_enabled') and self.servo.tracking_enabled:
                    mode_text = "[TRACKING ACTIVE]"
                elif self.has_clicked:
                    mode_text = "[TRACKING SELECTED - WAITING]"
                else:
                    mode_text = "[NORMAL MODE]"

                info_lines = [
                    f"{mode_text} {detection_summary}",
                    f"conf: {self.detector.get_confidence_threshold():.2f}"
                ]

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
                        mouse_text += f" - Click to track"
                    h, w = display_frame.shape[:2]
                    text_size = cv2.getTextSize(mouse_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.putText(display_frame, mouse_text, (w - text_size[0] - 10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # 9. 添加启动提示
                if not self.has_clicked and self.servo:
                    cv2.putText(display_frame, "Click on a person to start tracking",
                                (10, display_frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 10. 摄像头信息绘制
                display_frame = self.camera.draw_info(display_frame, info_text, show_fps=False)
                display_frame = self.draw_dataset_info(display_frame)

                # 11. 绘制舵机指令面板（新增）
                # display_frame = self.draw_servo_command_panel(display_frame)

                # 12. 显示画面
                cv2.imshow(self.window_name, display_frame)

                time.sleep(0.005)  # 每帧休眠 5ms，释放 CPU

                # 13. 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                elif key == ord('c'):
                    # 清除追踪选择
                    self.detector.clear_selection()
                    self.is_tracking = False
                    self.has_clicked = False
                    self.current_servo_command = "追踪已清除"
                    if self.servo and hasattr(self.servo, 'enable_tracking'):
                        self.servo.enable_tracking(False)
                    if self.pid_controller:
                        if hasattr(self.pid_controller, 'reset'):
                            self.pid_controller.reset()
                        if hasattr(self.pid_controller, 'set_tracking_enabled'):
                            self.pid_controller.set_tracking_enabled(False)
                    print("\n🔄 已清除追踪选择，舵机追踪已停止")
                elif key == ord('r') and self.servo:
                    if hasattr(self.servo, 'tracking_enabled') and self.servo.tracking_enabled:
                        if hasattr(self.servo, 'reset_to_center'):
                            self.servo.reset_to_center()
                        if self.pid_controller and hasattr(self.pid_controller, 'reset'):
                            self.pid_controller.reset()
                        self.current_servo_command = "重置到中心"
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
                    if hasattr(self.detector, 'get_tracking_summary'):
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

    def _save_frame(self, frame):
        """保存当前画面"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "tracking" if self.is_tracking else "normal"
            filename = f"detection_{mode}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n📸 画面已保存: {filename}")
        except Exception as e:
            print(f"\n❌ 保存失败: {e}")

    def cleanup(self):
        print("\n正在关闭系统...")

        # 在释放舵机前，将舵机移动到初始位置（或保持当前位置）
        if self.servo:
            try:
                # 直接跳转到初始角度
                self.servo.set_angle_immediate(Config.SERVO_PAN_INIT_ANGLE, Config.SERVO_TILT_INIT_ANGLE)
                print(
                    f"🔄 舵机已设置为初始位置: pan={Config.SERVO_PAN_INIT_ANGLE}°, tilt={Config.SERVO_TILT_INIT_ANGLE}°")
                time.sleep(0.3)  # 等待舵机实际转动（物理时间）
            except Exception as e:
                print(f"⚠️ 设置舵机初始位置时出错: {e}")
            # 清理舵机资源
            if hasattr(self.servo, 'cleanup'):
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