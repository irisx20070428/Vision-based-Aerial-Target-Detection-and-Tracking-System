import cv2
import time
import sys
import os
import argparse
import json
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

    def __init__(self, conf_threshold=None, no_servo=False):
        print("=" * 70)
        print("🎯 树莓派人物追踪系统 - 云台控制版")
        print("=" * 70)

        # 保存舵机禁用标志
        self.no_servo = no_servo

        if self.no_servo:
            print("⚠️ 调试模式: 舵机控制已禁用")
            print("   可以正常进行检测和追踪，但不会控制物理云台")

        # 设置阈值
        if conf_threshold:
            Config.YOLO_CONF_THRESHOLD = conf_threshold

        self.camera = None
        self.detector = None
        self.servo = None
        self.pid_controller = None

        self.frame = None  # 保存当前帧供鼠标回调使用
        self.detections = []
        self.running = True
        self.show_detections = True
        self.show_dataset_info = False  # 添加数据集信息显示

        # 性能监控
        self.last_analysis_time = 0

        self._init_modules()

    def _init_modules(self):
        """初始化所有模块"""
        try:
            # 1. 摄像头
            print("\n📹 初始化摄像头...")
            self.camera = CameraManager()

            # 2. 舵机（如果不禁用）
            if not self.no_servo:
                print("\n🤖 初始化舵机...")
                self.servo = ServoController()

                # 3. PID控制器
                print("\n🎛️ 初始化PID控制器...")
                self.pid_controller = PanTiltController()
            else:
                print("\n⚠️ 舵机模块已禁用")
                self.servo = None
                self.pid_controller = None

            # 4. YOLO检测器（集成特征追踪）
            print("\n🎯 初始化YOLO检测器...")
            self.detector = YOLOPersonDetector(
                conf_threshold=Config.YOLO_CONF_THRESHOLD
            )

            # 5. 窗口设置
            self.window_name = "Person Tracking System - Raspberry Pi"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)

            # 设置鼠标回调
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
        print("   - 左键点击: 开始追踪该人物")
        print("")
        print("⌨️ 键盘命令:")
        print("   - 'q' 或 ESC: 退出程序")
        print("   - 'c': 清除追踪选择")
        if not self.no_servo:
            print("   - 'r': 云台重置到中心")
        print("   - 's': 保存当前画面")
        print("   - 'a': 分析数据集统计信息")
        print("   - 'i': 显示/隐藏数据集信息")
        print("   - '+': 提高置信度阈值")
        print("   - '-': 降低置信度阈值")
        print("   - 'd': 切换检测框显示")
        print("   - 'p': 打印追踪统计")
        print("=" * 70)

    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数 - 支持特征追踪

        参数:
            event: 鼠标事件类型
            x, y: 鼠标坐标
        """
        if self.detector is None:
            return

        # 更新鼠标位置到检测器
        self.detector.update_mouse_position(x, y)

        # 处理鼠标点击
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            hovered_index = self.detector.hovered_person_index
            if hovered_index >= 0 and self.frame is not None:
                # 传入当前帧用于裁剪人物图像
                self.detector.select_hovered_person(
                    self.detections,
                    hovered_index,
                    self.frame
                )
                # 启用PID跟踪（如果有舵机）
                if self.pid_controller:
                    self.pid_controller.set_tracking_enabled(True)
                print(f"📸 已保存人物图像到数据集，开始追踪")

    def draw_dataset_info(self, frame):
        """
        在画面上绘制数据集信息（与Mac版对齐）
        """
        if not self.show_dataset_info or self.detector is None:
            return frame

        h, w = frame.shape[:2]

        # 创建信息面板（右上角）
        panel_x = w - 300
        panel_y = 100
        panel_w = 280
        panel_h = 200

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制边框
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (255, 255, 255), 1)

        # 绘制标题
        cv2.putText(frame, "📊 Dataset Info",
                    (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 获取数据集信息
        dataset = self.detector.smart_tracker.dataset
        y_offset = panel_y + 55

        # 总样本数
        total_samples = sum(len(samples) for samples in dataset.dataset.values())
        cv2.putText(frame, f"Total samples: {total_samples}",
                    (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

        # 不同人物数
        persons = len(dataset.dataset)
        cv2.putText(frame, f"Different persons: {persons}",
                    (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

        # 角度统计
        if total_samples > 0:
            cv2.putText(frame, "Angle distribution:",
                        (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20

            # 统计角度
            from collections import Counter
            angles = []
            for samples in dataset.dataset.values():
                for sample in samples:
                    angle = sample['features']['face_angle'][0].value
                    angles.append(angle)

            angle_counts = Counter(angles)
            for angle, count in list(angle_counts.items())[:3]:  # 最多显示3个
                percentage = count / total_samples * 100
                cv2.putText(frame, f"  {angle}: {count} ({percentage:.0f}%)",
                            (panel_x + 10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 18

        return frame

    def run(self):
        """主循环 - 完整版"""
        print("\n开始检测...")

        last_servo_update = time.time()
        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                # 1. 获取摄像头画面
                self.frame = self.camera.get_frame()
                if self.frame is None:
                    time.sleep(0.01)
                    continue

                # 2. 执行YOLO检测
                self.detections, _ = self.detector.detect(self.frame)

                # 3. 追踪模式：获取目标位置
                target_center = None
                target_bbox = None
                tracking_score = 0

                if self.detector.is_selecting_mode:
                    # 使用智能追踪器在新帧中寻找目标
                    best_match, score = self.detector.smart_tracker.track_in_new_frame(
                        self.detections, self.frame
                    )
                    tracking_score = score

                    if best_match:
                        x1, y1, x2, y2 = best_match['bbox']
                        target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        target_bbox = (x1, y1, x2, y2)

                        # 更新追踪器中的选中人物信息
                        self.detector.selected_person = [x1, y1, x2, y2, best_match.get('conf', 0), 0]
                    else:
                        # 追踪丢失提示
                        if frame_count % 30 == 0:  # 每30帧提示一次
                            print("⚠️ 追踪丢失，请重新选择人物")

                # 4. PID控制 - 计算云台角度（如果有舵机）
                if self.pid_controller and self.detector.is_selecting_mode and target_center:
                    # 计算目标角度
                    pan_angle, tilt_angle = self.pid_controller.compute_angles(
                        target_center[0], target_center[1]
                    )

                    # 设置舵机目标
                    if self.servo:
                        self.servo.set_target(pan_angle, tilt_angle)

                # 5. 更新舵机位置（如果有舵机）
                if self.servo:
                    current_time = time.time()
                    dt = current_time - last_servo_update
                    if dt > 0.02:  # 50Hz更新
                        self.servo.update(dt)
                        last_servo_update = current_time

                # 6. 绘制检测框
                if self.show_detections:
                    self.frame = self.detector.draw_detections(self.frame, self.detections)

                # 7. 获取系统状态
                if self.servo:
                    servo_angles = self.servo.get_current_angles()
                else:
                    servo_angles = (0, 0)

                # 8. 构建信息文本
                mode = "TRACKING" if self.detector.is_selecting_mode else "DETECTION"

                # 获取检测摘要
                detection_summary = self.detector.get_detection_summary(self.detections)

                # 构建多行信息
                info_lines = [
                    f"Mode: {mode}",
                    f"{detection_summary}",
                    f"Threshold: {self.detector.get_confidence_threshold():.2f}"
                ]

                # 添加舵机信息（如果不禁用）
                if not self.no_servo:
                    info_lines.append(f"Pan: {servo_angles[0]:.1f}° | Tilt: {servo_angles[1]:.1f}°")

                # 添加追踪信息
                if self.detector.is_selecting_mode and self.detector.smart_tracker.tracked_person_id is not None:
                    tracked_id = self.detector.smart_tracker.tracked_person_id
                    info_lines.insert(2, f"Tracking Person #{tracked_id} | Score: {tracking_score:.2f}")

                # 添加误差信息
                if self.detector.is_selecting_mode and target_center:
                    error_x = Config.IMAGE_CENTER_X - target_center[0]
                    error_y = Config.IMAGE_CENTER_Y - target_center[1]
                    info_lines.append(f"Error: ({error_x:+d}, {error_y:+d}) px")

                info_text = "\n".join(info_lines)

                # 9. 绘制鼠标位置提示
                if self.detector.mouse_x >= 0 and self.detector.mouse_y >= 0:
                    mouse_text = f"Mouse: ({self.detector.mouse_x}, {self.detector.mouse_y})"
                    if self.detector.hovered_person_index >= 0:
                        mouse_text += f" - Person #{self.detector.hovered_person_index + 1}"

                    # 在画面顶部右侧显示
                    h, w = self.frame.shape[:2]
                    text_size = cv2.getTextSize(mouse_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.putText(self.frame, mouse_text, (w - text_size[0] - 10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # 10. 绘制摄像头信息
                display_frame = self.camera.draw_info(self.frame, info_text)

                # 11. 绘制数据集信息（如果开启）
                display_frame = self.draw_dataset_info(display_frame)

                # 12. 添加角度分布图（追踪模式下）
                if self.detector.is_selecting_mode and self.detector.smart_tracker.tracking_history:
                    display_frame = self.detector.visualizer.draw_angle_distribution(
                        display_frame,
                        self.detector.smart_tracker.tracking_history
                    )

                # 13. 显示画面
                cv2.imshow(self.window_name, display_frame)

                # 14. 按键处理
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q 或 ESC
                    print("\n👋 退出程序")
                    self.running = False

                elif key == ord('c'):  # 清除选择
                    self.detector.clear_selection()
                    if self.pid_controller:
                        self.pid_controller.reset()
                    print("\n🔄 已清除追踪选择")

                elif key == ord('r'):  # 云台重置（如果有舵机）
                    if self.servo:
                        self.servo.reset_to_center()
                        if self.pid_controller:
                            self.pid_controller.reset()
                        print("\n🔄 云台已重置到中心")
                    else:
                        print("\n⚠️ 舵机未启用，无法重置")

                elif key == ord('s'):  # 保存画面
                    self._save_frame(display_frame)

                elif key == ord('a'):  # 分析数据集
                    print("\n📊 正在分析数据集...")
                    analyze_dataset()

                elif key == ord('i'):  # 切换数据集信息显示
                    self.show_dataset_info = not self.show_dataset_info
                    print(f"\n📊 数据集信息显示: {'开启' if self.show_dataset_info else '关闭'}")

                elif key == ord('+') or key == ord('='):  # 提高置信度阈值
                    new_conf = self.detector.get_confidence_threshold() + 0.05
                    self.detector.set_confidence_threshold(min(new_conf, 0.9))

                elif key == ord('-') or key == ord('_'):  # 降低置信度阈值
                    new_conf = self.detector.get_confidence_threshold() - 0.05
                    self.detector.set_confidence_threshold(max(new_conf, 0.1))

                elif key == ord('d'):  # 切换检测框显示
                    self.show_detections = not self.show_detections
                    print(f"\n👁️ 检测框显示: {'开启' if self.show_detections else '关闭'}")

                elif key == ord('p'):  # 打印追踪统计
                    if self.detector.is_selecting_mode:
                        summary = self.detector.smart_tracker.get_tracking_summary()
                        print("\n" + summary)

                # 可选：每30秒自动分析一次数据集
                current_time = time.time()
                if current_time - self.last_analysis_time > 30:
                    if self.show_dataset_info:
                        self.last_analysis_time = current_time

                # 性能统计（每100帧打印一次）
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 性能统计: FPS={fps:.1f}, 检测人数={len(self.detections)}")

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
            mode = "tracking" if self.detector.is_selecting_mode else "normal"

            # 如果有追踪中的人物，添加到文件名
            if self.detector.is_selecting_mode and self.detector.smart_tracker.tracked_person_id is not None:
                person_id = self.detector.smart_tracker.tracked_person_id
                filename = f"detection_{mode}_person{person_id}_{timestamp}.jpg"
            else:
                filename = f"detection_{mode}_{timestamp}.jpg"

            cv2.imwrite(filename, frame)
            print(f"\n📸 画面已保存: {filename}")

            # 同时保存当前帧到数据集（可选）
            if self.detector.is_selecting_mode and self.frame is not None:
                dataset_frame = self.frame.copy()
                cv2.imwrite(f"dataset_frame_{timestamp}.jpg", dataset_frame)

        except Exception as e:
            print(f"\n❌ 保存失败: {e}")

    def cleanup(self):
        """清理资源"""
        if self.camera:
            self.camera.release()
        if self.servo:
            self.servo.cleanup()
        cv2.destroyAllWindows()

        # 打印最终数据集统计
        if self.detector and hasattr(self.detector, 'smart_tracker'):
            dataset = self.detector.smart_tracker.dataset
            total_samples = sum(len(samples) for samples in dataset.dataset.values())
            print(f"\n📊 会话结束 - 数据集总样本数: {total_samples}")

            # 保存会话统计
            stats = {
                'timestamp': datetime.now().isoformat(),
                'total_samples': total_samples,
                'persons_tracked': len(dataset.dataset)
            }

            try:
                with open(f"session_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                    json.dump(stats, f, indent=2)
            except:
                pass

        print("✅ 系统已关闭")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='树莓派人物追踪系统')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值 (0-1)')
    parser.add_argument('--no-servo', action='store_true', help='禁用舵机控制（调试模式）')
    args = parser.parse_args()

    # 显示运行模式
    if args.no_servo:
        print("⚠️ 调试模式: 舵机控制已禁用")
        print("   可以正常进行检测和追踪，但不会控制物理云台")
        print("   按 Enter 继续...")
        input()

    app = PersonTrackingSystem(conf_threshold=args.conf, no_servo=args.no_servo)
    app.run()