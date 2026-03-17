# main.py (完整版 - 集成特征追踪)
import cv2
import time
from datetime import datetime
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera import MacCameraPreview
from yolo_detector import YOLOPersonDetector
from analyze_dataset import analyze_dataset  # 导入数据集分析工具


class PersonDetectionApp:
    """人物检测应用程序 - 集成特征追踪"""

    def __init__(self, conf_threshold=0.5):
        """初始化应用"""
        print("=" * 70)
        print("🎯 实时人物检测系统 - 特征追踪版")
        print("=" * 70)

        self.camera = None
        self.detector = None
        self.frame = None  # 保存当前帧供鼠标回调使用

        try:
            # 1. 初始化摄像头
            print("\n📹 初始化摄像头...")
            self.camera = MacCameraPreview()

            # 2. 初始化YOLO检测器（已经集成了特征追踪）
            print("\n🤖 初始化YOLO检测器...")
            self.detector = YOLOPersonDetector(conf_threshold=conf_threshold)

            # 3. 窗口设置
            self.window_name = "Person Tracking System - Hover to highlight | Click to track | 'a' to analyze"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1000, 700)

            # ==== 设置鼠标回调 ====
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            # 4. 运行状态
            self.running = True
            self.show_detections = True
            self.detections = []  # 保存当前检测结果
            self.show_dataset_info = False  # 是否显示数据集信息
            self.last_analysis_time = 0

            print("\n" + "=" * 70)
            print("▶️ 系统启动成功！")
            print("=" * 70)
            print("🖱️ 鼠标操作:")
            print("   - 悬停: 人物框变黑色高亮")
            print("   - 左键点击: 开始追踪该人物")
            print("")
            print("⌨️ 键盘命令:")
            print("   - 'q' 或 ESC: 退出程序")
            print("   - 'c': 清除追踪选择")
            print("   - 's': 保存当前画面")
            print("   - 'a': 分析数据集统计信息")
            print("   - 'i': 显示/隐藏数据集信息")
            print("   - '+': 提高置信度阈值")
            print("   - '-': 降低置信度阈值")
            print("   - 'd': 切换检测框显示")
            print("=" * 70)

        except Exception as e:
            print(f"\n❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()
            sys.exit(1)

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
                print(f"📸 已保存人物图像到数据集")

    def draw_dataset_info(self, frame):
        """
        在画面上绘制数据集信息
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
        """运行主循环"""

        if not self.camera or not self.detector:
            print("❌ 系统未正确初始化")
            return

        print("\n开始检测...")

        try:
            while self.running:
                # 1. 获取摄像头画面
                frame = self.camera.get_frame()
                if frame is None:
                    print("❌ 无法获取摄像头画面")
                    time.sleep(0.1)
                    continue

                # 保存当前帧供鼠标回调使用
                self.frame = frame.copy()

                # 2. 执行YOLO检测
                self.detections, _ = self.detector.detect(frame)

                # 3. 绘制检测框
                if self.show_detections:
                    frame = self.detector.draw_detections(frame, self.detections)

                # 4. 获取检测摘要
                detection_summary = self.detector.get_detection_summary(self.detections)

                # 5. 添加模式信息
                if self.detector.is_selecting_mode:
                    mode_text = "[TRACKING MODE]"
                    # 获取追踪摘要
                    if hasattr(self.detector.smart_tracker, 'get_tracking_summary'):
                        tracking_summary = self.detector.smart_tracker.get_tracking_summary()
                        # 简化显示
                        history = self.detector.smart_tracker.tracking_history
                        if history:
                            angles = [h['angle'] for h in history[-10:]]
                            if angles:
                                from collections import Counter
                                angle_counts = Counter(angles)
                                main_angle = angle_counts.most_common(1)[0][0]
                                mode_text += f" [{main_angle}]"
                else:
                    mode_text = "[NORMAL MODE]"

                # 6. 构建信息文本
                info_text = f"{mode_text} {detection_summary} | conf: {self.detector.get_confidence_threshold():.2f}"

                # 7. 添加鼠标位置提示
                if self.detector.mouse_x >= 0 and self.detector.mouse_y >= 0:
                    mouse_text = f"Mouse: ({self.detector.mouse_x}, {self.detector.mouse_y})"
                    if self.detector.hovered_person_index >= 0:
                        mouse_text += f" - Person #{self.detector.hovered_person_index + 1}"

                    # 在画面顶部右侧显示
                    h, w = frame.shape[:2]
                    text_size = cv2.getTextSize(mouse_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.putText(frame, mouse_text, (w - text_size[0] - 10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # 8. 绘制摄像头信息
                display_frame = self.camera.draw_info(frame, info_text)

                # 9. 绘制数据集信息（如果开启）
                display_frame = self.draw_dataset_info(display_frame)

                # 10. 显示画面
                cv2.imshow(self.window_name, display_frame)

                # 11. 按键处理
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q 或 ESC
                    print("\n👋 退出程序")
                    self.running = False

                elif key == ord('c'):  # 清除选择
                    self.detector.clear_selection()
                    print("\n🔄 已清除追踪选择")

                elif key == ord('s'):  # 保存画面
                    self.save_frame(display_frame)

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

                # 每秒自动分析一次数据集（可选）
                current_time = time.time()
                if current_time - self.last_analysis_time > 30:  # 每30秒
                    if self.show_dataset_info:
                        self.last_analysis_time = current_time

        except KeyboardInterrupt:
            print("\n\n👋 用户中断")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def save_frame(self, frame):
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

            import json
            with open(f"session_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(stats, f, indent=2)

        print("✅ 系统已关闭")


# ========== 简化版快速启动 ==========

def quick_start():
    """快速启动函数 - 用于测试"""
    print("=" * 50)
    print("🚀 快速启动模式")
    print("=" * 50)

    # 使用默认配置
    app = PersonDetectionApp(conf_threshold=0.5)
    app.run()


# ========== 批量处理模式 ==========

def batch_process_mode(video_file=None):
    """
    批量处理模式 - 处理视频文件并分析
    """
    print("=" * 50)
    print("📼 批量处理模式")
    print("=" * 50)

    if video_file is None:
        print("请指定视频文件路径")
        return

    # 这里可以添加视频文件处理逻辑
    print(f"处理视频文件: {video_file}")
    # TODO: 实现视频文件处理


# ========== 主程序入口 ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='实时人物检测系统 - 特征追踪版')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值 (0-1)')
    parser.add_argument('--quick', action='store_true', help='快速启动模式')
    parser.add_argument('--video', type=str, help='处理视频文件路径')

    args = parser.parse_args()

    try:
        if args.video:
            batch_process_mode(args.video)
        elif args.quick:
            quick_start()
        else:
            app = PersonDetectionApp(conf_threshold=args.conf)
            app.run()

    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback

        traceback.print_exc()