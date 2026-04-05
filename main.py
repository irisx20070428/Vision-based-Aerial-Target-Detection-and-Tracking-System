# main.py (最终版 - 包含降低 YOLO 检测频率优化)
import cv2
import time
from datetime import datetime
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera import MacCameraPreview
from yolo_detector import YOLOPersonDetector
from analyze_dataset import analyze_dataset


class PersonDetectionApp:
    """人物检测应用程序 - 特征追踪版"""

    def __init__(self, conf_threshold=0.5):
        print("=" * 70)
        print("🎯 实时人物检测系统 - 特征追踪版")
        print("=" * 70)

        self.camera = None
        self.detector = None
        self.frame = None

        try:
            print("\n📹 初始化摄像头...")
            self.camera = MacCameraPreview()

            print("\n🤖 初始化YOLO检测器...")
            self.detector = YOLOPersonDetector(conf_threshold=conf_threshold)

            self.window_name = "Person Tracking System - Hover to highlight | Click to track | 'a' to analyze"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1000, 700)

            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            self.running = True
            self.show_detections = True
            self.detections = []
            self.show_dataset_info = False
            self.last_analysis_time = 0

            # 降低 YOLO 检测频率的变量
            self.detect_skip = 2          # 每3帧检测一次
            self.detect_counter = 0
            self.last_detections = []

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
        if self.detector is None:
            return
        self.detector.update_mouse_position(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            hovered_index = self.detector.hovered_person_index
            if hovered_index >= 0 and self.frame is not None:
                self.detector.select_hovered_person(self.detections, hovered_index, self.frame)
                print(f"📸 已保存人物图像到数据集")

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

        try:
            while self.running:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                self.frame = frame.copy()

                # ---- 降低 YOLO 检测频率 ----
                self.detect_counter += 1
                if self.detect_counter % self.detect_skip == 0:
                    self.detections, _ = self.detector.detect(frame)
                    self.last_detections = self.detections
                else:
                    self.detections = self.last_detections

                if self.show_detections:
                    frame = self.detector.draw_detections(frame, self.detections)

                detection_summary = self.detector.get_detection_summary(self.detections)

                if self.detector.is_selecting_mode:
                    mode_text = "[TRACKING MODE]"
                else:
                    mode_text = "[NORMAL MODE]"

                info_text = f"{mode_text} {detection_summary} | conf: {self.detector.get_confidence_threshold():.2f}"

                if self.detector.mouse_x >= 0 and self.detector.mouse_y >= 0:
                    mouse_text = f"Mouse: ({self.detector.mouse_x}, {self.detector.mouse_y})"
                    if self.detector.hovered_person_index >= 0:
                        mouse_text += f" - Person #{self.detector.hovered_person_index + 1}"
                    h, w = frame.shape[:2]
                    text_size = cv2.getTextSize(mouse_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.putText(frame, mouse_text, (w - text_size[0] - 10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                display_frame = self.camera.draw_info(frame, info_text)
                display_frame = self.draw_dataset_info(display_frame)

                cv2.imshow(self.window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                elif key == ord('c'):
                    self.detector.clear_selection()
                    print("\n🔄 已清除追踪选择")
                elif key == ord('s'):
                    self.save_frame(display_frame)
                elif key == ord('a'):
                    print("\n📊 正在分析数据集...")
                    analyze_dataset()
                elif key == ord('i'):
                    self.show_dataset_info = not self.show_dataset_info
                    print(f"\n📊 数据集信息显示: {'开启' if self.show_dataset_info else '关闭'}")
                elif key == ord('+') or key == ord('='):
                    new_conf = self.detector.get_confidence_threshold() + 0.05
                    self.detector.set_confidence_threshold(min(new_conf, 0.9))
                elif key == ord('-') or key == ord('_'):
                    new_conf = self.detector.get_confidence_threshold() - 0.05
                    self.detector.set_confidence_threshold(max(new_conf, 0.1))
                elif key == ord('d'):
                    self.show_detections = not self.show_detections
                    print(f"\n👁️ 检测框显示: {'开启' if self.show_detections else '关闭'}")
                elif key == ord('p'):
                    if self.detector.is_selecting_mode:
                        summary = self.detector.smart_tracker.get_tracking_summary()
                        print("\n" + summary)

                current_time = time.time()
                if current_time - self.last_analysis_time > 30:
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
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "tracking" if self.detector.is_selecting_mode else "normal"
            if self.detector.is_selecting_mode and self.detector.smart_tracker.tracked_person_id is not None:
                person_id = self.detector.smart_tracker.tracked_person_id
                filename = f"detection_{mode}_person{person_id}_{timestamp}.jpg"
            else:
                filename = f"detection_{mode}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n📸 画面已保存: {filename}")
            if self.detector.is_selecting_mode and self.frame is not None:
                dataset_frame = self.frame.copy()
                cv2.imwrite(f"dataset_frame_{timestamp}.jpg", dataset_frame)
        except Exception as e:
            print(f"\n❌ 保存失败: {e}")

    def cleanup(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        if self.detector and hasattr(self.detector, 'smart_tracker'):
            dataset = self.detector.smart_tracker.dataset
            total_samples = sum(len(samples) for samples in dataset.dataset.values())
            print(f"\n📊 会话结束 - 数据集总样本数: {total_samples}")
            stats = {
                'timestamp': datetime.now().isoformat(),
                'total_samples': total_samples,
                'persons_tracked': len(dataset.dataset)
            }
            import json
            with open(f"session_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(stats, f, indent=2)
        print("✅ 系统已关闭")


def quick_start():
    app = PersonDetectionApp(conf_threshold=0.5)
    app.run()


def batch_process_mode(video_file=None):
    print("=" * 50)
    print("📼 批量处理模式")
    print("=" * 50)
    if video_file is None:
        print("请指定视频文件路径")
        return
    print(f"处理视频文件: {video_file}")


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