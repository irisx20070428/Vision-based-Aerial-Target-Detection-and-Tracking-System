import cv2
import torch
import numpy as np
import ssl
import warnings
from feature_tracker import SmartTracker, TrackingVisualizer
from config import Config

# 解决SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


class YOLOPersonDetector:
    """YOLO人物检测器类 - 集成特征追踪和跳帧优化"""

    def __init__(self, conf_threshold=0.5, device='cpu'):
        print("=" * 50)
        print("🎯 YOLO人物检测器初始化")
        print("=" * 50)

        self.device = torch.device(device)
        self.conf_threshold = conf_threshold

        # 跳帧优化
        self.frame_skip = Config.YOLO_FRAME_SKIP
        self.frame_count = 0
        self.last_detections = []
        self.last_frame = None

        print("\n1. 加载YOLOv5模型...")
        print("   (第一次运行会下载模型，约需1-2分钟)")

        try:
            # 使用指定模型
            self.model = torch.hub.load('ultralytics/yolov5', Config.YOLO_MODEL,
                                        pretrained=True,
                                        device=self.device,
                                        trust_repo=True)
            print(f"✅ 模型加载成功: {Config.YOLO_MODEL}")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            raise e

        # 配置模型
        self.model.conf = conf_threshold
        self.model.classes = Config.YOLO_TARGET_CLASSES
        self.model.iou = 0.45
        self.model.max_det = 20

        # 特征追踪器
        self.smart_tracker = SmartTracker(similarity_threshold=Config.SIMILARITY_THRESHOLD)
        self.visualizer = TrackingVisualizer()

        # 鼠标交互变量
        self.mouse_x = -1
        self.mouse_y = -1
        self.hovered_person_index = -1
        self.selected_person_index = -1
        self.selected_person = None
        self.is_selecting_mode = False

        print(f"\n✅ YOLO模型配置完成")
        print(f"   - 置信度阈值: {conf_threshold}")
        print(f"   - 跳帧检测: 每{self.frame_skip}帧检测一次")
        print(f"   - 检测目标: person")
        print("=" * 50)

    def detect(self, frame):
        """
        检测画面中的人物 - 支持跳帧优化
        """
        if frame is None:
            return [], frame

        self.frame_count += 1
        self.last_frame = frame

        # 跳帧检测
        if self.frame_count % self.frame_skip != 0:
            # 如果在追踪模式下，使用追踪结果预测
            if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
                return self._predict_detections(), frame
            return self.last_detections, frame

        detections = []

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 可选：缩小图像加速检测（树莓派优化）
            use_resize = frame.shape[1] > 480 and self.device.type == 'cpu'

            if use_resize:
                scale = 480 / frame.shape[1]
                small_frame = cv2.resize(rgb_frame, (480, int(frame.shape[0] * scale)))
                results = self.model(small_frame)

                # 还原坐标
                if len(results.xyxy[0]) > 0:
                    for det in results.xyxy[0]:
                        x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                        detections.append([x1, y1, x2, y2, float(conf), int(cls_id)])
            else:
                results = self.model(rgb_frame)
                if len(results.xyxy[0]) > 0:
                    for det in results.xyxy[0]:
                        x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
                        detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_id)])

            self.last_detections = detections

        except Exception as e:
            print(f"检测错误: {e}")
            detections = self.last_detections

        return detections, frame

    def _predict_detections(self):
        """
        预测下一帧的检测结果（简单运动预测）
        """
        if not self.last_detections:
            return []

        # 如果在追踪模式下，优先使用追踪结果
        if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
            best_match, _ = self.smart_tracker.track_in_new_frame(self.last_detections, self.last_frame)
            if best_match:
                x1, y1, x2, y2 = best_match['bbox']
                conf = best_match.get('conf', 0)
                return [[x1, y1, x2, y2, conf, 0]]

        return self.last_detections

    def update_mouse_position(self, x, y):
        """更新鼠标位置"""
        self.mouse_x = x
        self.mouse_y = y

    def find_hovered_person(self, detections):
        """查找鼠标悬停的人物"""
        if self.mouse_x < 0 or self.mouse_y < 0 or len(detections) == 0:
            return -1

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            if x1 <= self.mouse_x <= x2 and y1 <= self.mouse_y <= y2:
                return i
        return -1

    def select_hovered_person(self, detections, index, frame):
        """选择悬停的人物"""
        if 0 <= index < len(detections):
            det = detections[index]
            x1, y1, x2, y2, conf, cls_id = det

            person_img = frame[y1:y2, x1:x2]
            person_id = self.smart_tracker.select_person(
                person_img,
                (x1, y1, x2, y2),
                {'confidence': conf}
            )

            self.selected_person_index = index
            self.selected_person = det
            self.is_selecting_mode = True

            print(f"\n🎯 已选中人物 #{index + 1} (ID: {person_id})")
            print(f"   位置: ({x1}, {y1}) - ({x2}, {y2})")
            print(f"   置信度: {conf:.2f}")

    def draw_detections(self, frame, detections):
        """在画面上绘制检测框"""
        if frame is None:
            return None

        if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
            best_match, score = self.smart_tracker.track_in_new_frame(detections, frame)

            if best_match:
                frame = self.visualizer.draw_tracking_info(
                    frame, best_match, score, self.smart_tracker.tracked_person_id
                )
                x1, y1, x2, y2 = best_match['bbox']
                conf = best_match.get('conf', 0)
                self.selected_person = [x1, y1, x2, y2, conf, 0]
            else:
                h, w = frame.shape[:2]
                cv2.putText(frame, "⚠️ TRACKING LOST", (w // 2 - 150, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.hovered_person_index = self.find_hovered_person(detections)

            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id = det

                if i == self.hovered_person_index:
                    color = (0, 0, 0)
                    thickness = 3
                    label = f"CLICK to track: {conf:.2f}"
                else:
                    if conf > 0.7:
                        color = (0, 255, 0)
                    elif conf > 0.5:
                        color = (0, 255, 255)
                    else:
                        color = (0, 165, 255)
                    thickness = 2
                    label = f"person: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def clear_selection(self):
        """清除选中状态"""
        self.selected_person_index = -1
        self.selected_person = None
        self.is_selecting_mode = False
        print("\n🔄 已清除选择，但数据集已保存")

    def get_detection_summary(self, detections):
        """获取检测摘要"""
        if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
            history = self.smart_tracker.tracking_history
            if history:
                last_score = history[-1]['score'] if history else 0
                angle = history[-1]['angle'] if history else "unknown"
                return f"Tracking Person #{self.smart_tracker.tracked_person_id} | Score: {last_score:.2f} | Angle: {angle}"
            else:
                return f"Tracking Person #{self.smart_tracker.tracked_person_id}"

        count = len(detections)
        if count == 0:
            return "No person detected"
        elif count == 1:
            conf = detections[0][4]
            return f"1 person detected (conf: {conf:.2f})"
        else:
            avg_conf = sum(d[4] for d in detections) / count
            return f"{count} persons detected (avg conf: {avg_conf:.2f})"

    def set_confidence_threshold(self, threshold):
        """设置置信度阈值"""
        self.conf_threshold = max(0.1, min(0.9, threshold))
        self.model.conf = self.conf_threshold
        print(f"📊 置信度阈值调整为: {self.conf_threshold:.2f}")

    def get_confidence_threshold(self):
        """获取当前置信度阈值"""
        return self.conf_threshold

    def get_tracking_history(self):
        """获取追踪历史"""
        return self.smart_tracker.tracking_history

    def get_dataset_stats(self):
        """获取数据集统计"""
        if hasattr(self.smart_tracker, 'dataset'):
            dataset = self.smart_tracker.dataset
            total_samples = sum(len(samples) for samples in dataset.dataset.values())
            persons = len(dataset.dataset)
            return total_samples, persons
        return 0, 0