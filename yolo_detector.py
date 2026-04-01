# yolo_detector.py - 性能优化版
import cv2
import torch
import numpy as np
import ssl
import warnings
from feature_tracker import SmartTracker, TrackingVisualizer
from config import Config

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


class YOLOPersonDetector:
    def __init__(self, conf_threshold=0.5, device='cpu'):
        print("=" * 50)
        print("🎯 YOLO人物检测器初始化 (性能优化版)")
        print("=" * 50)

        self.device = torch.device(device)
        self.conf_threshold = conf_threshold

        # 性能优化参数
        self.frame_skip = getattr(Config, 'YOLO_FRAME_SKIP', 1)
        self.yolo_image_size = getattr(Config, 'YOLO_IMAGE_SIZE', 320)
        self.frame_count = 0
        self.last_detections = []
        self.last_frame = None

        # 特征提取间隔
        self.feature_extract_interval = getattr(Config, 'FEATURE_EXTRACT_INTERVAL', 3)
        self.feature_frame_count = 0

        print("\n1. 加载YOLOv5模型...")
        print("   使用轻量级模型: yolov5n")

        try:
            model_name = getattr(Config, 'YOLO_MODEL', 'yolov5n')
            self.model = torch.hub.load('ultralytics/yolov5', model_name,
                                        pretrained=True,
                                        device=self.device,
                                        trust_repo=True)
            print(f"✅ 模型加载成功: {model_name}")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            raise e

        # 配置模型优化
        self.model.conf = conf_threshold
        self.model.classes = getattr(Config, 'YOLO_TARGET_CLASSES', [0])
        self.model.iou = 0.45
        self.model.max_det = 10  # 减少最大检测数
        self.model.agnostic = False
        self.model.multi_label = False

        # 禁用一些功能提升速度
        self.model.augment = False

        # 特征追踪器
        similarity_threshold = getattr(Config, 'SIMILARITY_THRESHOLD', 0.6)
        self.smart_tracker = SmartTracker(similarity_threshold=similarity_threshold)
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
        print(f"   - 检测尺寸: {self.yolo_image_size}px")
        print("=" * 50)

    def detect(self, frame):
        """性能优化版检测"""
        if frame is None:
            return [], frame

        self.frame_count += 1
        self.feature_frame_count += 1
        self.last_frame = frame

        # 跳帧检测
        if self.frame_count % self.frame_skip != 0:
            if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
                return self._predict_detections(), frame
            return self.last_detections, frame

        detections = []

        try:
            # 缩小图像加速检测
            h, w = frame.shape[:2]
            scale = min(self.yolo_image_size / w, self.yolo_image_size / h)

            if scale < 1:
                new_w = int(w * scale)
                new_h = int(h * scale)
                small_frame = cv2.resize(frame, (new_w, new_h))
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = self.model(rgb_frame)

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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        """简单预测"""
        if not self.last_detections:
            return []

        if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
            best_match, _ = self.smart_tracker.track_in_new_frame(self.last_detections, self.last_frame)
            if best_match:
                x1, y1, x2, y2 = best_match['bbox']
                conf = best_match.get('conf', 0)
                return [[x1, y1, x2, y2, conf, 0]]

        return self.last_detections

    # 其他方法保持不变...
    def update_mouse_position(self, x, y):
        self.mouse_x = x
        self.mouse_y = y

    def find_hovered_person(self, detections):
        if self.mouse_x < 0 or self.mouse_y < 0 or len(detections) == 0:
            return -1
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            if x1 <= self.mouse_x <= x2 and y1 <= self.mouse_y <= y2:
                return i
        return -1

    def select_hovered_person(self, detections, index, frame):
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

    def draw_detections(self, frame, detections):
        if frame is None:
            return None

        # 只在追踪模式下绘制详细信息，节省性能
        if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
            best_match, score = self.smart_tracker.track_in_new_frame(detections, frame)
            if best_match:
                x1, y1, x2, y2 = best_match['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (0, 0, 255), -1)
            else:
                h, w = frame.shape[:2]
                cv2.putText(frame, "LOST", (w // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # 简化绘制
            self.hovered_person_index = self.find_hovered_person(detections)
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id = det
                if i == self.hovered_person_index:
                    color = (0, 0, 255)
                    thickness = 2
                else:
                    color = (0, 255, 0)
                    thickness = 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame

    def clear_selection(self):
        self.selected_person_index = -1
        self.selected_person = None
        self.is_selecting_mode = False

    def get_detection_summary(self, detections):
        count = len(detections)
        if count == 0:
            return "No person"
        elif count == 1:
            return f"1 person ({detections[0][4]:.2f})"
        else:
            return f"{count} persons"

    def set_confidence_threshold(self, threshold):
        self.conf_threshold = max(0.1, min(0.9, threshold))
        self.model.conf = self.conf_threshold

    def get_confidence_threshold(self):
        return self.conf_threshold

    def get_tracking_history(self):
        return self.smart_tracker.tracking_history

    def get_dataset_stats(self):
        if hasattr(self.smart_tracker, 'dataset'):
            dataset = self.smart_tracker.dataset
            total_samples = sum(len(samples) for samples in dataset.dataset.values())
            persons = len(dataset.dataset)
            return total_samples, persons
        return 0, 0