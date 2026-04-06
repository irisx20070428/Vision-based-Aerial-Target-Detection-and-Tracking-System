# yolo_detector.py (最终版 - 集成帧跳过、丢失容忍、IoU辅助)
import cv2
import torch
import numpy as np
import ssl
import warnings
from feature_tracker import SmartTracker
from config import Config

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


class YOLOPersonDetector:
    def __init__(self, conf_threshold=0.5, device='cpu'):
        print("=" * 50)
        print("🎯 YOLO人物检测器初始化")
        print("=" * 50)

        self.device = torch.device(device)
        self.conf_threshold = conf_threshold or Config.YOLO_CONF_THRESHOLD

        print("\n1. 加载YOLOv5模型...")
        print("   (第一次运行会下载模型，约需1-2分钟)")
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                                        pretrained=True,
                                        device=self.device,
                                        trust_repo=True)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            raise e

        self.model.conf = conf_threshold
        self.model.classes = [0]  # 只检测person
        self.model.iou = 0.45
        self.model.max_det = 20

        self.smart_tracker = SmartTracker(similarity_threshold=0.3)  # 降低阈值
        # self.visualizer = TrackingVisualizer()

        self.mouse_x = -1
        self.mouse_y = -1
        self.hovered_person_index = -1
        self.selected_person_index = -1
        self.selected_person = None
        self.is_selecting_mode = False

        # 帧跳过优化
        self.frame_skip = Config.YOLO_FRAME_SKIP
        self.frame_count = 0
        self.cached_match = None
        self.prev_track_bbox = None

        # 丢失容忍
        self.lost_frame_count = 0
        self.max_lost_frames = 10

        print(f"\n✅ YOLO模型配置完成")
        print(f"   - 置信度阈值: {conf_threshold}")
        print(f"   - 检测目标: person")
        print("=" * 50)

    def detect(self, frame):
        if frame is None:
            return [], frame
        detections = []
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_frame)
            if len(results.xyxy[0]) > 0:
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
                    if int(cls_id) == 0:
                        detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_id)])
        except Exception as e:
            print(f"检测错误: {e}")
        return detections, frame

    def update_mouse_position(self, x, y):
        self.mouse_x = x
        self.mouse_y = y

    def find_hovered_person(self, detections):
        if self.mouse_x < 0 or self.mouse_y < 0 or len(detections) == 0:
            return -1
        for i, det in enumerate(detections):
            x1, y1, x2, y2, _, _ = det
            if x1 <= self.mouse_x <= x2 and y1 <= self.mouse_y <= y2:
                return i
        return -1

    def select_hovered_person(self, detections, index, frame):
        if 0 <= index < len(detections):
            det = detections[index]
            x1, y1, x2, y2, conf, _ = det
            person_img = frame[y1:y2, x1:x2]

            # 保存到数据集（异步，不阻塞）
            person_id = self.smart_tracker.select_person(person_img, (x1, y1, x2, y2), {'confidence': conf})

            # 🔥 关键：记录上一帧位置，重置丢失计数，进入追踪模式
            self.prev_track_bbox = (x1, y1, x2, y2)
            self.lost_frame_count = 0
            self.is_selecting_mode = True
            self.selected_person_index = index
            self.selected_person = det

            print(f"\n🎯 已选中人物 #{index + 1} (ID: {person_id})")
            print(f"   位置: ({x1}, {y1}) - ({x2}, {y2})")

    def draw_detections(self, frame, detections):
        """
        绘制检测框和追踪框（追踪模式使用 IoU 匹配，无特征提取，保证流畅）
        """
        if frame is None:
            return None

        # ========== 追踪模式 ==========
        if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
            self.frame_count += 1
            best_match = None
            best_iou = 0.0

            # 用 IoU 寻找最佳匹配（不提取任何特征）
            if self.prev_track_bbox is not None:
                for det in detections:
                    x1, y1, x2, y2, conf, _ = det
                    iou = self._compute_iou(self.prev_track_bbox, (x1, y1, x2, y2))
                    if iou > best_iou:
                        best_iou = iou
                        best_match = {
                            'bbox': (x1, y1, x2, y2),
                            'score': iou,
                            'angle': 'unknown',
                            'conf': conf
                        }

            # 匹配成功（IoU > 阈值）
            if best_match and best_iou > 0.3:
                self.lost_frame_count = 0
                self.prev_track_bbox = best_match['bbox']
                x1, y1, x2, y2 = best_match['bbox']
                conf = best_match['conf']
                # 绘制追踪框（绿色，粗边框）
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"TRACKING (IoU={best_iou:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"conf={conf:.2f}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # 匹配失败：增加丢失计数
                self.lost_frame_count += 1
                if self.lost_frame_count <= self.max_lost_frames and self.prev_track_bbox is not None:
                    # 预测位置（简单沿用上一帧位置）
                    x1, y1, x2, y2 = self.prev_track_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                    cv2.putText(frame, "predicting...", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                else:
                    # 真正丢失：清空状态
                    self.cached_match = None
                    self.prev_track_bbox = None
                    self.lost_frame_count = 0
                    self.is_selecting_mode = False  # 自动退出追踪模式
                    h, w = frame.shape[:2]
                    cv2.putText(frame, "⚠️ TRACKING LOST", (w // 2 - 150, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 注意：追踪模式下不再绘制普通检测框，以免干扰（也可以选择绘制半透明框，但通常不需要）
            return frame

        # ========== 普通模式（未选中任何人） ==========
        # 找出鼠标悬停的人物索引
        self.hovered_person_index = self.find_hovered_person(detections)

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, _ = det

            # 根据状态选择颜色
            if i == self.hovered_person_index:
                color = (0, 0, 0)  # 黑色 - 鼠标悬停
                thickness = 3
                label = f"CLICK to track: {conf:.2f}"
            else:
                if conf > 0.7:
                    color = (0, 255, 0)  # 绿色
                elif conf > 0.5:
                    color = (0, 255, 255)  # 黄色
                else:
                    color = (0, 165, 255)  # 橙色
                thickness = 2
                label = f"person: {conf:.2f}"

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签背景和文字
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def clear_selection(self):
        self.selected_person_index = -1
        self.selected_person = None
        self.is_selecting_mode = False
        print("\n🔄 已清除选择，但数据集已保存")

    def get_detection_summary(self, detections):
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
            return f"1 person detected (conf: {detections[0][4]:.2f})"
        else:
            avg_conf = sum(d[4] for d in detections) / count
            return f"{count} persons detected (avg conf: {avg_conf:.2f})"

    def set_confidence_threshold(self, threshold):
        self.conf_threshold = max(0.1, min(0.9, threshold))
        self.model.conf = self.conf_threshold
        print(f"📊 置信度阈值调整为: {self.conf_threshold:.2f}")

    def get_confidence_threshold(self):
        return self.conf_threshold

    def _compute_iou(self, box1, box2):
        """计算两个边界框的 IoU（交并比）"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0