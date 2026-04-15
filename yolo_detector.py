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

        self.tracker = None  # OpenCV 追踪器
        self.tracker_type = 'CSRT'  # 或 'KCF'（更快但稍弱）

        print("=" * 50)
        print("🎯 YOLO人物检测器初始化")
        print("=" * 50)

        self.device = torch.device(device)
        self.conf_threshold = conf_threshold or Config.YOLO_CONF_THRESHOLD

        print("\n1. 加载YOLOv5模型...")
        print("   (第一次运行会下载模型，约需1-2分钟)")
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n',
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
        self.max_lost_frames = 3

        print(f"\n✅ YOLO模型配置完成")
        print(f"   - 置信度阈值: {conf_threshold}")
        print(f"   - 检测目标: person")
        print("=" * 50)

    def detect(self, frame):
        if frame is None:
            return [], frame
        detections = []
        try:
            h, w = frame.shape[:2]
            # 设置检测时的目标尺寸（宽度不超过 320，保持宽高比）
            target_width = 320
            scale = target_width / w
            if scale < 1:  # 只有当原图宽度大于 target_width 时才缩小
                new_w = target_width
                new_h = int(h * scale)
                small_frame = cv2.resize(frame, (new_w, new_h))
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = self.model(rgb_small)
                # 将检测框坐标还原到原图
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                    detections.append([x1, y1, x2, y2, float(conf), int(cls_id)])
            else:
                # 原图已经很小，直接检测
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(rgb_frame)
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
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
            bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)
            # 初始化追踪器
            if self.tracker_type == 'CSRT':
                self.tracker = cv2.TrackerCSRT_create()
            else:
                self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(frame, bbox)
            self.is_selecting_mode = True
            self.selected_person = det
            self.prev_track_bbox = (x1, y1, x2, y2)
            print(f"🎯 已选中人物并启动追踪器")

    def draw_detections(self, frame, detections):
        if frame is None:
            return frame

        # ========== 追踪模式 ==========
        if self.is_selecting_mode and self.tracker is not None:
            # 更新追踪器
            success, bbox = self.tracker.update(frame)
            h, w = frame.shape[:2]

            if success:
                x, y, bw, bh = [int(v) for v in bbox]
                x1, y1, x2, y2 = x, y, x + bw, y + bh

                # ===== 新增：有效性检查 =====
                # 1. 检查框是否超出画面边界（允许少量超出，但超出太多则无效）
                margin = 20
                if (x2 < -margin or x1 > w + margin or
                        y2 < -margin or y1 > h + margin):
                    success = False
                # 2. 检查框面积是否合理（避免漂移到极小区域）
                elif bw * bh < 100:  # 面积太小，认为丢失
                    success = False
                # 3. 可选：检查框中心是否离画面中心太远（如果人物完全离开）
                # center_x = (x1 + x2) // 2
                # center_y = (y1 + y2) // 2
                # if center_x < -50 or center_x > w+50 or center_y < -50 or center_y > h+50:
                #     success = False
                # ===========================

            if success:
                # 更新上一帧位置
                self.prev_track_bbox = (x1, y1, x2, y2)
                # 绘制绿色追踪框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, "TRACKING (CSRT)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # 追踪失败：立即清除追踪状态
                cv2.putText(frame, "⚠️ TRACKING LOST", (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.clear_selection()  # 退出追踪模式
                self.tracker = None  # 释放追踪器
                self.prev_track_bbox = None  # 清除上一帧位置
            return frame  # 追踪模式下不画普通检测框

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