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

        self.last_center = None
        self.last_velocity = (0, 0)

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
        self.is_lost = False
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
        """选择悬停的人物开始追踪"""
        if 0 <= index < len(detections):
            det = detections[index]
            x1, y1, x2, y2, conf, _ = det
            bbox = (x1, y1, x2 - x1, y2 - y1)

            # 初始化追踪器
            if self.tracker_type == 'CSRT':
                self.tracker = cv2.TrackerCSRT_create()
            else:
                self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(frame, bbox)

            # 设置追踪状态
            self.is_selecting_mode = True
            self.selected_person = det  # 保存完整检测信息

            print(f"🎯 已选中人物并启动追踪器")
            print(f"   位置: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"   置信度: {conf:.2f}")

    def draw_detections(self, frame, detections):
        if frame is None:
            return frame

        # ========== 追踪模式 ==========
        if self.is_selecting_mode:
            if self.tracker is not None:
                success, bbox = self.tracker.update(frame)
                h, w = frame.shape[:2]
                lost_this_frame = False

                if success:
                    x, y, bw, bh = [int(v) for v in bbox]
                    x1, y1, x2, y2 = x, y, x + bw, y + bh
                    # print(f"[TRACK] bbox = ({x1},{y1},{x2},{y2}), lost_this_frame = {lost_this_frame}")

                    margin = 50
                    if (x2 < -margin or x1 > w + margin or y2 < -margin or y1 > h + margin or
                            x1 < -margin or y1 < -margin):
                        lost_this_frame = True

                        if (x2 < -margin or x1 > w + margin or y2 < -margin or y1 > h + margin or
                                x1 < -margin or y1 < -margin):
                            lost_this_frame = True
                            # print("[TRACK] lost due to out of bounds")
                        elif bw * bh < 30:
                            lost_this_frame = True
                            # print(f"[TRACK] lost due to small area: {bw * bh}")
                        else:
                            aspect = bw / bh if bh > 0 else 0
                            if aspect < 0.1 or aspect > 2.0:
                                lost_this_frame = True
                                # print(f"[TRACK] lost due to bad aspect: {aspect:.2f}")

                    elif bw * bh < 30:
                        lost_this_frame = True

                        if (x2 < -margin or x1 > w + margin or y2 < -margin or y1 > h + margin or
                                x1 < -margin or y1 < -margin):
                            lost_this_frame = True
                            # print("[TRACK] lost due to out of bounds")
                        elif bw * bh < 30:
                            lost_this_frame = True
                            # print(f"[TRACK] lost due to small area: {bw * bh}")
                        else:
                            aspect = bw / bh if bh > 0 else 0
                            if aspect < 0.1 or aspect > 2.0:
                                lost_this_frame = True
                                # print(f"[TRACK] lost due to bad aspect: {aspect:.2f}")

                    else:
                        aspect = bw / bh if bh > 0 else 0
                        if aspect < 0.1 or aspect > 2.0:
                            lost_this_frame = True

                            if (x2 < -margin or x1 > w + margin or y2 < -margin or y1 > h + margin or
                                    x1 < -margin or y1 < -margin):
                                lost_this_frame = True
                                # print("[TRACK] lost due to out of bounds")
                            elif bw * bh < 30:
                                lost_this_frame = True
                                # print(f"[TRACK] lost due to small area: {bw * bh}")
                            else:
                                aspect = bw / bh if bh > 0 else 0
                                if aspect < 0.1 or aspect > 2.0:
                                    lost_this_frame = True
                                    # print(f"[TRACK] lost due to bad aspect: {aspect:.2f}")

                    # IoU 检查已注释
                else:
                    lost_this_frame = True

                if not lost_this_frame:
                    # ----- 成功追踪 -----
                    self.lost_frame_count = 0
                    self.is_lost = False  # 新增：清除丢失标志
                    self.prev_track_bbox = (x1, y1, x2, y2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    if hasattr(self, 'last_center') and self.last_center is not None:
                        vx = cx - self.last_center[0]
                        vy = cy - self.last_center[1]
                        self.last_velocity = (vx, vy)
                    else:
                        self.last_velocity = (0, 0)
                    self.last_center = (cx, cy)
                    if self.selected_person is not None:
                        self.selected_person = [x1, y1, x2, y2, self.selected_person[4], 0]
                        # print(f"[TRACK] updated selected_person to: {self.selected_person[:4]}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, "TRACKING (CSRT)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # ----- 追踪丢失 -----
                    self.lost_frame_count += 1
                    if self.lost_frame_count <= self.max_lost_frames:
                        self.is_lost = True  # 新增：标记为丢失状态
                        # print(f"[TRACK] predicting, lost_count={self.lost_frame_count}")

                        if self.prev_track_bbox is not None:
                            if hasattr(self, 'last_center') and self.last_center is not None and self.last_velocity != (
                            0, 0):
                                w_box = self.prev_track_bbox[2] - self.prev_track_bbox[0]
                                h_box = self.prev_track_bbox[3] - self.prev_track_bbox[1]
                                pred_cx = self.last_center[0] + self.last_velocity[0]
                                pred_cy = self.last_center[1] + self.last_velocity[1]
                                pred_x1 = pred_cx - w_box // 2
                                pred_y1 = pred_cy - h_box // 2
                                pred_x2 = pred_cx + w_box // 2
                                pred_y2 = pred_cy + h_box // 2
                                cv2.rectangle(frame, (pred_x1, pred_y1), (pred_x2, pred_y2), (128, 128, 128), 2)
                                cv2.putText(frame, "predicting...", (pred_x1, pred_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (128, 128, 128), 1)
                            else:
                                x1, y1, x2, y2 = self.prev_track_bbox
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                                cv2.putText(frame, "predicting...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (128, 128, 128), 1)
                    else:
                        # 彻底丢失，清除追踪状态
                        self.clear_selection()
                        self.tracker = None
                        self.prev_track_bbox = None
                        self.lost_frame_count = 0
                        self.is_lost = False  # 新增：清除丢失标志
                        cv2.putText(frame, "⚠️ TRACKING LOST", (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame
            else:
                # 异常：追踪器丢失但状态未清除
                self.clear_selection()

        # ========== 普通模式（未选中任何人） ==========
        self.hovered_person_index = self.find_hovered_person(detections)
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, _ = det
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
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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