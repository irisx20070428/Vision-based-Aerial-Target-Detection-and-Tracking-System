# yolo_detector.py - 完整版（使用增强追踪器）
import cv2
import torch
import numpy as np
import ssl
import warnings
from enhanced_tracker import EnhancedSmartTracker, FaceAngle
from config import Config

# 解决SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


class YOLOPersonDetector:
    """YOLO人物检测器类 - 集成增强特征追踪"""

    def __init__(self, conf_threshold=0.5, device='cpu'):
        """
        初始化YOLO检测器
        """
        print("=" * 50)
        print("🎯 YOLO人物检测器初始化")
        print("=" * 50)

        # 设置设备
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold

        # 跳帧优化
        self.frame_skip = getattr(Config, 'YOLO_FRAME_SKIP', 2)
        self.frame_count = 0
        self.last_detections = []
        self.last_frame = None

        # 加载YOLOv5模型
        print("\n1. 加载YOLOv5模型...")
        print("   (第一次运行会下载模型，约需1-2分钟)")

        try:
            # 使用指定模型
            model_name = getattr(Config, 'YOLO_MODEL', 'yolov5n')
            self.model = torch.hub.load('ultralytics/yolov5', model_name,
                                        pretrained=True,
                                        device=self.device,
                                        trust_repo=True)
            print(f"✅ 模型加载成功: {model_name}")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            raise e

        # 配置模型
        self.model.conf = conf_threshold
        self.model.classes = [0]  # 只检测person
        self.model.iou = 0.45
        self.model.max_det = 20  # 最多检测20个人

        # ==== 增强特征追踪器 ====
        self.smart_tracker = EnhancedSmartTracker(similarity_threshold=0.45)

        # ==== 鼠标交互相关变量 ====
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

        参数:
            frame: 输入图像 (BGR格式)

        返回:
            detections: 检测结果列表
            annotated_frame: 标注后的图像
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
            # YOLO需要RGB格式
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
        """
        更新鼠标位置 - 由鼠标回调函数调用

        参数:
            x, y: 鼠标在图像中的坐标
        """
        self.mouse_x = x
        self.mouse_y = y

    def find_hovered_person(self, detections):
        """
        查找鼠标悬停的人物

        参数:
            detections: 检测结果列表

        返回:
            index: 悬停的人物索引，-1表示没有
        """
        if self.mouse_x < 0 or self.mouse_y < 0 or len(detections) == 0:
            return -1

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det

            # 检查鼠标是否在检测框内
            if x1 <= self.mouse_x <= x2 and y1 <= self.mouse_y <= y2:
                return i

        return -1

    def select_hovered_person(self, detections, index, frame):
        """
        选择悬停的人物 - 使用增强特征追踪

        参数:
            detections: 检测结果列表
            index: 要选择的人物索引
            frame: 当前帧图像
        """
        if 0 <= index < len(detections):
            det = detections[index]
            x1, y1, x2, y2, conf, cls_id = det

            # 裁剪人物图像
            person_img = frame[y1:y2, x1:x2]

            # 使用智能追踪器选择人物
            person_id = self.smart_tracker.select_person(
                person_img,
                (x1, y1, x2, y2),
                {'confidence': conf}
            )

            self.selected_person_index = index
            self.selected_person = det
            self.is_selecting_mode = True

            print(f"\n🎯 已选中人物 #{index + 1}")
            print(f"   位置: ({x1}, {y1}) - ({x2}, {y2})")
            print(f"   置信度: {conf:.2f}")

    def draw_detections(self, frame, detections):
        """
        在画面上绘制检测框（带增强特征追踪）

        参数:
            frame: 原始图像
            detections: 检测结果列表

        返回:
            frame: 标注后的图像
        """
        if frame is None:
            return None

        # 如果在追踪模式下
        if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
            # 使用增强追踪器在新帧中寻找目标
            best_match, score = self.smart_tracker.track_in_new_frame(detections, frame)

            if best_match:
                # 使用增强版追踪绘制
                frame = self.smart_tracker.draw_tracking_info(frame, best_match, score)
                # 绘制角度分布图
                frame = self.smart_tracker.draw_angle_chart(frame)

                # 更新选中人物信息
                x1, y1, x2, y2 = best_match['bbox']
                conf = best_match.get('conf', 0)
                self.selected_person = [x1, y1, x2, y2, conf, 0]
            else:
                # 追踪丢失
                h, w = frame.shape[:2]
                cv2.putText(frame, "⚠️ TRACKING LOST", (w // 2 - 150, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            # 普通模式下，找出鼠标悬停的人物
            self.hovered_person_index = self.find_hovered_person(detections)

            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id = det

                # 根据状态选择颜色
                if i == self.hovered_person_index:
                    color = (0, 0, 0)  # 黑色 - 鼠标悬停
                    thickness = 3
                    label = f"CLICK to track: {conf:.2f}"
                else:
                    # 根据置信度设置颜色
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

                # 计算标签大小
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # 绘制标签背景
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

                # 绘制标签文字
                text_color = (255, 255, 255)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        return frame

    def clear_selection(self):
        """清除选中状态"""
        self.selected_person_index = -1
        self.selected_person = None
        self.is_selecting_mode = False
        print("\n🔄 已清除选择，但数据集已保存")

    def get_detection_summary(self, detections):
        """
        获取检测摘要

        参数:
            detections: 检测结果列表

        返回:
            summary: 摘要字符串
        """
        if self.is_selecting_mode and self.smart_tracker.tracked_person_id is not None:
            # 从追踪器获取状态
            history = self.smart_tracker.tracking_data['history']
            if history:
                last = history[-1]
                score = last.get('score', 0)
                angle = last.get('face_angle', 'unknown')
                return f"Tracking | Score: {score:.2f} | Angle: {angle}"
            else:
                return f"Tracking Mode"

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

    def get_tracking_summary(self):
        """获取追踪摘要"""
        return self.smart_tracker.get_summary()