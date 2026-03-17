# yolo_detector.py (完整修复版)
import cv2
import torch
import numpy as np
import ssl
import warnings
from feature_tracker import SmartTracker, TrackingVisualizer

# 解决SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


class YOLOPersonDetector:
    """YOLO人物检测器类 - 集成特征追踪"""

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

        # 加载YOLOv5模型
        print("\n1. 加载YOLOv5模型...")
        print("   (第一次运行会下载模型，约需1-2分钟)")

        try:
            # 尝试加载模型
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                                        pretrained=True,
                                        device=self.device,
                                        trust_repo=True)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            raise e

        # 配置模型
        self.model.conf = conf_threshold
        self.model.classes = [0]  # 只检测person
        self.model.iou = 0.45
        self.model.max_det = 20  # 最多检测20个人

        # ==== 特征追踪器 ====
        self.smart_tracker = SmartTracker(similarity_threshold=0.6)
        self.visualizer = TrackingVisualizer()

        # ==== 鼠标交互相关变量 ====
        self.mouse_x = -1
        self.mouse_y = -1
        self.hovered_person_index = -1
        self.selected_person_index = -1
        self.selected_person = None
        self.is_selecting_mode = False

        print(f"\n✅ YOLO模型配置完成")
        print(f"   - 置信度阈值: {conf_threshold}")
        print(f"   - 检测目标: person")
        print("=" * 50)

    def detect(self, frame):
        """
        检测画面中的人物 - 这是主程序调用的方法

        参数:
            frame: 输入图像 (BGR格式)

        返回:
            detections: 检测结果列表
            annotated_frame: 标注后的图像
        """
        if frame is None:
            return [], frame

        detections = []

        try:
            # YOLO需要RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 执行检测
            results = self.model(rgb_frame)

            # 获取检测结果
            if len(results.xyxy[0]) > 0:
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()

                    # 只保留person (class 0)
                    if int(cls_id) == 0:
                        detections.append([int(x1), int(y1), int(x2), int(y2),
                                           float(conf), int(cls_id)])
        except Exception as e:
            print(f"检测错误: {e}")

        return detections, frame

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
        选择悬停的人物 - 使用特征追踪

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

            print(f"\n🎯 已选中人物 #{index + 1} (ID: {person_id})")
            print(f"   位置: ({x1}, {y1}) - ({x2}, {y2})")
            print(f"   置信度: {conf:.2f}")

    def draw_detections(self, frame, detections):
        """
        在画面上绘制检测框（带特征追踪）

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
            # 使用智能追踪器在新帧中寻找目标
            best_match, score = self.smart_tracker.track_in_new_frame(detections, frame)

            if best_match:
                # 绘制追踪信息
                frame = self.visualizer.draw_tracking_info(
                    frame,
                    best_match,
                    score,
                    self.smart_tracker.tracked_person_id
                )

                # 绘制角度分布
                frame = self.visualizer.draw_angle_distribution(
                    frame,
                    self.smart_tracker.tracking_history
                )

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
        # 不重置smart_tracker，保持数据集
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
        return self.smart_tracker.tracking_history if hasattr(self.smart_tracker, 'tracking_history') else []

    def get_dataset_stats(self):
        """获取数据集统计"""
        if hasattr(self.smart_tracker, 'dataset'):
            dataset = self.smart_tracker.dataset
            total_samples = sum(len(samples) for samples in dataset.dataset.values())
            persons = len(dataset.dataset)
            return total_samples, persons
        return 0, 0


