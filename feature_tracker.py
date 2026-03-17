# feature_tracker.py
import cv2
import numpy as np
import os
import json
from datetime import datetime
from enum import Enum


class FaceAngle(Enum):
    """人脸角度枚举"""
    FRONT = "front"  # 正脸
    SIDE = "side"  # 侧脸
    PROFILE = "profile"  # 侧面轮廓
    UP = "up"  # 仰视
    DOWN = "down"  # 俯视
    BACK = "back"  # 背对
    UNKNOWN = "unknown"


class PersonFeatureExtractor:
    """人物特征提取器"""

    def __init__(self):
        # 特征提取参数
        self.hist_bins = [8, 8, 8]  # 颜色直方图bin数
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # 特征权重配置
        self.weights = {
            'color_hist': 0.3,
            'hog': 0.3,
            'edge': 0.2,
            'texture': 0.2
        }

    def extract_color_histogram(self, person_img):
        """
        提取颜色直方图特征
        原理：统计人物区域的RGB颜色分布
        """
        # 转换到HSV颜色空间（对光照变化更鲁棒）
        hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)

        # 计算3D直方图
        hist = cv2.calcHist([hsv], [0, 1, 2], None,
                            self.hist_bins, [0, 180, 0, 256, 0, 256])

        # 归一化
        cv2.normalize(hist, hist)
        return hist.flatten()

    def extract_hog_features(self, person_img):
        """
        提取HOG特征（人体轮廓和形状）
        原理：梯度方向直方图，捕捉人体轮廓
        """
        # 调整大小到统一尺寸
        resized = cv2.resize(person_img, (64, 128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 计算HOG特征
        hog_features = self.hog.compute(gray)
        return hog_features.flatten()

    def extract_edge_features(self, person_img):
        """
        提取边缘特征（用于判断姿态）
        原理：Canny边缘检测，反映人体姿态
        """
        gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)

        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 计算边缘密度（边缘像素比例）
        edge_density = np.sum(edges > 0) / edges.size

        # 计算边缘方向直方图
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi

        # 方向直方图
        hist_dir, _ = np.histogram(direction[magnitude > 50],
                                   bins=36, range=(-180, 180))
        hist_dir = hist_dir / (np.sum(hist_dir) + 1e-6)

        return np.concatenate([[edge_density], hist_dir])

    def extract_texture_features(self, person_img):
        """
        提取纹理特征（LBP局部二值模式）
        原理：捕捉服装纹理
        """
        gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)

        # 简化的LBP
        lbp = np.zeros_like(gray)
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i - 1, j - 1] > center) << 7
                code |= (gray[i - 1, j] > center) << 6
                code |= (gray[i - 1, j + 1] > center) << 5
                code |= (gray[i, j + 1] > center) << 4
                code |= (gray[i + 1, j + 1] > center) << 3
                code |= (gray[i + 1, j] > center) << 2
                code |= (gray[i + 1, j - 1] > center) << 1
                code |= (gray[i, j - 1] > center) << 0
                lbp[i, j] = code

        # LBP直方图
        hist_lbp = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [256], [0, 256])
        cv2.normalize(hist_lbp, hist_lbp)

        return hist_lbp.flatten()

    def estimate_face_angle(self, person_img):
        """
        估计人脸角度（正脸/侧脸/俯视/仰视）
        返回角度枚举和置信度
        """
        # 这里简化处理，实际可用人脸关键点检测
        h, w = person_img.shape[:2]

        # 使用宽高比初步判断
        aspect_ratio = w / h

        if aspect_ratio > 0.8:  # 较宽，可能是正脸
            return FaceAngle.FRONT, 0.7
        elif aspect_ratio < 0.4:  # 很窄，可能是侧脸
            return FaceAngle.SIDE, 0.6
        elif aspect_ratio < 0.6:  # 中等，可能是侧面轮廓
            return FaceAngle.PROFILE, 0.5
        else:
            return FaceAngle.UNKNOWN, 0.3

    def extract_all_features(self, person_img):
        """
        提取所有特征
        """
        features = {
            'color_hist': self.extract_color_histogram(person_img),
            'hog': self.extract_hog_features(person_img),
            'edge': self.extract_edge_features(person_img),
            'texture': self.extract_texture_features(person_img),
            'face_angle': self.estimate_face_angle(person_img)
        }

        return features


class FeatureComparator:
    """特征比较器 - 计算相似度"""

    def __init__(self):
        self.extractor = PersonFeatureExtractor()

    def compute_similarity(self, features1, features2):
        """
        计算两组特征的相似度
        返回0-1之间的分数
        """
        scores = {}

        # 颜色直方图相似度（相关系数）
        if 'color_hist' in features1 and 'color_hist' in features2:
            scores['color_hist'] = cv2.compareHist(
                features1['color_hist'].reshape(-1, 1).astype(np.float32),
                features2['color_hist'].reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )

        # HOG特征相似度（余弦相似度）
        if 'hog' in features1 and 'hog' in features2:
            hog1 = features1['hog'].flatten()
            hog2 = features2['hog'].flatten()
            norm1 = np.linalg.norm(hog1)
            norm2 = np.linalg.norm(hog2)
            if norm1 > 0 and norm2 > 0:
                scores['hog'] = np.dot(hog1, hog2) / (norm1 * norm2)
            else:
                scores['hog'] = 0

        # 边缘特征相似度
        if 'edge' in features1 and 'edge' in features2:
            edge1 = features1['edge'].flatten()
            edge2 = features2['edge'].flatten()
            scores['edge'] = 1 - np.linalg.norm(edge1 - edge2) / (np.linalg.norm(edge1) + np.linalg.norm(edge2) + 1e-6)

        # 纹理特征相似度
        if 'texture' in features1 and 'texture' in features2:
            texture1 = features1['texture'].flatten()
            texture2 = features2['texture'].flatten()
            scores['texture'] = 1 - np.linalg.norm(texture1 - texture2) / (
                        np.linalg.norm(texture1) + np.linalg.norm(texture2) + 1e-6)

        # 角度相似度
        if 'face_angle' in features1 and 'face_angle' in features2:
            angle1, conf1 = features1['face_angle']
            angle2, conf2 = features2['face_angle']

            if angle1 == angle2:
                scores['face_angle'] = conf1 * conf2
            else:
                scores['face_angle'] = 0

        # 加权平均
        weights = self.extractor.weights
        total_score = 0
        total_weight = 0

        for key, score in scores.items():
            if key in weights:
                total_score += score * weights[key]
                total_weight += weights[key]

        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0


class FeatureDataset:
    """特征数据集 - 保存人物特征用于对比"""

    def __init__(self, save_dir="person_dataset"):
        self.save_dir = save_dir
        self.dataset = {}  # person_id -> {features, images, metadata}
        self.next_id = 0

        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(f"{save_dir}/images")
            os.makedirs(f"{save_dir}/features")

    def add_person_sample(self, person_img, features, metadata=None):
        """
        添加人物样本到数据集
        """
        person_id = self.next_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存图像
        img_filename = f"{self.save_dir}/images/person_{person_id}_{timestamp}.jpg"
        cv2.imwrite(img_filename, person_img)

        # 保存特征
        feature_data = {
            'person_id': person_id,
            'timestamp': timestamp,
            'features': {
                'color_hist': features['color_hist'].tolist(),
                'hog': features['hog'].tolist() if hasattr(features['hog'], 'tolist') else features['hog'],
                'edge': features['edge'].tolist(),
                'texture': features['texture'].tolist(),
                'face_angle': [features['face_angle'][0].value, features['face_angle'][1]]
            },
            'metadata': metadata or {},
            'image_path': img_filename
        }

        feature_filename = f"{self.save_dir}/features/person_{person_id}_{timestamp}.json"
        with open(feature_filename, 'w') as f:
            json.dump(feature_data, f, indent=2)

        # 添加到内存数据集
        if person_id not in self.dataset:
            self.dataset[person_id] = []

        self.dataset[person_id].append({
            'features': features,
            'timestamp': timestamp,
            'image_path': img_filename,
            'metadata': metadata
        })

        print(f"✅ 已保存人物 #{person_id} 的样本 (角度: {features['face_angle'][0].value})")

        return person_id

    def find_best_match(self, query_features, comparator):
        """
        在数据集中查找最佳匹配
        """
        best_match = None
        best_score = 0
        best_person_id = None

        for person_id, samples in self.dataset.items():
            for sample in samples:
                score = comparator.compute_similarity(query_features, sample['features'])

                if score > best_score:
                    best_score = score
                    best_match = sample
                    best_person_id = person_id

        return best_person_id, best_match, best_score


class SmartTracker:
    """智能追踪器 - 基于特征匹配"""

    def __init__(self, similarity_threshold=0.6):
        self.extractor = PersonFeatureExtractor()
        self.comparator = FeatureComparator()
        self.dataset = FeatureDataset()

        self.tracked_person_id = None
        self.tracked_features = None
        self.similarity_threshold = similarity_threshold
        self.tracking_history = []

    def select_person(self, person_img, person_bbox, frame_metadata=None):
        """
        选择人物开始追踪
        """
        # 提取特征
        features = self.extractor.extract_all_features(person_img)

        # 添加到数据集
        metadata = {
            'bbox': person_bbox,
            'frame_metadata': frame_metadata
        }
        person_id = self.dataset.add_person_sample(person_img, features, metadata)

        # 开始追踪
        self.tracked_person_id = person_id
        self.tracked_features = features

        print(f"\n🎯 开始追踪人物 #{person_id}")
        print(f"   初始角度: {features['face_angle'][0].value}")

        return person_id

    def track_in_new_frame(self, new_detections, frame):
        """
        在新帧中追踪人物
        返回最佳匹配和相似度分数
        """
        if self.tracked_person_id is None:
            return None, 0

        best_match = None
        best_score = 0
        best_features = None

        # 对每个检测到的人物计算相似度
        for det in new_detections:
            x1, y1, x2, y2, conf, cls_id = det

            # 裁剪人物图像
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            # 提取特征
            features = self.extractor.extract_all_features(person_img)

            # 与追踪目标的特征比较
            score = self.comparator.compute_similarity(self.tracked_features, features)

            # 记录所有候选
            candidate = {
                'bbox': (x1, y1, x2, y2),
                'features': features,
                'score': score,
                'angle': features['face_angle'][0].value,
                'conf': conf
            }

            if score > best_score:
                best_score = score
                best_match = candidate
                best_features = features

        # 如果相似度超过阈值，认为是同一个人
        if best_score >= self.similarity_threshold and best_match:
            # 更新追踪特征（可以平滑更新）
            self.tracked_features = self.smooth_update(
                self.tracked_features,
                best_features,
                alpha=0.3
            )

            # 记录追踪历史
            self.tracking_history.append({
                'timestamp': datetime.now().isoformat(),
                'score': best_score,
                'angle': best_match['angle']
            })

            return best_match, best_score

        return None, best_score

    def smooth_update(self, old_features, new_features, alpha=0.3):
        """
        平滑更新特征（避免突变）
        alpha: 新特征的权重
        """
        smoothed = {}

        for key in old_features:
            if key == 'face_angle':
                # 角度特殊处理
                old_angle, old_conf = old_features[key]
                new_angle, new_conf = new_features[key]

                if old_angle == new_angle:
                    smoothed[key] = (old_angle, old_conf * (1 - alpha) + new_conf * alpha)
                else:
                    smoothed[key] = new_features[key]

            elif isinstance(old_features[key], np.ndarray):
                smoothed[key] = old_features[key] * (1 - alpha) + new_features[key] * alpha
            else:
                smoothed[key] = new_features[key]

        return smoothed

    def get_tracking_summary(self):
        """
        获取追踪摘要
        """
        if not self.tracking_history:
            return "No tracking data"

        angles = [h['angle'] for h in self.tracking_history]
        angle_counts = {}
        for angle in angles:
            angle_counts[angle] = angle_counts.get(angle, 0) + 1

        summary = f"追踪人物 #{self.tracked_person_id}\n"
        summary += f"角度分布:\n"
        for angle, count in angle_counts.items():
            percentage = count / len(angles) * 100
            summary += f"  - {angle}: {percentage:.1f}% ({count}帧)\n"

        return summary


class TrackingVisualizer:
    """追踪可视化器"""

    @staticmethod
    def draw_tracking_info(frame, best_match, score, tracked_id):
        """
        在画面上绘制追踪信息
        """
        if best_match is None:
            return frame

        x1, y1, x2, y2 = best_match['bbox']

        # 根据相似度设置颜色
        if score > 0.8:
            color = (0, 255, 0)  # 绿色 - 高相似度
        elif score > 0.6:
            color = (0, 255, 255)  # 黄色 - 中等相似度
        else:
            color = (0, 165, 255)  # 橙色 - 低相似度

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # 绘制信息
        info_text = f"Person #{tracked_id} | Score: {score:.2f} | {best_match['angle']}"
        cv2.putText(frame, info_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制置信度
        conf_text = f"conf: {best_match['conf']:.2f}"
        cv2.putText(frame, conf_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        return frame

    @staticmethod
    def draw_angle_distribution(frame, tracking_history):
        """
        绘制角度分布图
        """
        if not tracking_history:
            return frame

        h, w = frame.shape[:2]

        # 统计角度
        angles = [h['angle'] for h in tracking_history[-100:]]  # 最近100帧
        unique_angles = list(set(angles))

        # 创建分布图区域
        chart_x, chart_y = w - 200, h - 150
        chart_w, chart_h = 180, 130

        # 绘制背景
        cv2.rectangle(frame, (chart_x, chart_y),
                      (chart_x + chart_w, chart_y + chart_h),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (chart_x, chart_y),
                      (chart_x + chart_w, chart_y + chart_h),
                      (255, 255, 255), 1)

        # 绘制标题
        cv2.putText(frame, "Angle Distribution",
                    (chart_x + 10, chart_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 绘制角度比例
        y_offset = chart_y + 40
        for angle in unique_angles[:5]:  # 最多显示5个
            count = angles.count(angle)
            percentage = count / len(angles) * 100

            text = f"{angle}: {percentage:.0f}%"
            cv2.putText(frame, text, (chart_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 20

        return frame