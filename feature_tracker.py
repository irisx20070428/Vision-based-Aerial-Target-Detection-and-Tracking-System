# feature_tracker.py - 修复 calcHist 错误
import cv2
import numpy as np
import os
import json
from datetime import datetime
from enum import Enum
from config import Config


class FaceAngle(Enum):
    FRONT = "front"
    SIDE = "side"
    PROFILE = "profile"
    UP = "up"
    DOWN = "down"
    BACK = "back"
    UNKNOWN = "unknown"


class PersonFeatureExtractor:
    """人物特征提取器 - 修复版"""

    def __init__(self):
        self.use_simplified = getattr(Config, 'USE_SIMPLIFIED_FEATURES', False)

        # 使用正确的hist参数
        self.hist_bins = [8, 8, 8]  # 3通道各8个bin
        self.weights = {
            'color_hist': 0.5,
            'edge': 0.3,
            'texture': 0.2
        }

    def extract_color_histogram(self, person_img):
        """提取颜色直方图 - 修复calcHist错误"""
        if person_img is None or person_img.size == 0:
            return np.zeros(512)  # 8*8*8 = 512

        try:
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)

            # 正确的calcHist参数格式
            # 对于3通道图像，需要传入3个通道的列表
            hist = cv2.calcHist(
                [hsv],  # 图像列表
                [0, 1, 2],  # 使用的通道 [H, S, V]
                None,  # 不使用mask
                self.hist_bins,  # bin数量 [8, 8, 8]
                [0, 180, 0, 256, 0, 256]  # 范围 [H:0-180, S:0-256, V:0-256]
            )

            # 归一化
            cv2.normalize(hist, hist)
            return hist.flatten()

        except Exception as e:
            print(f"颜色直方图提取错误: {e}")
            return np.zeros(512)

    def extract_edge_features(self, person_img):
        """提取边缘特征"""
        if person_img is None or person_img.size == 0:
            return np.array([0])

        try:
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            return np.array([edge_density])
        except Exception as e:
            print(f"边缘特征提取错误: {e}")
            return np.array([0])

    def extract_texture_features(self, person_img):
        """提取纹理特征"""
        if person_img is None or person_img.size == 0:
            return np.zeros(16)

        try:
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64))

            # LBP特征
            lbp = np.zeros_like(small)
            for i in range(1, small.shape[0] - 1):
                for j in range(1, small.shape[1] - 1):
                    center = small[i, j]
                    code = 0
                    code |= (small[i - 1, j] > center) << 0
                    code |= (small[i, j + 1] > center) << 1
                    code |= (small[i + 1, j] > center) << 2
                    code |= (small[i, j - 1] > center) << 3
                    lbp[i, j] = code

            hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [16], [0, 16])
            cv2.normalize(hist, hist)
            return hist.flatten()

        except Exception as e:
            print(f"纹理特征提取错误: {e}")
            return np.zeros(16)

    def estimate_face_angle(self, person_img):
        """估计人脸角度"""
        if person_img is None or person_img.size == 0:
            return FaceAngle.UNKNOWN, 0.3

        h, w = person_img.shape[:2]
        aspect_ratio = w / h if h > 0 else 0.5

        if aspect_ratio > 0.8:
            return FaceAngle.FRONT, 0.7
        elif aspect_ratio < 0.4:
            return FaceAngle.SIDE, 0.6
        elif aspect_ratio < 0.6:
            return FaceAngle.PROFILE, 0.5
        else:
            return FaceAngle.UNKNOWN, 0.3

    def extract_all_features(self, person_img):
        """提取所有特征"""
        if person_img is None or person_img.size == 0:
            print("人物图像为空，无法提取特征")
            return None

        try:
            features = {
                'color_hist': self.extract_color_histogram(person_img),
                'edge': self.extract_edge_features(person_img),
                'texture': self.extract_texture_features(person_img),
                'face_angle': self.estimate_face_angle(person_img)
            }
            return features
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None


class FeatureComparator:
    """特征比较器"""

    def __init__(self):
        self.extractor = PersonFeatureExtractor()

    def compute_similarity(self, features1, features2):
        """计算相似度"""
        if features1 is None or features2 is None:
            return 0.5  # 默认中等分数

        scores = {}

        # 颜色直方图相似度
        if 'color_hist' in features1 and 'color_hist' in features2:
            try:
                hist1 = features1['color_hist'].reshape(-1, 1).astype(np.float32)
                hist2 = features2['color_hist'].reshape(-1, 1).astype(np.float32)
                scores['color_hist'] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                scores['color_hist'] = max(0, min(1, scores['color_hist']))
            except:
                scores['color_hist'] = 0.5

        # 边缘特征相似度
        if 'edge' in features1 and 'edge' in features2:
            try:
                edge1 = features1['edge'].flatten()
                edge2 = features2['edge'].flatten()
                diff = np.linalg.norm(edge1 - edge2)
                scores['edge'] = 1 - min(1, diff / (np.linalg.norm(edge1) + np.linalg.norm(edge2) + 1e-6))
            except:
                scores['edge'] = 0.5

        # 纹理特征相似度
        if 'texture' in features1 and 'texture' in features2:
            try:
                texture1 = features1['texture'].flatten()
                texture2 = features2['texture'].flatten()
                diff = np.linalg.norm(texture1 - texture2)
                scores['texture'] = 1 - min(1, diff / (np.linalg.norm(texture1) + np.linalg.norm(texture2) + 1e-6))
            except:
                scores['texture'] = 0.5

        # 角度相似度
        if 'face_angle' in features1 and 'face_angle' in features2:
            angle1, conf1 = features1['face_angle']
            angle2, conf2 = features2['face_angle']
            if angle1 == angle2:
                scores['face_angle'] = conf1 * conf2
            else:
                scores['face_angle'] = 0.3

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
        return 0.5


class FeatureDataset:
    """特征数据集"""

    def __init__(self, save_dir="person_dataset"):
        self.save_dir = save_dir
        self.dataset = {}
        self.next_id = 0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(f"{save_dir}/images", exist_ok=True)
            os.makedirs(f"{save_dir}/features", exist_ok=True)

    def add_person_sample(self, person_img, features, metadata=None):
        """添加样本"""
        if features is None or person_img is None:
            return None

        person_id = self.next_id
        self.next_id += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存图像
        try:
            img_filename = f"{self.save_dir}/images/person_{person_id}_{timestamp}.jpg"
            cv2.imwrite(img_filename, person_img)
        except:
            pass

        # 添加到内存
        if person_id not in self.dataset:
            self.dataset[person_id] = []

        self.dataset[person_id].append({
            'features': features,
            'timestamp': timestamp,
            'image_path': img_filename,
            'metadata': metadata
        })

        print(f"✅ 已保存人物 #{person_id}")
        return person_id

    def find_best_match(self, query_features, comparator):
        """查找最佳匹配"""
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
    """智能追踪器 - 修复版"""

    def __init__(self, similarity_threshold=None):
        if similarity_threshold is None:
            similarity_threshold = getattr(Config, 'SIMILARITY_THRESHOLD', 0.5)

        self.extractor = PersonFeatureExtractor()
        self.comparator = FeatureComparator()
        self.dataset = FeatureDataset()

        self.tracked_person_id = None
        self.tracked_features = None
        self.similarity_threshold = similarity_threshold
        self.tracking_history = []

        # 追踪平滑
        self.last_bbox = None
        self.last_score = 0
        self.lost_counter = 0
        self.max_lost_frames = 10

    def select_person(self, person_img, person_bbox, frame_metadata=None):
        """选择人物"""
        if person_img is None or person_img.size == 0:
            print("人物图像为空，无法选择")
            return None

        features = self.extractor.extract_all_features(person_img)
        if features is None:
            print("特征提取失败，无法选择")
            return None

        metadata = {'bbox': person_bbox, 'frame_metadata': frame_metadata}
        person_id = self.dataset.add_person_sample(person_img, features, metadata)

        if person_id is None:
            return None

        self.tracked_person_id = person_id
        self.tracked_features = features
        self.last_bbox = person_bbox
        self.lost_counter = 0

        print(f"\n🎯 开始追踪人物 #{person_id}")
        print(f"   相似度阈值: {self.similarity_threshold}")

        return person_id

    def _calculate_iou(self, bbox1, bbox2):
        """计算IOU"""
        if bbox1 is None or bbox2 is None:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 交集
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # 并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def track_in_new_frame(self, new_detections, frame):
        """在新帧中追踪"""
        if self.tracked_person_id is None:
            return None, 0

        # 如果没有检测到任何人
        if len(new_detections) == 0:
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                print(f"⚠️ 人物 #{self.tracked_person_id} 追踪丢失")
            return None, 0

        best_match = None
        best_score = 0
        best_features = None

        for det in new_detections:
            x1, y1, x2, y2, conf, cls_id = det

            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            # 提取特征
            features = self.extractor.extract_all_features(person_img)
            if features is None:
                continue

            # 特征相似度
            score = self.comparator.compute_similarity(self.tracked_features, features)

            # 位置重叠度
            iou = self._calculate_iou(self.last_bbox, (x1, y1, x2, y2))

            # 综合得分
            combined_score = score * 0.7 + iou * 0.3

            candidate = {
                'bbox': (x1, y1, x2, y2),
                'features': features,
                'score': score,
                'combined_score': combined_score,
                'angle': features['face_angle'][0].value,
                'conf': conf
            }

            if combined_score > best_score:
                best_score = combined_score
                best_match = candidate
                best_features = features

        # 判断是否匹配成功
        if best_score >= self.similarity_threshold and best_match:
            # 更新特征
            if best_features:
                self.tracked_features = self._smooth_update(
                    self.tracked_features,
                    best_features,
                    alpha=0.2
                )

            self.last_bbox = best_match['bbox']
            self.last_score = best_match['score']
            self.lost_counter = 0

            # 记录历史
            self.tracking_history.append({
                'timestamp': datetime.now().isoformat(),
                'score': best_match['score'],
                'angle': best_match['angle']
            })

            # 限制历史长度
            if len(self.tracking_history) > 100:
                self.tracking_history = self.tracking_history[-100:]

            return best_match, best_match['score']
        else:
            self.lost_counter += 1
            return None, best_score

    def _smooth_update(self, old_features, new_features, alpha=0.2):
        """平滑更新特征"""
        if old_features is None or new_features is None:
            return new_features

        smoothed = {}

        for key in old_features:
            if key == 'face_angle':
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
        """获取追踪摘要"""
        if not self.tracking_history:
            return "No tracking data"

        avg_score = sum(h['score'] for h in self.tracking_history) / len(self.tracking_history)
        angles = [h['angle'] for h in self.tracking_history]
        angle_counts = {}
        for angle in angles:
            angle_counts[angle] = angle_counts.get(angle, 0) + 1

        summary = f"追踪人物 #{self.tracked_person_id}\n"
        summary += f"平均分数: {avg_score:.2f}\n"
        summary += f"总帧数: {len(angles)}\n"
        summary += f"角度分布:\n"
        for angle, count in angle_counts.items():
            percentage = count / len(angles) * 100
            summary += f"  - {angle}: {percentage:.1f}%\n"

        return summary


class TrackingVisualizer:
    """追踪可视化器"""

    @staticmethod
    def draw_tracking_info(frame, best_match, score, tracked_id):
        if best_match is None:
            return frame

        x1, y1, x2, y2 = best_match['bbox']

        if score > 0.7:
            color = (0, 255, 0)
        elif score > 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        info_text = f"ID:{tracked_id} | {score:.2f}"
        cv2.putText(frame, info_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        return frame

    @staticmethod
    def draw_angle_distribution(frame, tracking_history):
        if not tracking_history:
            return frame

        h, w = frame.shape[:2]
        angles = [h['angle'] for h in tracking_history[-50:]]
        unique_angles = list(set(angles))

        chart_x, chart_y = w - 150, h - 120
        chart_w, chart_h = 140, 100

        cv2.rectangle(frame, (chart_x, chart_y),
                      (chart_x + chart_w, chart_y + chart_h),
                      (0, 0, 0), -1)

        y_offset = chart_y + 20
        for angle in unique_angles[:3]:
            count = angles.count(angle)
            percentage = count / len(angles) * 100
            text = f"{angle}: {percentage:.0f}%"
            cv2.putText(frame, text, (chart_x + 5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            y_offset += 15

        return frame