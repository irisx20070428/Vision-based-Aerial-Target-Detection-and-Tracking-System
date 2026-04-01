import cv2
import numpy as np
import os
import json
from datetime import datetime
from enum import Enum
from config import Config


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
    """人物特征提取器 - 树莓派优化版"""

    def __init__(self):
        # 根据配置选择是否使用简化特征
        self.use_simplified = Config.USE_SIMPLIFIED_FEATURES

        if self.use_simplified:
            # 简化特征配置（树莓派优化）
            self.hist_bins = [4, 4, 4]  # 减少bin数
            self.use_hog = False
            self.weights = {
                'color_hist': 0.6,  # 提高颜色权重
                'edge': 0.2,
                'texture': 0.2
            }
        else:
            # 完整特征配置
            self.hist_bins = [8, 8, 8]
            self.use_hog = True
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.weights = {
                'color_hist': 0.3,
                'hog': 0.3,
                'edge': 0.2,
                'texture': 0.2
            }

    def extract_color_histogram(self, person_img):
        """提取颜色直方图特征"""
        # 缩小图像加速
        if self.use_simplified:
            small_img = cv2.resize(person_img, (32, 32))
        else:
            small_img = cv2.resize(person_img, (64, 64))

        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)

        # 计算3D直方图
        hist = cv2.calcHist([hsv], [0, 1, 2], None,
                            self.hist_bins, [0, 180, 0, 256, 0, 256])

        # 归一化
        cv2.normalize(hist, hist)
        return hist.flatten()

    def extract_hog_features(self, person_img):
        """提取HOG特征（如果启用）"""
        if not self.use_hog:
            return np.array([])

        resized = cv2.resize(person_img, (64, 128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hog_features = self.hog.compute(gray)
        return hog_features.flatten()

    def extract_edge_features(self, person_img):
        """提取边缘特征 - 优化版"""
        gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)

        # 缩小图像
        small = cv2.resize(gray, (32, 32))

        if self.use_simplified:
            # 简化版：只返回边缘密度
            edges = cv2.Canny(small, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            return np.array([edge_density])
        else:
            # 完整版：包含方向直方图
            edges = cv2.Canny(small, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            sobelx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            direction = np.arctan2(sobely, sobelx) * 180 / np.pi

            hist_dir, _ = np.histogram(direction[magnitude > 50],
                                       bins=18, range=(-180, 180))
            hist_dir = hist_dir / (np.sum(hist_dir) + 1e-6)

            return np.concatenate([[edge_density], hist_dir])

    def extract_texture_features(self, person_img):
        """提取纹理特征 - 优化版"""
        gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)

        # 缩小图像
        small = cv2.resize(gray, (32, 32))

        # 简化LBP
        lbp = np.zeros_like(small)
        for i in range(1, small.shape[0] - 1):
            for j in range(1, small.shape[1] - 1):
                center = small[i, j]
                code = 0
                if self.use_simplified:
                    # 4方向LBP（更快）
                    code |= (small[i - 1, j] > center) << 0
                    code |= (small[i, j + 1] > center) << 1
                    code |= (small[i + 1, j] > center) << 2
                    code |= (small[i, j - 1] > center) << 3
                else:
                    # 8方向LBP
                    code |= (small[i - 1, j - 1] > center) << 7
                    code |= (small[i - 1, j] > center) << 6
                    code |= (small[i - 1, j + 1] > center) << 5
                    code |= (small[i, j + 1] > center) << 4
                    code |= (small[i + 1, j + 1] > center) << 3
                    code |= (small[i + 1, j] > center) << 2
                    code |= (small[i + 1, j - 1] > center) << 1
                    code |= (small[i, j - 1] > center) << 0
                lbp[i, j] = code

        # LBP直方图（减少bin数）
        bin_num = 16 if self.use_simplified else 256
        hist_lbp = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [bin_num], [0, bin_num])
        cv2.normalize(hist_lbp, hist_lbp)

        return hist_lbp.flatten()

    def estimate_face_angle(self, person_img):
        """估计人脸角度"""
        h, w = person_img.shape[:2]
        aspect_ratio = w / h

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
        features = {
            'color_hist': self.extract_color_histogram(person_img),
            'edge': self.extract_edge_features(person_img),
            'texture': self.extract_texture_features(person_img),
            'face_angle': self.estimate_face_angle(person_img)
        }

        # 如果启用HOG，添加HOG特征
        if self.use_hog:
            hog_features = self.extract_hog_features(person_img)
            if len(hog_features) > 0:
                features['hog'] = hog_features

        return features


class FeatureComparator:
    """特征比较器 - 计算相似度"""

    def __init__(self):
        self.extractor = PersonFeatureExtractor()

    def compute_similarity(self, features1, features2):
        """计算两组特征的相似度"""
        scores = {}

        # 颜色直方图相似度
        if 'color_hist' in features1 and 'color_hist' in features2:
            scores['color_hist'] = cv2.compareHist(
                features1['color_hist'].reshape(-1, 1).astype(np.float32),
                features2['color_hist'].reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )

        # HOG特征相似度（如果存在）
        if 'hog' in features1 and 'hog' in features2 and len(features1['hog']) > 0:
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
        self.dataset = {}
        self.next_id = 0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(f"{save_dir}/images")
            os.makedirs(f"{save_dir}/features")

    def add_person_sample(self, person_img, features, metadata=None):
        """添加人物样本到数据集"""
        person_id = self.next_id
        self.next_id += 1

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
                'edge': features['edge'].tolist(),
                'texture': features['texture'].tolist(),
                'face_angle': [features['face_angle'][0].value, features['face_angle'][1]]
            },
            'metadata': metadata or {},
            'image_path': img_filename
        }

        # 如果有HOG特征，也保存
        if 'hog' in features and len(features['hog']) > 0:
            feature_data['features']['hog'] = features['hog'].tolist()

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
        """在数据集中查找最佳匹配"""
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

    def __init__(self, similarity_threshold=None):
        if similarity_threshold is None:
            similarity_threshold = Config.SIMILARITY_THRESHOLD

        self.extractor = PersonFeatureExtractor()
        self.comparator = FeatureComparator()
        self.dataset = FeatureDataset()

        self.tracked_person_id = None
        self.tracked_features = None
        self.similarity_threshold = similarity_threshold
        self.tracking_history = []

    def select_person(self, person_img, person_bbox, frame_metadata=None):
        """选择人物开始追踪"""
        features = self.extractor.extract_all_features(person_img)

        metadata = {
            'bbox': person_bbox,
            'frame_metadata': frame_metadata
        }
        person_id = self.dataset.add_person_sample(person_img, features, metadata)

        self.tracked_person_id = person_id
        self.tracked_features = features

        print(f"\n🎯 开始追踪人物 #{person_id}")
        print(f"   初始角度: {features['face_angle'][0].value}")

        return person_id

    def track_in_new_frame(self, new_detections, frame):
        """在新帧中追踪人物"""
        if self.tracked_person_id is None:
            return None, 0

        best_match = None
        best_score = 0
        best_features = None

        for det in new_detections:
            x1, y1, x2, y2, conf, cls_id = det

            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            features = self.extractor.extract_all_features(person_img)
            score = self.comparator.compute_similarity(self.tracked_features, features)

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

        if best_score >= self.similarity_threshold and best_match:
            self.tracked_features = self.smooth_update(
                self.tracked_features,
                best_features,
                alpha=0.3
            )

            self.tracking_history.append({
                'timestamp': datetime.now().isoformat(),
                'score': best_score,
                'angle': best_match['angle']
            })

            # 限制历史记录长度
            if len(self.tracking_history) > Config.TRACKING_HISTORY_LEN:
                self.tracking_history = self.tracking_history[-Config.TRACKING_HISTORY_LEN:]

            return best_match, best_score

        return None, best_score

    def smooth_update(self, old_features, new_features, alpha=0.3):
        """平滑更新特征"""
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

        angles = [h['angle'] for h in self.tracking_history]
        angle_counts = {}
        for angle in angles:
            angle_counts[angle] = angle_counts.get(angle, 0) + 1

        summary = f"追踪人物 #{self.tracked_person_id}\n"
        summary += f"总帧数: {len(angles)}\n"
        summary += f"角度分布:\n"
        for angle, count in angle_counts.items():
            percentage = count / len(angles) * 100
            summary += f"  - {angle}: {percentage:.1f}% ({count}帧)\n"

        return summary


class TrackingVisualizer:
    """追踪可视化器"""

    @staticmethod
    def draw_tracking_info(frame, best_match, score, tracked_id):
        """在画面上绘制追踪信息"""
        if best_match is None:
            return frame

        x1, y1, x2, y2 = best_match['bbox']

        if score > 0.8:
            color = (0, 255, 0)
        elif score > 0.6:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        info_text = f"Person #{tracked_id} | Score: {score:.2f} | {best_match['angle']}"
        cv2.putText(frame, info_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        conf_text = f"conf: {best_match['conf']:.2f}"
        cv2.putText(frame, conf_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        return frame

    @staticmethod
    def draw_angle_distribution(frame, tracking_history):
        """绘制角度分布图"""
        if not tracking_history:
            return frame

        h, w = frame.shape[:2]
        angles = [h['angle'] for h in tracking_history[-100:]]
        unique_angles = list(set(angles))

        chart_x, chart_y = w - 200, h - 150
        chart_w, chart_h = 180, 130

        cv2.rectangle(frame, (chart_x, chart_y),
                      (chart_x + chart_w, chart_y + chart_h),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (chart_x, chart_y),
                      (chart_x + chart_w, chart_y + chart_h),
                      (255, 255, 255), 1)

        cv2.putText(frame, "Angle Distribution",
                    (chart_x + 10, chart_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        y_offset = chart_y + 40
        for angle in unique_angles[:5]:
            count = angles.count(angle)
            percentage = count / len(angles) * 100
            text = f"{angle}: {percentage:.0f}%"
            cv2.putText(frame, text, (chart_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 20

        return frame