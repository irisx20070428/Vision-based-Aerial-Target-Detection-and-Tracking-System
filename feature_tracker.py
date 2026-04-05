# feature_tracker.py (最终版 - 支持 IoU 辅助匹配和 prev_bbox)
import cv2
import numpy as np
import os
import json
import threading
from datetime import datetime
from enum import Enum
from collections import Counter


class FaceAngle(Enum):
    FRONT = "front"
    SIDE = "side"
    PROFILE = "profile"
    UP = "up"
    DOWN = "down"
    BACK = "back"
    UNKNOWN = "unknown"


class PersonFeatureExtractor:
    def __init__(self):
        self.hist_bins = [8, 8, 8]
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.weights = {
            'color_hist': 0.6,
            'hog': 0.0,
            'edge': 0.2,
            'texture': 0.2
        }

    def extract_color_histogram(self, person_img):
        hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, self.hist_bins, [0,180,0,256,0,256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def extract_hog_features(self, person_img):
        resized = cv2.resize(person_img, (64,128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hog_features = self.hog.compute(gray)
        return hog_features.flatten()

    def extract_edge_features(self, person_img):
        gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        hist_dir, _ = np.histogram(direction[magnitude > 50], bins=36, range=(-180,180))
        hist_dir = hist_dir / (np.sum(hist_dir) + 1e-6)
        return np.concatenate([[edge_density], hist_dir])

    def extract_texture_features(self, person_img):
        gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
        lbp = np.zeros_like(gray)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i,j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp[i,j] = code
        hist_lbp = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [256], [0,256])
        cv2.normalize(hist_lbp, hist_lbp)
        return hist_lbp.flatten()

    def estimate_face_angle(self, person_img):
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
        features = {
            'color_hist': self.extract_color_histogram(person_img),
            'hog': np.array([]),
            'edge': self.extract_edge_features(person_img),
            'texture': self.extract_texture_features(person_img),
            'face_angle': self.estimate_face_angle(person_img)
        }
        return features


class FeatureComparator:
    def __init__(self):
        self.extractor = PersonFeatureExtractor()

    def compute_similarity(self, features1, features2):
        scores = {}
        if 'color_hist' in features1 and 'color_hist' in features2:
            scores['color_hist'] = cv2.compareHist(
                features1['color_hist'].reshape(-1,1).astype(np.float32),
                features2['color_hist'].reshape(-1,1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )
        if 'hog' in features1 and 'hog' in features2 and len(features1['hog']) > 0 and len(features2['hog']) > 0:
            hog1 = features1['hog'].flatten()
            hog2 = features2['hog'].flatten()
            norm1 = np.linalg.norm(hog1)
            norm2 = np.linalg.norm(hog2)
            scores['hog'] = np.dot(hog1, hog2) / (norm1 * norm2 + 1e-6)
        if 'edge' in features1 and 'edge' in features2:
            edge1 = features1['edge'].flatten()
            edge2 = features2['edge'].flatten()
            scores['edge'] = 1 - np.linalg.norm(edge1 - edge2) / (np.linalg.norm(edge1) + np.linalg.norm(edge2) + 1e-6)
        if 'texture' in features1 and 'texture' in features2:
            tex1 = features1['texture'].flatten()
            tex2 = features2['texture'].flatten()
            scores['texture'] = 1 - np.linalg.norm(tex1 - tex2) / (np.linalg.norm(tex1) + np.linalg.norm(tex2) + 1e-6)
        if 'face_angle' in features1 and 'face_angle' in features2:
            angle1, conf1 = features1['face_angle']
            angle2, conf2 = features2['face_angle']
            scores['face_angle'] = conf1 * conf2 if angle1 == angle2 else 0

        weights = self.extractor.weights
        total_score = 0
        total_weight = 0
        for key, score in scores.items():
            if key in weights:
                total_score += score * weights[key]
                total_weight += weights[key]
        return total_score / total_weight if total_weight > 0 else 0


class FeatureDataset:
    def __init__(self, save_dir="person_dataset"):
        import threading
        self._lock = threading.Lock()
        self.save_dir = save_dir
        self.dataset = {}
        self.next_id = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(f"{save_dir}/images")
            os.makedirs(f"{save_dir}/features")

    def add_person_sample(self, person_img, features, metadata=None):
        """异步添加人物样本（不阻塞主线程）"""
        person_id = self.next_id
        self.next_id += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 准备要保存的数据（复制一份，避免后续修改影响）
        img_copy = person_img.copy()
        features_copy = {
            'color_hist': features['color_hist'].copy(),
            'hog': features['hog'].copy(),
            'edge': features['edge'].copy(),
            'texture': features['texture'].copy(),
            'face_angle': features['face_angle']  # 元组不可变，直接引用
        }

        # 定义实际保存的函数（将在新线程中运行）
        def _save():
            # 保存图像
            img_filename = f"{self.save_dir}/images/person_{person_id}_{timestamp}.jpg"
            cv2.imwrite(img_filename, img_copy)

            # 保存特征 JSON
            feature_data = {
                'person_id': person_id,
                'timestamp': timestamp,
                'features': {
                    'color_hist': features_copy['color_hist'].tolist(),
                    'hog': features_copy['hog'].tolist(),
                    'edge': features_copy['edge'].tolist(),
                    'texture': features_copy['texture'].tolist(),
                    'face_angle': [features_copy['face_angle'][0].value, features_copy['face_angle'][1]]
                },
                'metadata': metadata or {},
                'image_path': img_filename
            }
            with open(f"{self.save_dir}/features/person_{person_id}_{timestamp}.json", 'w') as f:
                json.dump(feature_data, f, indent=2)

            # 更新内存中的数据集（注意线程安全，这里简单加锁）
            with self._lock:
                if person_id not in self.dataset:
                    self.dataset[person_id] = []
                self.dataset[person_id].append({
                    'features': features_copy,
                    'timestamp': timestamp,
                    'image_path': img_filename,
                    'metadata': metadata
                })

            print(f"✅ 已保存人物 #{person_id} 的样本 (角度: {features_copy['face_angle'][0].value})")

        # 启动后台线程执行保存，不等待
        thread = threading.Thread(target=_save, daemon=True)
        thread.start()

        return person_id

    def find_best_match(self, query_features, comparator):
        best_person_id = None
        best_score = 0
        for person_id, samples in self.dataset.items():
            for sample in samples:
                score = comparator.compute_similarity(query_features, sample['features'])
                if score > best_score:
                    best_score = score
                    best_person_id = person_id
        return best_person_id, best_score


class SmartTracker:
    def __init__(self, similarity_threshold=0.4):
        self.extractor = PersonFeatureExtractor()
        self.comparator = FeatureComparator()
        self.dataset = FeatureDataset()
        self.tracked_person_id = None
        self.tracked_features = None
        self.similarity_threshold = similarity_threshold
        self.tracking_history = []

    def select_person(self, person_img, person_bbox, frame_metadata=None):
        features = self.extractor.extract_all_features(person_img)
        metadata = {'bbox': person_bbox, 'frame_metadata': frame_metadata}
        person_id = self.dataset.add_person_sample(person_img, features, metadata)
        self.tracked_person_id = person_id
        self.tracked_features = features
        print(f"\n🎯 开始追踪人物 #{person_id}")
        return person_id

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def track_in_new_frame(self, new_detections, frame, prev_bbox=None):


        if self.tracked_person_id is None:
            return None, 0

        if len(new_detections) > 3:
            # 按与 prev_bbox 的中心距离排序，取前3个
            if prev_bbox is not None:
                cx_prev = (prev_bbox[0] + prev_bbox[2]) / 2
                cy_prev = (prev_bbox[1] + prev_bbox[3]) / 2
                dets_with_dist = []
                for det in new_detections:
                    cx = (det[0] + det[2]) / 2
                    cy = (det[1] + det[3]) / 2
                    dist = (cx - cx_prev) ** 2 + (cy - cy_prev) ** 2
                    dets_with_dist.append((dist, det))
                dets_with_dist.sort(key=lambda x: x[0])
                new_detections = [det for _, det in dets_with_dist[:2]]


        best_match = None
        best_score = 0
        best_features = None

        for det in new_detections:
            x1, y1, x2, y2, conf, _ = det
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue
            features = self.extractor.extract_all_features(person_img)
            score = self.comparator.compute_similarity(self.tracked_features, features)
            if prev_bbox is not None:
                iou = self._compute_iou(prev_bbox, (x1, y1, x2, y2))
                combined = 0.7 * score + 0.3 * iou
            else:
                combined = score
            candidate = {
                'bbox': (x1, y1, x2, y2),
                'features': features,
                'score': score,
                'combined': combined,
                'angle': features['face_angle'][0].value,
                'conf': conf
            }
            if combined > best_score:
                best_score = combined
                best_match = candidate
                best_features = features

        if best_match and best_match['combined'] >= self.similarity_threshold:
            self.tracked_features = self.smooth_update(self.tracked_features, best_features, alpha=0.3)
            self.tracking_history.append({
                'timestamp': datetime.now().isoformat(),
                'score': best_match['score'],
                'angle': best_match['angle']
            })
            return best_match, best_match['score']
        else:
            return None, best_score

    def smooth_update(self, old_features, new_features, alpha=0.3):
        smoothed = {}
        for key in old_features:
            if key == 'face_angle':
                old_angle, old_conf = old_features[key]
                new_angle, new_conf = new_features[key]
                if old_angle == new_angle:
                    smoothed[key] = (old_angle, old_conf*(1-alpha) + new_conf*alpha)
                else:
                    smoothed[key] = new_features[key]
            elif isinstance(old_features[key], np.ndarray):
                smoothed[key] = old_features[key] * (1-alpha) + new_features[key] * alpha
            else:
                smoothed[key] = new_features[key]
        return smoothed

    def get_tracking_summary(self):
        if not self.tracking_history:
            return "No tracking data"
        angles = [h['angle'] for h in self.tracking_history]
        angle_counts = Counter(angles)
        summary = f"追踪人物 #{self.tracked_person_id}\n"
        for angle, count in angle_counts.items():
            percentage = count / len(angles) * 100
            summary += f"  - {angle}: {percentage:.1f}% ({count}帧)\n"
        return summary


class TrackingVisualizer:
    @staticmethod
    def draw_tracking_info(frame, best_match, score, tracked_id):
        if best_match is None:
            return frame
        x1, y1, x2, y2 = best_match['bbox']
        if score > 0.8:
            color = (0,255,0)
        elif score > 0.6:
            color = (0,255,255)
        else:
            color = (0,165,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        info_text = f"Person #{tracked_id} | Score: {score:.2f} | {best_match['angle']}"
        cv2.putText(frame, info_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        conf_text = f"conf: {best_match['conf']:.2f}"
        cv2.putText(frame, conf_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        center_x = (x1+x2)//2
        center_y = (y1+y2)//2
        cv2.circle(frame, (center_x, center_y), 5, (0,0,255), -1)
        return frame

    @staticmethod
    def draw_angle_distribution(frame, tracking_history):
        if not tracking_history:
            return frame
        h, w = frame.shape[:2]
        angles = [h['angle'] for h in tracking_history[-100:]]
        unique_angles = list(set(angles))
        chart_x, chart_y = w-200, h-150
        chart_w, chart_h = 180, 130
        cv2.rectangle(frame, (chart_x, chart_y), (chart_x+chart_w, chart_y+chart_h), (0,0,0), -1)
        cv2.rectangle(frame, (chart_x, chart_y), (chart_x+chart_w, chart_y+chart_h), (255,255,255), 1)
        cv2.putText(frame, "Angle Distribution", (chart_x+10, chart_y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        y_offset = chart_y + 40
        for angle in unique_angles[:5]:
            count = angles.count(angle)
            percentage = count / len(angles) * 100
            text = f"{angle}: {percentage:.0f}%"
            cv2.putText(frame, text, (chart_x+10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            y_offset += 20
        return frame