# enhanced_tracker.py - 增强版追踪器（运行时数据记录，不保存文件）
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from enum import Enum


class FaceAngle(Enum):
    FRONT = "front"
    SIDE = "side"
    PROFILE = "profile"
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


class PersonFeatureExtractor:
    """人物特征提取器"""

    def __init__(self):
        self.hist_bins = [8, 8, 8]
        self.weights = {
            'color_hist': 0.35,
            'edge': 0.25,
            'texture': 0.25,
            'face_angle': 0.15
        }

        # 加载人脸检测器用于角度估计
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            self.face_cascade = None

    def extract_color_histogram(self, person_img):
        if person_img is None or person_img.size == 0:
            return np.zeros(512)
        try:
            hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, self.hist_bins, [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except:
            return np.zeros(512)

    def extract_edge_features(self, person_img):
        if person_img is None or person_img.size == 0:
            return np.zeros(9)
        try:
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (32, 32))
            edges = cv2.Canny(small, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            sobelx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            direction = np.arctan2(sobely, sobelx) * 180 / np.pi

            hist_dir, _ = np.histogram(direction[magnitude > 20], bins=8, range=(-180, 180))
            hist_dir = hist_dir / (np.sum(hist_dir) + 1e-6)

            return np.concatenate([[edge_density], hist_dir])
        except:
            return np.zeros(9)

    def extract_texture_features(self, person_img):
        if person_img is None or person_img.size == 0:
            return np.zeros(16)
        try:
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (32, 32))

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
        except:
            return np.zeros(16)

    def estimate_face_angle(self, person_img):
        """估计人脸角度 - 正脸/侧脸/仰视/俯视"""
        if person_img is None or person_img.size == 0:
            return FaceAngle.UNKNOWN, 0.3

        h, w = person_img.shape[:2]
        aspect_ratio = w / h if h > 0 else 0.5

        # 尝试用 Haar Cascade 检测正脸
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    face_ratio = fw / fh if fh > 0 else 1

                    # 根据人脸在图像中的位置判断仰视/俯视
                    face_y_center = fy + fh / 2
                    image_center_y = h / 2

                    if face_y_center < image_center_y * 0.6:
                        return FaceAngle.UP, 0.75
                    elif face_y_center > image_center_y * 1.4:
                        return FaceAngle.DOWN, 0.75
                    elif face_ratio > 0.7:
                        return FaceAngle.FRONT, 0.85
                    elif face_ratio > 0.5:
                        return FaceAngle.PROFILE, 0.7
                    else:
                        return FaceAngle.SIDE, 0.6
            except:
                pass

        # 根据宽高比判断（备用方案）
        if aspect_ratio > 0.85:
            return FaceAngle.FRONT, 0.7
        elif aspect_ratio > 0.65:
            return FaceAngle.PROFILE, 0.6
        elif aspect_ratio > 0.45:
            return FaceAngle.SIDE, 0.55
        else:
            return FaceAngle.UNKNOWN, 0.4

    def extract_all_features(self, person_img):
        if person_img is None or person_img.size == 0:
            return None
        try:
            features = {
                'color_hist': self.extract_color_histogram(person_img),
                'edge': self.extract_edge_features(person_img),
                'texture': self.extract_texture_features(person_img),
                'face_angle': self.estimate_face_angle(person_img)
            }
            return features
        except:
            return None


class MotionPredictor:
    """运动预测器 - 预测人物下一帧位置"""

    def __init__(self, max_history=5):
        self.positions = deque(maxlen=max_history)
        self.velocities = deque(maxlen=max_history)
        self.predicted_center = None

    def update(self, center_x, center_y):
        """更新位置并预测下一帧"""
        if len(self.positions) > 0:
            last_x, last_y = self.positions[-1]
            vx = center_x - last_x
            vy = center_y - last_y
            self.velocities.append((vx, vy))

        self.positions.append((center_x, center_y))

        # 计算平均速度
        if len(self.velocities) > 0:
            avg_vx = sum(v[0] for v in self.velocities) / len(self.velocities)
            avg_vy = sum(v[1] for v in self.velocities) / len(self.velocities)
            self.predicted_center = (int(center_x + avg_vx), int(center_y + avg_vy))
        else:
            self.predicted_center = (center_x, center_y)

        return self.predicted_center

    def predict(self):
        """返回预测位置"""
        if self.predicted_center:
            return self.predicted_center
        elif len(self.positions) > 0:
            return self.positions[-1]
        return None


class EnhancedSmartTracker:
    """增强版智能追踪器 - 带运动预测、实时数据记录、智能找回"""

    def __init__(self, similarity_threshold=0.45):
        self.extractor = PersonFeatureExtractor()
        self.motion_predictor = MotionPredictor()

        self.tracked_person_id = None
        self.tracked_features = None
        self.similarity_threshold = similarity_threshold

        # 特征库 - 存储不同角度的特征
        self.feature_library = {}  # {angle: features}

        # 追踪数据 - 只在运行时使用，不保存文件
        self.tracking_data = {
            'frame_count': 0,
            'history': [],  # 每帧数据
            'angle_stats': {},  # 角度统计
            'scores': []  # 相似度分数
        }

        # 追踪状态
        self.last_bbox = None
        self.lost_counter = 0
        self.max_lost_frames = 15
        self.frame_count = 0

        # 调试输出控制
        self.last_print_frame = 0

    def select_person(self, person_img, person_bbox, frame_metadata=None):
        """选择人物开始追踪"""
        if person_img is None or person_img.size == 0:
            return None

        features = self.extractor.extract_all_features(person_img)
        if features is None:
            return None

        self.tracked_person_id = id(person_img)  # 简单ID
        self.tracked_features = features

        # 初始化特征库
        angle = features['face_angle'][0].value
        self.feature_library = {angle: features.copy()}

        self.last_bbox = person_bbox
        self.lost_counter = 0
        self.frame_count = 0

        # 初始化运动预测器
        center_x = (person_bbox[0] + person_bbox[2]) // 2
        center_y = (person_bbox[1] + person_bbox[3]) // 2
        self.motion_predictor.update(center_x, center_y)

        # 重置追踪数据
        self.tracking_data = {
            'frame_count': 0,
            'history': [],
            'angle_stats': {},
            'scores': []
        }

        self._record_data('select', features, 1.0, person_bbox)

        print(f"\n🎯 开始追踪人物")
        print(f"   初始角度: {angle}")
        print(f"   相似度阈值: {self.similarity_threshold}")

        return self.tracked_person_id

    def track_in_new_frame(self, new_detections, frame):
        """在新帧中追踪人物 - 带预测和找回"""
        if self.tracked_person_id is None:
            return None, 0

        self.frame_count += 1

        # 1. 预测目标位置
        predicted_center = self.motion_predictor.predict()
        predicted_bbox = None
        if predicted_center and self.last_bbox:
            w = self.last_bbox[2] - self.last_bbox[0]
            h = self.last_bbox[3] - self.last_bbox[1]
            predicted_bbox = (
                max(0, predicted_center[0] - w // 2),
                max(0, predicted_center[1] - h // 2),
                predicted_center[0] + w // 2,
                predicted_center[1] + h // 2
            )

        # 2. 评估所有检测框
        candidates = []
        for det in new_detections:
            x1, y1, x2, y2, conf, cls_id = det
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            features = self.extractor.extract_all_features(person_img)
            if features is None:
                continue

            # 特征相似度
            feature_score = self._compute_similarity(self.tracked_features, features)

            # 位置相似度（基于预测）
            location_score = 0
            if predicted_bbox:
                location_score = self._compute_iou(predicted_bbox, (x1, y1, x2, y2))

            # 综合得分
            combined_score = feature_score * 0.7 + location_score * 0.3

            candidates.append({
                'bbox': (x1, y1, x2, y2),
                'features': features,
                'feature_score': feature_score,
                'combined_score': combined_score,
                'angle': features['face_angle'][0].value,
                'angle_conf': features['face_angle'][1],
                'conf': conf
            })

        # 3. 选择最佳匹配
        best_match = None
        best_score = 0

        if candidates:
            candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            best_match = candidates[0]
            best_score = best_match['combined_score']

        # 4. 判断追踪状态
        if best_match and best_score >= self.similarity_threshold:
            # 追踪成功
            self._update_tracking(best_match)
            self.lost_counter = 0
            self._record_data('tracking', best_match['features'], best_score, best_match['bbox'])

            # 每30帧打印状态
            if self.frame_count - self.last_print_frame >= 30:
                self._print_status(best_match['feature_score'])
                self.last_print_frame = self.frame_count

            return best_match, best_match['feature_score']

        elif best_match and best_score >= self.similarity_threshold * 0.7:
            # 低置信度追踪
            self._update_tracking(best_match)
            self.lost_counter += 1
            self._record_data('low', best_match['features'], best_score, best_match['bbox'])

            if self.lost_counter % 5 == 1:
                print(f"⚠️ 低置信度: {best_score:.2f}")

            return best_match, best_match['feature_score']

        else:
            # 追踪丢失，尝试找回
            self.lost_counter += 1
            self._record_data('lost', None, best_score, None)

            if self.lost_counter % 5 == 1:
                print(f"⚠️ 追踪丢失 ({self.lost_counter}/{self.max_lost_frames})")

            # 尝试找回
            if self.lost_counter <= self.max_lost_frames:
                recovered = self._attempt_recovery(candidates)
                if recovered:
                    print(f"✅ 找回目标!")
                    self._update_tracking(recovered)
                    self.lost_counter = 0
                    self._record_data('recovered', recovered['features'], recovered['feature_score'], recovered['bbox'])
                    return recovered, recovered['feature_score']

            # 使用预测框（如果可用）
            if predicted_bbox:
                return {'bbox': predicted_bbox, 'feature_score': best_score, 'angle': 'predicted',
                        'conf': 0}, best_score

            return None, best_score

    def _compute_similarity(self, features1, features2):
        """计算特征相似度"""
        if features1 is None or features2 is None:
            return 0.5

        scores = {}

        # 颜色直方图
        if 'color_hist' in features1 and 'color_hist' in features2:
            try:
                hist1 = features1['color_hist'].reshape(-1, 1).astype(np.float32)
                hist2 = features2['color_hist'].reshape(-1, 1).astype(np.float32)
                scores['color_hist'] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                scores['color_hist'] = max(0, min(1, scores['color_hist']))
            except:
                scores['color_hist'] = 0.5

        # 边缘特征
        if 'edge' in features1 and 'edge' in features2:
            try:
                edge1 = features1['edge'].flatten()
                edge2 = features2['edge'].flatten()
                diff = np.linalg.norm(edge1 - edge2)
                scores['edge'] = 1 - min(1, diff / (np.linalg.norm(edge1) + np.linalg.norm(edge2) + 1e-6))
            except:
                scores['edge'] = 0.5

        # 纹理特征
        if 'texture' in features1 and 'texture' in features2:
            try:
                tex1 = features1['texture'].flatten()
                tex2 = features2['texture'].flatten()
                diff = np.linalg.norm(tex1 - tex2)
                scores['texture'] = 1 - min(1, diff / (np.linalg.norm(tex1) + np.linalg.norm(tex2) + 1e-6))
            except:
                scores['texture'] = 0.5

        # 角度匹配
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

        return total_score / total_weight if total_weight > 0 else 0.5

    def _compute_iou(self, bbox1, bbox2):
        """计算IOU"""
        if bbox1 is None or bbox2 is None:
            return 0

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)

        if x2 <= x1 or y2 <= y1:
            return 0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _update_tracking(self, match):
        """更新追踪状态"""
        if 'features' in match:
            # 平滑更新特征
            self.tracked_features = self._smooth_update(
                self.tracked_features, match['features'], alpha=0.2
            )

            # 更新特征库（保存不同角度）
            angle = match['angle']
            self.feature_library[angle] = match['features'].copy()

        # 更新位置和运动预测
        self.last_bbox = match['bbox']
        center_x = (match['bbox'][0] + match['bbox'][2]) // 2
        center_y = (match['bbox'][1] + match['bbox'][3]) // 2
        self.motion_predictor.update(center_x, center_y)

    def _attempt_recovery(self, candidates):
        """尝试找回丢失的目标"""
        if not candidates:
            return None

        best_recovery = None
        best_score = 0

        for candidate in candidates:
            # 与特征库中的所有角度进行匹配
            max_score = 0
            for angle, saved_features in self.feature_library.items():
                score = self._compute_similarity(saved_features, candidate['features'])
                max_score = max(max_score, score)

            # 位置约束
            if self.last_bbox:
                iou = self._compute_iou(self.last_bbox, candidate['bbox'])
                max_score = max_score * 0.8 + iou * 0.2

            if max_score > best_score and max_score > self.similarity_threshold * 0.7:
                best_score = max_score
                best_recovery = candidate

        return best_recovery

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

    def _record_data(self, status, features, score, bbox):
        """记录追踪数据到内存"""
        data = {
            'frame': self.frame_count,
            'status': status,
            'score': round(float(score), 3),
            'lost_counter': self.lost_counter
        }

        if features:
            angle, conf = features['face_angle']
            data['face_angle'] = angle.value
            data['angle_confidence'] = round(float(conf), 3)

            # 更新角度统计
            angle_name = angle.value
            if angle_name not in self.tracking_data['angle_stats']:
                self.tracking_data['angle_stats'][angle_name] = 0
            self.tracking_data['angle_stats'][angle_name] += 1

        if bbox:
            data['bbox'] = bbox

        self.tracking_data['history'].append(data)
        self.tracking_data['scores'].append(score)
        self.tracking_data['frame_count'] = self.frame_count

        # 限制历史记录长度（只保留最近1000帧）
        if len(self.tracking_data['history']) > 1000:
            self.tracking_data['history'] = self.tracking_data['history'][-1000:]
            self.tracking_data['scores'] = self.tracking_data['scores'][-1000:]

    def _print_status(self, score):
        """打印追踪状态"""
        recent = self.tracking_data['history'][-30:]
        if not recent:
            return

        avg_score = sum(d['score'] for d in recent) / len(recent)

        # 统计最近30帧的角度
        angles = [d.get('face_angle', 'unknown') for d in recent if 'face_angle' in d]
        if angles:
            from collections import Counter
            angle_counts = Counter(angles)
            main_angle = angle_counts.most_common(1)[0][0]
        else:
            main_angle = "unknown"

        print(f"📊 追踪 | 分数:{score:.2f} | 平均:{avg_score:.2f} | 角度:{main_angle} | 丢失:{self.lost_counter}")

    def draw_tracking_info(self, frame, best_match, score):
        """在画面上绘制追踪信息"""
        if best_match is None:
            return frame

        x1, y1, x2, y2 = best_match['bbox']

        # 根据分数选择颜色
        if score > 0.7:
            color = (0, 255, 0)  # 绿色 - 高置信度
        elif score > 0.5:
            color = (0, 255, 255)  # 黄色 - 中等
        else:
            color = (0, 165, 255)  # 橙色 - 低置信度

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # 绘制预测方向（如果有运动）
        predicted = self.motion_predictor.predict()
        if predicted:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.arrowedLine(frame, (center_x, center_y), predicted, (255, 0, 0), 2, tipLength=0.3)

        # 获取当前角度
        angle = best_match.get('angle', 'unknown')

        # 显示信息
        info_text = f"Score: {score:.2f} | {angle}"
        cv2.putText(frame, info_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        return frame

    def draw_angle_chart(self, frame):
        """在画面上绘制角度分布图"""
        if not self.tracking_data['history']:
            return frame

        h, w = frame.shape[:2]

        # 统计最近100帧的角度
        recent = self.tracking_data['history'][-100:]
        angles = [d.get('face_angle', 'unknown') for d in recent if 'face_angle' in d]

        if not angles:
            return frame

        from collections import Counter
        angle_counts = Counter(angles)

        # 绘制图表区域
        chart_x, chart_y = w - 180, h - 140
        chart_w, chart_h = 170, 130

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
        for angle, count in angle_counts.most_common(4):
            percentage = count / len(angles) * 100
            text = f"{angle}: {percentage:.0f}%"
            cv2.putText(frame, text, (chart_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            y_offset += 18

        return frame

    def get_summary(self):
        """获取追踪摘要"""
        if not self.tracking_data['history']:
            return "无追踪数据"

        total = len(self.tracking_data['history'])
        tracking = sum(1 for d in self.tracking_data['history'] if d['status'] == 'tracking')
        recovered = sum(1 for d in self.tracking_data['history'] if d['status'] == 'recovered')
        lost = sum(1 for d in self.tracking_data['history'] if d['status'] == 'lost')

        scores = [d['score'] for d in self.tracking_data['history']]
        avg_score = sum(scores) / len(scores) if scores else 0

        summary = f"""
📊 追踪摘要
{'=' * 30}
总帧数: {total}
成功追踪: {tracking} ({tracking / total * 100:.1f}%)
找回次数: {recovered}
丢失帧数: {lost}
平均相似度: {avg_score:.3f}

角度分布:
"""
        for angle, count in self.tracking_data['angle_stats'].items():
            pct = count / total * 100
            summary += f"  {angle}: {count} ({pct:.1f}%)\n"

        return summary