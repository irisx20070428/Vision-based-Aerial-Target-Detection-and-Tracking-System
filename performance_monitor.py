# performance_monitor.py
import time
import threading
from collections import deque


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.detect_times = deque(maxlen=30)
        self.track_times = deque(maxlen=30)
        self.fps_values = deque(maxlen=30)
        self.running = True

    def record_detect(self, duration_ms):
        self.detect_times.append(duration_ms)

    def record_track(self, duration_ms):
        self.track_times.append(duration_ms)

    def record_fps(self, fps):
        self.fps_values.append(fps)

    def get_stats(self):
        if not self.detect_times:
            return "No data"

        avg_detect = sum(self.detect_times) / len(self.detect_times)
        avg_track = sum(self.track_times) / len(self.track_times) if self.track_times else 0
        avg_fps = sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0

        return f"Detect:{avg_detect:.1f}ms Track:{avg_track:.1f}ms FPS:{avg_fps:.1f}"

    def start_monitoring(self, interval=5):
        """定期打印性能统计"""

        def monitor_loop():
            while self.running:
                time.sleep(interval)
                print(f"📊 性能: {self.get_stats()}")

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()