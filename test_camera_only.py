
# test_camera_only.py
import cv2
import sys

print("测试摄像头...")

# 测试 ID 0
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ 摄像头 ID 0 可用")
    ret, frame = cap.read()
    if ret:
        print(f"   - 分辨率: {frame.shape[1]} x {frame.shape[0]}")
    cap.release()
else:
    print("❌ 摄像头 ID 0 不可用")

    # 测试 ID 1
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        print("✅ 摄像头 ID 1 可用")
        ret, frame = cap.read()
        if ret:
            print(f"   - 分辨率: {frame.shape[1]} x {frame.shape[0]}")
        cap.release()
    else:
        print("❌ 摄像头 ID 1 也不可用")
        print("请检查摄像头连接")