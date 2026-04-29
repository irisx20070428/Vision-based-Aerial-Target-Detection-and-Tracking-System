import cv2

points = []


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("image", img)

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2
            h, w = img.shape[:2]
            center_x, center_y = w // 2, h // 2

            dx = box_cx - center_x
            dy = box_cy - center_y

            print(f"\n方框中心: ({box_cx}, {box_cy})")
            print(f"图像中心: ({center_x}, {center_y})")
            print(f"水平距离: {dx:+d} px  (目标中心在图像中心的 {'右侧' if dx > 0 else '左侧'})")
            print(f"垂直距离: {dy:+d} px  (目标中心在图像中心的 {'下方' if dy > 0 else '上方'})")
            print(f"欧氏距离: {(dx ** 2 + dy ** 2) ** 0.5:.1f} px")

            cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.line(img, (box_cx, box_cy), (center_x, center_y), (255, 0, 0), 2)
            cv2.imshow("image", img)

            points = []


img_path = "test_image/test5.2/top_right_2.jpg"  # 请替换为实际路径
img = cv2.imread(img_path)
if img is None:
    print("无法加载图片，请检查路径")
    exit()

cv2.imshow("image", img)
cv2.setMouseCallback("image", mouse_callback)
print("请依次点击检测框的左上角和右下角（两个点）")
cv2.waitKey(0)
cv2.destroyAllWindows()