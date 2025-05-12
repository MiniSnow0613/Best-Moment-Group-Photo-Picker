import cv2
import numpy as np
from ultralytics import YOLO

def score_photo(image_path, area_threshold_ratio=0.3):

    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片: {image_path}")
    height, width, _ = img.shape

    # 載入 YOLOv8 模型
    model = YOLO('yolov8m.pt')

    # 偵測人（class_id = 0）
    results = model.predict(image_path, classes=[0], verbose=False)

    # 收集所有 bounding boxes 和面積
    boxes = []
    areas = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        box_area = (x2 - x1) * (y2 - y1)
        boxes.append((x1, y1, x2, y2))
        areas.append(box_area)

    # 如果沒有偵測到人
    if not areas:
        print("沒有偵測到人物。")
        return 0

    # 找出最大面積
    max_area = max(areas)

    # 只保留主要人物，畫 bounding box [綠色框]
    filtered_boxes = []
    for (box, area) in zip(boxes, areas):
        if area >= max_area * area_threshold_ratio:
            filtered_boxes.append(box)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # 取得整體人物邊界(最大最小xy)
    x_min = min(box[0] for box in filtered_boxes)
    y_min = min(box[1] for box in filtered_boxes)
    x_max = max(box[2] for box in filtered_boxes)
    y_max = max(box[3] for box in filtered_boxes)

    # 畫整體 bounding box [紫色框]
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

    # 計算總邊界中心 [黃色點]
    total_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
    cv2.circle(img, total_center, 5, (0, 255, 255), -1)

    # 最佳構圖位置(照片中心，高度為1/3位置處) [藍色點]
    photo_center = (width // 2, 2 * height // 3)
    cv2.circle(img, photo_center, 5, (255, 0, 0), -1)

    # 中心距離分數
    center_distance = np.linalg.norm(np.array(total_center) - np.array(photo_center))
    score = max(0, 100 - (center_distance / max(width, height)) * 100)

    # 顯示資訊
    # print(f"圖片: {image_path}")
    # print(f"構圖分數: {score:.2f}")

    # # 顯示圖片
    # cv2.imshow("Result", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return score
