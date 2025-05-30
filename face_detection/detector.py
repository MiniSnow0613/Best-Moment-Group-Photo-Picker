import os
import cv2
from retinaface import RetinaFace

def detect_and_crop_faces(img_path, label_save_dir=None):
    # 讀取圖片
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"無法讀取圖片：{img_path}")

    # 偵測圖片中的人臉並儲存
    results = RetinaFace.detect_faces(img_path)
    face_imgs, face_boxes = [], []

    # 新增一張副本圖片，用於標示編號
    img_with_labels = img_bgr.copy()

    # 迭代每個人臉
    for idx, key in enumerate(results):
        face = results[key]
        x1, y1, x2, y2 = face['facial_area']
        face_crop = img_bgr[y1:y2, x1:x2]
        face_imgs.append(face_crop)
        face_boxes.append((x1, y1, x2, y2))

        # 在副本圖片上畫出人臉邊框並標示編號
        cv2.rectangle(img_with_labels, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 綠色框
        cv2.putText(img_with_labels, f'Face {idx}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 如果指定了資料夾就儲存標示圖片
    if label_save_dir is not None:
        os.makedirs(label_save_dir, exist_ok=True)
        base_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(base_name)[0]
        labeled_img_path = os.path.join(label_save_dir, f"{name_no_ext}_labeled.jpg")
        cv2.imwrite(labeled_img_path, img_with_labels)
        print(f"標示編號的圖片已儲存為：{labeled_img_path}")

    # 回傳裁切後的人臉圖片和 bounding box
    return face_imgs, face_boxes
