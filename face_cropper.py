import cv2
from retinaface import RetinaFace

def detect_and_crop_faces(img_path):

    # 讀取圖片
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"無法讀取圖片：{img_path}")

    # 偵測圖片中的人臉並儲存
    results = RetinaFace.detect_faces(img_path)
    face_imgs, face_boxes = [], []

    # 迭代每個人臉
    for key in results:
        face = results[key]
        x1, y1, x2, y2 = face['facial_area']
        face_crop = img_bgr[y1:y2, x1:x2]
        face_imgs.append(face_crop)
        face_boxes.append((x1, y1, x2, y2))

    # 回傳裁切後的人臉圖片和 bounding box
    return face_imgs, face_boxes
