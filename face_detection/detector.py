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

    # # 新增一張副本圖片，用於標示編號
    # img_with_labels = img_bgr.copy()

    # 迭代每個人臉
    for idx, key in enumerate(results):
        face = results[key]
        x1, y1, x2, y2 = face['facial_area']
        face_crop = img_bgr[y1:y2, x1:x2]
        face_imgs.append(face_crop)
        face_boxes.append((x1, y1, x2, y2))

    #     # 在副本圖片上畫出人臉邊框並標示編號
    #     cv2.rectangle(img_with_labels, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 綠色框
    #     cv2.putText(img_with_labels, f'Face {idx}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # # 儲存標示編號的圖片
    # labeled_img_path = img_path.replace(".jpg", "_labeled.jpg")  # 例如：test.jpg -> test_labeled.jpg
    # cv2.imwrite(labeled_img_path, img_with_labels)
    # print(f"標示編號的圖片已儲存為：{labeled_img_path}")

    # 回傳裁切後的人臉圖片和 bounding box
    return face_imgs, face_boxes
