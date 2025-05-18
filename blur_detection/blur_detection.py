import cv2
import numpy as np

def compute_blur_score(gray_face):
    laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
    return laplacian.var()

def evaluate_face_blur_array(image, net, conf_threshold=0.5):
    """輸入 image (array) 與已載入的 DNN 模型，輸出人臉模糊度平均分數"""
    if image is None:
        return None

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 [104.0, 177.0, 123.0], False, False)
    net.setInput(blob)
    detections = net.forward()

    scores = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            face_roi = gray[y1:y2, x1:x2]

            if face_roi.size == 0:
                continue

            score = compute_blur_score(face_roi)
            scores.append(score)

    if not scores:
        return None

    return sum(scores) / len(scores)
