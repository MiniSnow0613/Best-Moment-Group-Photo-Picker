import os
import cv2
import numpy as np
import mediapipe as mp

# 初始化 MediaPipe FaceMesh
def initialize_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7
    )
    return face_mesh

# 提取眼睛與瞳孔的 normalized landmark
def extract_eye_landmarks(image, face_mesh):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        print("無法偵測人臉")
        return None

    face = results.multi_face_landmarks[0]
    def get_landmark(idx):
        lm = face.landmark[idx]
        return np.array([lm.x * w, lm.y * h])

    return {
        'left_eye': [get_landmark(33), get_landmark(133)],
        'right_eye': [get_landmark(362), get_landmark(263)],
        'iris_left': [get_landmark(468), get_landmark(469), get_landmark(470), get_landmark(471)],
        'iris_right': [get_landmark(473), get_landmark(474), get_landmark(475), get_landmark(476)],
    }

# 計算注視分數（0~1）
def eye_score(eye, iris):
    eye_center = np.mean(eye, axis=0)
    iris_center = np.mean(iris, axis=0)
    dist = np.linalg.norm(iris_center - eye_center)
    eye_width = np.linalg.norm(eye[0] - eye[1])

    if eye_width == 0:
        return 0.0

    norm_dist = dist / eye_width

    # 定義上下界（調整這兩個值）
    max_score_dist = 0.10  # 完全注視
    min_score_dist = 0.30  # 完全不注視

    if norm_dist <= max_score_dist:
        return 1.0
    elif norm_dist >= min_score_dist:
        return 0.0
    else:
        score = 1.0 - (norm_dist - max_score_dist) / (min_score_dist - max_score_dist)
        return round(score, 4)

# 主函式：輸入圖片，輸出 gaze score
def predict_gaze_score(image, face_mesh):
    landmarks = extract_eye_landmarks(image, face_mesh)
    if landmarks is None:
        return 0.0

    score_L = eye_score(landmarks['left_eye'], landmarks['iris_left'])
    score_R = eye_score(landmarks['right_eye'], landmarks['iris_right'])
    gaze_score = round((score_L + score_R) / 2, 4)
    return gaze_score


if __name__ == "__main__":
    img = cv2.imread('image.png')
    # 檢查圖片是否成功讀取
    if img is None:
        print("Failed to load image")
    else:
        # 計算並打印 Gaze Score
        score = predict_gaze_score(img)
        print("Gaze Score:", score)
