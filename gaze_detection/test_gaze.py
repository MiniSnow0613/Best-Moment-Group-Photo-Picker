import cv2
from gaze import initialize_face_mesh, predict_gaze_score

# 讀取圖片
img = cv2.imread('image.png')

# 檢查圖片是否成功讀取
if img is None:
    print("Failed to load image")
else:
    # 計算並打印 Gaze Score
    face_mesh = initialize_face_mesh()
    score = predict_gaze_score(img, face_mesh)
    print("Gaze Score:", score)
