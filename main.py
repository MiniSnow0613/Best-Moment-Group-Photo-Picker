from face_detection.detector import detect_and_crop_faces
from weights_config.config import get_face_weights_gui
from smile_detection.model import predict_smile_probs, predict_smile_ratio
from composition_analysis.analyzer import score_photo
from eye_status_detection.model import predict_eye_openness

img_path = "003.jpg" # 換成要計算的照片
faces, _ = detect_and_crop_faces(img_path) # 人臉圖片
weights = get_face_weights_gui(faces) # 權重

total_smile_weighted_score = 0
total_eye_weighted_score = 0
total_weight = 0

for i, (face_img, w) in enumerate(zip(faces, weights)):
    print(f"\n== Face {i} ==")
    print(f"weight: {w}")

    # smile_detection
    # smile_probs = predict_smile_probs(face_img)
    # print("每種類別機率：")
    # for k, v in smile_probs.items():
    #     print(f"{k}: {v:.2%}")
    smile_score = 100 * predict_smile_ratio(face_img)
    print(f"笑容分數：{smile_score:.2f}")

    # eye_status_detection
    # eye_score = predict_eye_openness(face_img)
    # print(f"閉眼分數：{eye_score['score']:.2f}")

    # blur_detection

    # gaze_detection
    print(f"視線分數：")

    total_smile_weighted_score += smile_score * w
    # total_eye_weighted_score += eye_score['score'] * w
    total_weight += w

total_smile_score = total_smile_weighted_score / total_weight if total_weight > 0 else 0
# total_eye_score = total_eye_weighted_score / total_weight if total_weight > 0 else 0

composition_score = score_photo(img_path, area_threshold_ratio=0)
print(f"\n構圖分數: {composition_score:.2f}")

final_score = total_smile_score * 0.8 + composition_score * 0.2
# final_score = total_smile_score * 0.4 + total_eye_score * 0.4 + composition_score * 0.2
print(f"\n總分: {final_score:.2f}")
