from face_detection.detector import detect_and_crop_faces
from weights_config.config import get_face_weights_gui
from smile_detection.model import predict_smile_ratio
from composition_analysis.analyzer import score_photo
from utils.mediapipe_utils import initialize_face_mesh
from eye_status_detection.model import load_model, infer_openness
from gaze_detection.gaze import predict_gaze_score

img_path = "002.jpg" # 換成要計算的照片
faces, _ = detect_and_crop_faces(img_path) # 人臉圖片
weights = get_face_weights_gui(faces) # 權重

face_mesh, _ = initialize_face_mesh()
eye_model, _ = load_model('models/eye_openness_model.pth')

total_smile_weighted_score = 0
total_eye_weighted_score = 0
total_gaze_weighted_score = 0
total_weight = 0

for i, (face_img, w) in enumerate(zip(faces, weights)):
    print(f"\n== Face {i} ==")
    print(f"weight: {w}")

    # smile_detection
    smile_score = 100 * predict_smile_ratio(face_img)
    print(f"笑容分數：{smile_score:.2f}")

    # eye_status_detection
    eye_score = infer_openness(eye_model, face_img, face_mesh, image_name=f"face_{i}")
    print(f"閉眼分數：{eye_score['score']:.2f}")

    # blur_detection

    # gaze_detection
    gaze_score = 100 * predict_gaze_score(face_img, face_mesh)
    print(f"視線分數：{gaze_score:.2f}")

    total_smile_weighted_score += smile_score * w
    total_eye_weighted_score += eye_score['score'] * w
    total_gaze_weighted_score += gaze_score * w
    total_weight += w

total_smile_score = total_smile_weighted_score / total_weight if total_weight > 0 else 0
total_gaze_score = total_gaze_weighted_score / total_weight if total_weight > 0 else 0
total_eye_score = total_eye_weighted_score / total_weight if total_weight > 0 else 0

composition_score = score_photo(img_path, area_threshold_ratio=0)
print(f"\n構圖分數: {composition_score:.2f}")

final_score = total_smile_score * 0.25 + total_eye_score * 0.25 + total_gaze_score * 0.25 + composition_score * 0.25
print(f"\n總分: {final_score:.2f}")
