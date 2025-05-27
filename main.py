import os
from face_detection.detector import detect_and_crop_faces
from weights_config.config import get_face_weights_gui
from smile_detection.model import predict_smile_ratio
from composition_analysis.analyzer import score_photo
from utils.mediapipe_utils import initialize_face_mesh
from eye_status_detection.model import load_model, infer_openness
from gaze_detection.gaze import predict_gaze_score
from blur_detection.model import compute_blur_score, blur_score_to_100

def process_single_image(img_path, face_mesh, eye_model):
    faces, _ = detect_and_crop_faces(img_path)
    weights = get_face_weights_gui(faces)

    total_smile_weighted_score = 0
    total_eye_weighted_score = 0
    total_gaze_weighted_score = 0
    total_weight = 0

    for i, (face_img, w) in enumerate(zip(faces, weights)):
        smile_score = 100 * predict_smile_ratio(face_img)
        eye_score = infer_openness(eye_model, face_img, face_mesh, image_name=f"face_{i}")['score']
        gaze_score = 100 * predict_gaze_score(face_img, face_mesh)

        total_smile_weighted_score += smile_score * w
        total_eye_weighted_score += eye_score * w
        total_gaze_weighted_score += gaze_score * w
        total_weight += w

    total_smile_score = total_smile_weighted_score / total_weight if total_weight > 0 else 0
    total_eye_score = total_eye_weighted_score / total_weight if total_weight > 0 else 0
    total_gaze_score = total_gaze_weighted_score / total_weight if total_weight > 0 else 0

    composition_score = score_photo(img_path, area_threshold_ratio=0)
    blur_raw = compute_blur_score(img_path)
    blur_score = blur_score_to_100(blur_raw)

    final_score = (total_smile_score * 0.20 + total_eye_score * 0.40 +
                   total_gaze_score * 0.15 + blur_score * 0.15 +
                   composition_score * 0.10)

    return {
        "img_path": img_path,
        "smile": total_smile_score,
        "eye": total_eye_score,
        "gaze": total_gaze_score,
        "composition": composition_score,
        "blur_raw": blur_raw,
        "blur": blur_score,
        "final": final_score
    }

def batch_process_folder(folder_path):
    face_mesh, _ = initialize_face_mesh()
    eye_model, _ = load_model('models/eye_openness_model.pth')

    results = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    for filename in os.listdir(folder_path):
        if os.path.splitext(filename)[1].lower() in valid_exts:
            img_path = os.path.join(folder_path, filename)
            print(f"處理圖片：{img_path}")
            result = process_single_image(img_path, face_mesh, eye_model)
            results.append(result)
            print(f"完成 {filename}，總分: {result['final']:.2f}")

    return results


if __name__ == "__main__":
    folder_path = "group_photos"
    all_results = batch_process_folder(folder_path)

    for res in all_results:
        print({
            "img_path": res['img_path'],
            "smile": round(res['smile'], 2),
            "eye": round(res['eye'], 2),
            "gaze": round(res['gaze'], 2),
            "composition": round(res['composition'], 2),
            "blur": round(res['blur'], 2),
            "final": round(res['final'], 2),
        })
