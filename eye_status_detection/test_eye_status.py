import cv2
from eye_status_detection.model import predict_eye_openness
from utils.mediapipe_utils import initialize_face_mesh

# 放在最外層資料夾進行測試
if __name__ == '__main__':
    image = "image.png"
    image_array = cv2.imread(image)
    face_mesh, _ = initialize_face_mesh()
    print("\nPredicting eye openness for image")
    result = predict_eye_openness(image_array, 'models/eye_openness_model.pth', image_name='input_image')
    print(f"Mediapipe Face Mesh initialization time: {result['init_time_ms']:.2f} ms")
    print(f"Model loading time: {result['load_time_ms']:.2f} ms")
    print(f"Inference time: {result['inference_time_ms']:.2f} ms")
    print(f"Image {result['image_name']} eye openness: {result['score']:.1f}%")