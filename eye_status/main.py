from utils import predict_eye_openness

# Main execution flow
if __name__ == '__main__':
    # Step 1: Predict eye openness for a single image
    print("\n===== Step 1: Predicting eye openness for image =====")
    result = predict_eye_openness('image.png', 'eye_openness_model.pth')  # Replace with image path
    print(f"Mediapipe Face Mesh initialization time: {result['init_time_ms']:.2f} ms")
    print(f"Model loading time: {result['load_time_ms']:.2f} ms")
    print(f"Inference time: {result['inference_time_ms']:.2f} ms")
    print(f"Image {result['image_name']} eye openness: {result['score']:.1f}%")