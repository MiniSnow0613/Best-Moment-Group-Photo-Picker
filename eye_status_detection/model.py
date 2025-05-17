import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from torchvision import transforms
from PIL import Image
import os
import logging
import time
from utils.stderr_utils import suppress_stderr_keywords

suppress_stderr_keywords([
    "inference_feedback_manager.cc",
    "landmark_projection_calculator.cc",
    "All log messages before absl::InitializeLog()"
])

# Suppress TensorFlow Lite and Mediapipe logs
os.environ["GLOG_minloglevel"] = "2"  # Suppress INFO and WARNING logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow Lite logs (0=ALL, 1=INFO, 2=WARNING, 3=ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Model definition
class EyeOpennessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 12 * 12, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Initialize Mediapipe Face Mesh
def initialize_face_mesh():
    """Initialize Mediapipe Face Mesh and measure initialization time"""
    try:
        start_time = time.perf_counter()
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        init_time_ms = (time.perf_counter() - start_time) * 1000
        return face_mesh, init_time_ms
    except Exception as e:
        print(f"Mediapipe initialization failed: {str(e)}")
        print("Ensure mediapipe is correctly installed")
        raise

# Load pre-trained model
def load_model(model_path):
    """Load the pre-trained eye openness model and measure loading time"""
    try:
        start_time = time.perf_counter()
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EyeOpennessCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        load_time_ms = (time.perf_counter() - start_time) * 1000
        print(f"Loaded model from {model_path}")
        return model, load_time_ms
    except Exception as e:
        print(f"Failed to load model {model_path}: {str(e)}")
        raise

# Inference function for a single image
def infer_openness(model, image_array, face_mesh, image_name='input_image'):
    """Predict average eye openness for a single person's image (numpy array) and measure inference time"""
    try:
        start_time = time.perf_counter()
        if not isinstance(image_array, np.ndarray) or image_array.ndim != 3 or image_array.shape[2] != 3:
            print("Invalid input: Expected a 3-channel BGR numpy array")
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            return {'image_name': image_name, 'score': 0.0, 'inference_time_ms': inference_time_ms}
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ih, iw = image_array.shape[:2]
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            print(f"No face detected in image: {image_name}")
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            return {'image_name': image_name, 'score': 0.0, 'inference_time_ms': inference_time_ms}
        face = results.multi_face_landmarks[0]
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])
        eye_images = []
        for eye_idx in [[33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]]:  # RIGHT_EYE_IDX, LEFT_EYE_IDX
            points = np.array([
                [int(face.landmark[idx].x * iw), int(face.landmark[idx].y * ih)]
                for idx in eye_idx
            ])
            x, y, w, h = cv2.boundingRect(points)
            margin = int(0.2 * max(w, h))
            x, y, w, h = x - margin, y - margin, w + 2 * margin, h + 2 * margin
            x, y = max(x, 0), max(y, 0)
            eye_crop = image_array[y:y+h, x:x+w]
            if eye_crop.size == 0:
                continue
            gray = cv2.cvtColor(cv2.resize(eye_crop, (48, 48)), cv2.COLOR_BGR2GRAY)
            img = transform(Image.fromarray(gray)).unsqueeze(0)
            eye_images.append(img)
        if not eye_images:
            print(f"No valid eye regions extracted in image: {image_name}")
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            return {'image_name': image_name, 'score': 0.0, 'inference_time_ms': inference_time_ms}
        eye_images = torch.cat(eye_images).to(device)
        with torch.no_grad():
            scores = model(eye_images).squeeze(1).cpu().numpy()
        score = round(np.mean(scores) * 100, 1)
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        return {'image_name': image_name, 'score': score, 'inference_time_ms': inference_time_ms}
    except Exception as e:
        print(f"Error processing image {image_name}: {str(e)}")
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        return {'image_name': image_name, 'score': 0.0, 'inference_time_ms': inference_time_ms}

# Predict eye openness (combines initialization, model loading, and inference)
def predict_eye_openness(image_array, model_path='eye_openness_model.pth', image_name='input_image'):
    """Initialize Mediapipe, load model, predict eye openness, and measure times"""
    try:
        face_mesh, init_time_ms = initialize_face_mesh()
        model, load_time_ms = load_model(model_path)
        result = infer_openness(model, image_array, face_mesh, image_name)
        result['init_time_ms'] = init_time_ms
        result['load_time_ms'] = load_time_ms
        return result
    except Exception as e:
        print(f"Prediction failed for image {image_name}: {str(e)}")
        return {'image_name': image_name, 'score': 0.0, 'init_time_ms': 0.0, 'load_time_ms': 0.0, 'inference_time_ms': 0.0}