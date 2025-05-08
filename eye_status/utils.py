import cv2
import torch
import numpy as np
import mediapipe as mp
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

# Initialize Mediapipe Face Mesh
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
except Exception as e:
    print(f"Mediapipe initialization failed: {str(e)}")
    print("Ensure mediapipe is correctly installed")
    exit(1)

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

class EyeDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")
        with open(label_file, 'r') as f:
            lines = f.readlines()
        self.samples = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) != 2:
                print(f"Warning: Invalid label file line, skipping: {line.strip()}")
                continue
            img_name, label = parts
            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found, skipping: {img_path}")
                continue
            try:
                float_label = float(label)
                if not 0.0 <= float_label <= 1.0:
                    print(f"Warning: Invalid label value (must be 0.0~1.0), skipping: {line.strip()}")
                    continue
                self.samples.append((img_name, float_label))
            except ValueError:
                print(f"Warning: Label cannot be converted to float, skipping: {line.strip()}")
                continue
        if not self.samples:
            raise ValueError("No valid samples found in label file")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        try:
            image = Image.open(os.path.join(self.image_dir, img_name)).convert('L')
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(float(label), dtype=torch.float32)
            return image, label
        except Exception as e:
            print(f"Error loading {img_name}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

def infer_openness(model, image_path):
    """Predict average eye openness for a single person's image"""
    try:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return 0.0
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            print(f"No face detected: {image_path}")
            return 0.0
        face = results.multi_face_landmarks[0]
        ih, iw = image.shape[:2]
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])
        eye_images = []
        for eye_idx in [RIGHT_EYE_IDX, LEFT_EYE_IDX]:
            points = np.array([
                [int(face.landmark[idx].x * iw), int(face.landmark[idx].y * ih)]
                for idx in eye_idx
            ])
            x, y, w, h = cv2.boundingRect(points)
            margin = int(0.2 * max(w, h))
            x, y, w, h = x - margin, y - margin, w + 2 * margin, h + 2 * margin
            x, y = max(x, 0), max(y, 0)
            eye_crop = image[y:y+h, x:x+w]
            if eye_crop.size == 0:
                continue
            gray = cv2.cvtColor(cv2.resize(eye_crop, (48, 48)), cv2.COLOR_BGR2GRAY)
            img = transform(Image.fromarray(gray)).unsqueeze(0)
            eye_images.append(img)
        if not eye_images:
            print(f"No valid eye regions extracted: {image_path}")
            return 0.0
        eye_images = torch.cat(eye_images).to(device)
        with torch.no_grad():
            scores = model(eye_images).squeeze(1).cpu().numpy()
        return round(np.mean(scores) * 100, 1)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return 0.0