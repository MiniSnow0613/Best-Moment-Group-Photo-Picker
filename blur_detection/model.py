import cv2
import numpy as np

def compute_blur_score(image_path):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    score = laplacian.var()
    return round(score, 4)

def blur_score_to_100(score, max_threshold=1500):
    score = min(score, max_threshold)
    return round((score / max_threshold) * 100, 2)
