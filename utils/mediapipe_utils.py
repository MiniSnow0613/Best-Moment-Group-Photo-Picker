import mediapipe as mp
import time

def initialize_face_mesh(static_mode=True, max_faces=1, refine=True, min_confidence=0.5):
    start = time.perf_counter()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=static_mode,
        max_num_faces=max_faces,
        refine_landmarks=refine,
        min_detection_confidence=min_confidence
    )
    elapsed = (time.perf_counter() - start) * 1000
    return face_mesh, elapsed
